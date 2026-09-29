# setup_env.ps1
# Sets up the ecqr2025 benchmarking environment.
#
# Run this script from the directory where you want your workspace to live.
# It will clone the required repositories into the current directory and
# install them into a managed Python environment named "ecqr_shipped_env".
#
# Usage:
#   cd C:\path\to\your\workspace
#   .\path\to\setup_env.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ENV_NAME      = "ecqr_shipped_env"
$PYTHON_VERSION = "3.10"
$ENV_MANAGER   = $null

$REPOS = @(
    [PSCustomObject]@{ url = "https://github.com/researchingotter1324/ecqr2025sub-bc";     dir = "ecqr2025sub-bc" },
    [PSCustomObject]@{ url = "https://github.com/researchingotter1324/ecqr2025sub-optuna"; dir = "ecqr2025sub-optuna" },
    [PSCustomObject]@{ url = "https://github.com/researchingotter1324/ecqr2025sub-smac";   dir = "ecqr2025sub-smac" },
    [PSCustomObject]@{ url = "https://github.com/researchingotter1324/ecqr2025sub-c";      dir = "ecqr2025sub-c" }
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
function Write-Step { param([string]$msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$msg) Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn { param([string]$msg) Write-Warning $msg }

function Confirm-Overwrite {
    param([string]$Label)
    $ans = Read-Host "    '$Label' already exists. Overwrite? [y/N]"
    return ($ans -match "^[Yy]$")
}

function Test-Command { param([string]$Name) return ($null -ne (Get-Command $Name -ErrorAction SilentlyContinue)) }

function Assert-LastExitCode {
    param([string]$Context)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Context failed (exit code $LASTEXITCODE)."
        exit 1
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Detect environment manager
# ─────────────────────────────────────────────────────────────────────────────
Write-Step "Detecting environment manager"

$uvAvailable    = Test-Command "uv"
$condaAvailable = Test-Command "conda"

if ($uvAvailable) {
    $ENV_MANAGER = "uv"
    Write-OK "uv found — will use uv for environment management."
} elseif ($condaAvailable) {
    Write-Warn "uv not found. Attempting to fall back to conda."
    $ENV_MANAGER = "conda"
    Write-OK "conda found — will use conda for environment management."
} else {
    Write-Error ("Neither 'uv' nor 'conda' is available on this system.`n" +
                 "Please install uv   : https://docs.astral.sh/uv/getting-started/installation/`n" +
                 "or conda/Miniconda  : https://docs.conda.io/en/latest/miniconda.html`n" +
                 "then re-run this script.")
    exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Environment: check / create
# ─────────────────────────────────────────────────────────────────────────────
function Test-EnvExists {
    if ($ENV_MANAGER -eq "uv") {
        return (Test-Path (Join-Path $PWD $ENV_NAME))
    } else {
        $list = conda env list 2>&1
        return ($list | Where-Object { $_ -match "(^|\s)$([regex]::Escape($ENV_NAME))(\s|$)" }).Count -gt 0
    }
}

function Remove-Env {
    Write-Step "Removing existing environment '$ENV_NAME'"
    if ($ENV_MANAGER -eq "uv") {
        Remove-Item -Recurse -Force (Join-Path $PWD $ENV_NAME)
    } else {
        conda env remove -n $ENV_NAME -y
        Assert-LastExitCode "conda env remove"
    }
    Write-OK "Removed."
}

function New-Env {
    Write-Step "Creating environment '$ENV_NAME' (Python $PYTHON_VERSION)"
    if ($ENV_MANAGER -eq "uv") {
        uv venv --python $PYTHON_VERSION (Join-Path $PWD $ENV_NAME)
        Assert-LastExitCode "uv venv"
    } else {
        conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        Assert-LastExitCode "conda create"
    }
    Write-OK "Environment created."
}

Write-Step "Checking for existing environment '$ENV_NAME'"

if (Test-EnvExists) {
    if (Confirm-Overwrite $ENV_NAME) {
        Remove-Env
        New-Env
    } else {
        Write-OK "Keeping existing environment. Dependencies will be installed into it."
    }
} else {
    New-Env
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pip-install helper
# ─────────────────────────────────────────────────────────────────────────────
function Invoke-PipInstall {
    param([string]$InstallPath)
    Write-Step "Installing '$InstallPath' into '$ENV_NAME'"
    if ($ENV_MANAGER -eq "uv") {
        $pythonExe = Join-Path $PWD "$ENV_NAME\Scripts\python.exe"
        uv pip install --python $pythonExe $InstallPath
        Assert-LastExitCode "uv pip install ($InstallPath)"
    } else {
        conda run -n $ENV_NAME pip install $InstallPath
        Assert-LastExitCode "conda pip install ($InstallPath)"
    }
    Write-OK "Installed."
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clone repositories and install
# ─────────────────────────────────────────────────────────────────────────────
foreach ($repo in $REPOS) {
    $repoDir = $repo.dir
    $repoUrl = $repo.url
    $repoPath = Join-Path $PWD $repoDir

    Write-Step "Repository: $repoDir"

    $shouldClone = $true
    $shouldInstall = $true

    if (Test-Path $repoPath) {
        if (Confirm-Overwrite $repoDir) {
            Write-Host "    Removing '$repoDir'..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $repoPath
        } else {
            Write-OK "Skipping clone of '$repoDir'. Will also skip its pip install."
            $shouldClone   = $false
            $shouldInstall = $false
        }
    }

    if ($shouldClone) {
        Write-Host "    Cloning $repoUrl ..." -ForegroundColor Yellow
        git clone $repoUrl $repoPath
        Assert-LastExitCode "git clone $repoUrl"
        Write-OK "Cloned."
    }

    # SMAC: SWIG is required to build the pyrfr C extension.
    if ($repoDir -eq "ecqr2025sub-smac" -and $shouldInstall) {
        Write-Step "Installing SWIG (required for SMAC's pyrfr dependency)"
        if ($ENV_MANAGER -eq "uv") {
            if ($condaAvailable) {
                Write-Warn ("uv does not manage system packages. Installing SWIG via conda into the base " +
                            "environment so it is accessible on PATH during the build.")
                conda install swig -y
                if ($LASTEXITCODE -ne 0) {
                    Write-Warn ("conda install swig failed. SMAC may not build correctly.`n" +
                                "You can install SWIG manually from https://www.swig.org/download.html " +
                                "and ensure it is on your PATH, then re-run pip install for ecqr2025sub-smac.")
                } else {
                    Write-OK "SWIG installed."
                }
            } else {
                Write-Warn ("SWIG must be available on PATH to build SMAC's pyrfr dependency.`n" +
                            "Install it from https://www.swig.org/download.html or via Chocolatey: choco install swig`n" +
                            "Continuing — installation may fail if SWIG is absent.")
            }
        } else {
            conda install -n $ENV_NAME swig -y
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "conda install swig failed. SMAC installation may not succeed."
            } else {
                Write-OK "SWIG installed into '$ENV_NAME'."
            }
        }
    }

    if ($shouldInstall) {
        Invoke-PipInstall $repoPath
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Done — print activation instructions
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " Setup complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host " Environment name : $ENV_NAME" -ForegroundColor White
Write-Host " Environment type : $ENV_MANAGER" -ForegroundColor White
Write-Host ""
Write-Host " To activate the environment:" -ForegroundColor Yellow
if ($ENV_MANAGER -eq "uv") {
    $envRoot = Join-Path $PWD $ENV_NAME
    Write-Host "   PowerShell : $envRoot\Scripts\Activate.ps1"
    Write-Host "   Cmd        : $envRoot\Scripts\activate.bat"
} else {
    Write-Host "   conda activate $ENV_NAME"
}
Write-Host ""
Write-Host " Then run benchmarks with:" -ForegroundColor Yellow
Write-Host "   cd ecqr2025sub-bc"
Write-Host "   python run.py"
Write-Host ""
Write-Host " NOTE: YAHPO Gym requires manual data setup." -ForegroundColor DarkYellow
Write-Host "   Download from https://github.com/slds-lmu/yahpo_data" -ForegroundColor DarkYellow
Write-Host "   and extract into: ecqr2025sub-bc\yahpo_bench_data\" -ForegroundColor DarkYellow
Write-Host " NOTE: JAHS-Bench-201 data downloads automatically on first run." -ForegroundColor DarkYellow
Write-Host ""
