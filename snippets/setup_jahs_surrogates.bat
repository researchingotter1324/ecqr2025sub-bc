@echo off
setlocal enabledelayedexpansion

set "URL=https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar"
set "DEST=jahs_bench_data"
set "TAR=%DEST%\assembled_surrogates.tar"

echo Creating destination folder: %DEST%
if not exist "%DEST%" mkdir "%DEST%"

echo.
echo Downloading with resume support...
echo File: %TAR%
curl -L -C - --fail --retry 10 --retry-delay 5 --retry-all-errors -o "%TAR%" "%URL%"

if errorlevel 1 (
    echo.
    echo Download failed. Re-run this script; curl will resume from the partial file.
    exit /b 1
)

echo.
echo Verifying tar archive...
tar -tf "%TAR%" >nul

if errorlevel 1 (
    echo.
    echo The tar file is incomplete or corrupted.
    echo Re-run this script to resume the download.
    exit /b 1
)

echo.
echo Extracting into %DEST%...
tar -xf "%TAR%" -C "%DEST%"

if errorlevel 1 (
    echo.
    echo Extraction failed.
    exit /b 1
)

echo.
echo Done. Surrogates are downloaded and extracted in:
echo %CD%\%DEST%