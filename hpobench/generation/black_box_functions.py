import numpy as np


def rastrigin(x, A=20):
    """Rastrigin function - multimodal optimization benchmark.

    Args:
        x: Input vector with shape (n_dimensions,).
        A: Amplitude parameter (default: 20).

    Returns:
        Function value at point x.
    """
    n = len(x)
    rastrigin_value = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return rastrigin_value


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """Ackley function - multimodal optimization benchmark.

    Args:
        x: Input vector with shape (n_dimensions,).
        a: Amplitude parameter (default: 20).
        b: Exponential decay parameter (default: 0.2).
        c: Oscillation frequency parameter (default: 2π).

    Returns:
        Function value at point x.
    """
    n = len(x)
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    ackley_value = term1 + term2 + a + np.exp(1)
    return ackley_value


def griewank(x):
    """Griewank function - multimodal optimization benchmark.

    Args:
        x: Input vector with shape (n_dimensions,).

    Returns:
        Function value at point x.
    """
    n = len(x)
    term1 = np.sum(x**2) / 4000
    term2 = 1
    for i in range(n):
        term2 *= np.cos(x[i] / np.sqrt(i + 1))
    griewank_value = term1 - term2 + 1
    return griewank_value


def weierstrass(x, a=0.5, b=3, kmax=20):
    """Weierstrass function - fractal optimization benchmark.

    Args:
        x: Input vector with shape (n_dimensions,).
        a: Amplitude parameter (default: 0.5).
        b: Frequency multiplier parameter (default: 3).
        kmax: Maximum summation index (default: 20).

    Returns:
        Function value at point x.
    """
    n = len(x)
    weierstrass_value = 0
    for i in range(n):
        for k in range(kmax + 1):
            weierstrass_value += (a**k) * np.cos(2 * np.pi * (b**k) * (x[i] + 0.5))
        for k in range(kmax + 1):
            weierstrass_value -= (a**k) * np.cos(2 * np.pi * (b**k) * 0.5)
    return weierstrass_value


def shekel(x, m=10):
    """Shekel function - multimodal optimization benchmark with variable local minima.

    Args:
        x: Input vector with shape (n_dimensions,).
        m: Number of local minima (default: 10).

    Returns:
        Function value at point x (negated for minimization).
    """
    n = len(x)
    A = np.random.rand(m, n) * 10  # random A matrix for each run
    C = np.random.rand(m) * 10
    shekel_value = 0
    for i in range(m):
        shekel_value -= 1 / (C[i] + np.sum((x - A[i]) ** 2))
    return -shekel_value


def hartmann6(x):
    """Hartmann 6-dimensional function - optimization benchmark.

    Args:
        x: Input vector with shape (6,).

    Returns:
        Function value at point x.
    """
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array(
        [
            [1.0, 1.2, 3.0, 3.2],
            [3.6, 1.6, 0.7, 3.9],
            [4.0, 1.6, 0.8, 3.4],
            [1.6, 0.0, 3.6, 0.8],
            [1.6, 0.0, 3.6, 0.8],
        ]
    )
    P = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5894],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )
    hartmann6_value = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * (x[j] - P[i, j]) ** 2
        hartmann6_value -= alpha[i] * np.exp(-inner_sum)
    return -hartmann6_value
