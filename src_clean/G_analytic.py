"""Closed-form structure behind G(D,N): the truncated (a+a†) spectrum is exactly the
roots of a Hermite polynomial, which gives a first-order (no-back-action) area-theorem
estimate for G. See notes/G-analytic-estimate.md for the derivation and validation
against the numerical fit.

Run: python3 src/G_analytic.py
Requires: numpy, scipy.
"""
import numpy as np
from numpy.polynomial.hermite import hermroots
from scipy.integrate import quad


def boson_eigensystem(N):
    """Eigendecomposition of the truncated (a+a†) operator on N Fock levels."""
    M = np.zeros((N, N))
    for n in range(N - 1):
        M[n, n + 1] = np.sqrt(n + 1)
        M[n + 1, n] = np.sqrt(n + 1)
    return np.linalg.eigh(M)  # M = eigvecs @ diag(eigvals) @ eigvecs.T


def top_level_population(t, D, N, eigvals, eigvecs):
    """p(t) = |<N-1| e^{-i D M t} |0>|^2 for the isolated (G=0) boson mode."""
    phase = np.exp(-1j * D * eigvals * t)
    amplitude = np.sum(eigvecs[N - 1, :] * eigvecs[0, :] * phase)
    return np.abs(amplitude) ** 2


def G_area_theorem(D, N, window_factor=2.5):
    """First-order estimate of G from the pi-pulse area theorem: a full spin flip needs
    integral 2 G p(t) dt = pi over the first collision window. Ignores that |N-1><N-1|
    does not commute with (a+a dag), so it runs low by a fairly stable ~1.17-1.25x
    (notes/G-analytic-estimate.md) -- use as a search seed, not as G itself."""
    eigvals, eigvecs = boson_eigensystem(N)
    t_window = window_factor * np.sqrt(N - 1) / D
    integral, _ = quad(lambda t: top_level_population(t, D, N, eigvals, eigvecs), 0, t_window, limit=200)
    return np.pi / (2 * integral)


if __name__ == '__main__':
    print("Eigenvalues of truncated (a+a†) vs sqrt(2) x roots of Hermite H_N:")
    for N in [4, 6, 8, 12]:
        eigvals, _ = boson_eigensystem(N)
        hroots = np.sort(hermroots([0] * N + [1]))
        max_diff = np.max(np.abs(np.sort(eigvals) - np.sqrt(2) * hroots))
        print(f"  N={N:2d}  max |eig - sqrt(2)*Hroot| = {max_diff:.2e}")

    print("\nG_area_theorem (first-order seed, correction factor ~1.2 applied separately):")
    for N, D in [(6, 1.0), (8, 1.0), (5, 1.0)]:
        print(f"  N={N}, D={D}: G_area = {G_area_theorem(D, N):.4f}")
