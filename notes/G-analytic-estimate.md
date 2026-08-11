# An analytical seed for G(D,N)

Status: **narrows the search, does not replace it.** The numerical fit remains ground
truth, because `|N-1><N-1|` does not commute with `(a+a†)` — the spin flip feeds back on
the mode, so any impulse-approximation estimate of `G` is first-order only. This was parked
at Drew's request (`DECISIONS.md` context, `PROJECT.md` open question (b)); revisited
2026-08-10 at Drew's request to look into it. Verification script: `src/G_analytic.py`.

## The exact part

The truncated `(a+a†)` operator on `N` Fock levels is a symmetric tridiagonal matrix,
off-diagonal element `√(n+1)` for `n = 0 .. N-2`. That is exactly the three-term-recurrence
(Jacobi) matrix for Hermite polynomials, so its eigenvalues are exact:

```
eig(a + a†) = √2 × {roots of the physicists' Hermite polynomial H_N}
```

Verified to machine precision for `N = 4, 6, 8, 12` (`max |eig − √2·Hroot| < 1e-14`,
`src/G_analytic.py`). This is the Gauss–Hermite quadrature node construction, not a new
result in general, but it's the exact structure underlying "the mode reflecting off the
ceiling" (`CLAUDE.md` gotchas) — the boson's free dynamics in the truncated space are
fully diagonalized by an `N`-dimensional eigendecomposition, no ODE solve required.

## The estimate

With that eigenbasis, the top-level population under free (`G=0`) evolution from vacuum
is exact and cheap:

```
p(t) = |<N-1| e^{-i D M t} |0>|²          (M = a+a†, eigendecomposed above)
```

Apply the area theorem, first order in `G` (i.e. pretending `|N-1><N-1|` and `(a+a†)`
commute, so the spin sees a classical time-dependent field `G·p(t)`): a full `π` flip
needs

```
∫ 2 G p(t) dt = π   over the same [0, 2.5·t_hit] window the numerical fit uses
⟹  G_area(D,N) = π / (2 ∫ p(t) dt)
```

`G_area` is closed-form up to one 1-D quadrature — no coupled 2N-dimensional `sesolve`.

## Validation against the numerical fit

`G_area` systematically **undershoots**, because it ignores the back-action noted above —
it is a first-order impulse approximation. The undershoot is not exactly constant, but it's
tight — swept `N = 3..20`, `D = 0.3..4`:

| N | D | G_area | G_numerical | ratio |
|---|---|--------|-------------|-------|
| 3 | 1.0 | 1.2990 | 1.5683 | 1.2072 |
| 4 | 1.0 | 1.5161 | 1.8053 | 1.1907 |
| 6 | 0.3 | 0.5643 | 0.7033 | 1.2463 |
| 6 | 1.0 | 1.8809 | 2.2548 | 1.1988 |
| 6 | 2.0 | 3.7619 | 4.4275 | 1.1769 |
| 7 | 4.0 | 8.1554 | 9.5595 | 1.1722 |
| 9 | 1.0 | 2.3224 | 2.7793 | 1.1967 |
| 12 | 1.0 | 2.6919 | 3.2288 | 1.1995 |
| 15 | 1.0 | 3.0162 | 3.6034 | 1.1947 |
| 20 | 1.0 | 3.4902 | 4.1653 | 1.1934 |

Ratio range **1.17–1.25**, mean ≈1.19–1.20, over roughly an order of magnitude in both
`N` and `D`. Mild real trend (lower `D` and smaller `N` push the ratio up slightly — more
back-action per Rabi cycle when the collision window is relatively longer/more curved),
not pure noise, but small enough that a single correction factor is a good search seed
everywhere tested.

## What this is used for

`Classical Simulation.py`'s `find_optimal_SpinBosonInteractionCoefficent` now computes
`G_seed = 1.2 × G_area_theorem(D, N)` and does a narrow local scan + refine around it,
instead of a blind 400-point scan over `(0.05π, 4π)`. Falls back to the original full-range
scan if the seed window doesn't contain a near-full-flip point, so correctness doesn't
depend on the ratio holding outside the tested range.

## What this is not

Not a closed-form `G(D,N)` in the sense the open question originally asked for — the
correction factor is fitted, not derived, and has a small residual `N`/`D` dependence.
A genuine closed form would need the back-action correction at the operator level (the
Magnus-expansion second-order term from `[H_CNOT, H_boson] ≠ 0`), which was not attempted
here. If the search-seed speedup isn't enough and a real closed form is wanted later,
that Magnus term is the next thing to look at.
