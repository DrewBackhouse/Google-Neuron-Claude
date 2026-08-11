# Optimal NumberOfTrotterSteps per data point

Status: **method retired and replaced; replacement not yet implemented.** The adaptive
schedule currently returned by `neuron_circuit.recommended_trotter_steps(t)` and used by
`NuronSim.py` (`UseAdaptiveTrotterSteps`, default `True`) is the **superseded** one from
D-16 — the code has not yet caught up with D-17/D-18. Do not treat its numbers as current.

History of this note in one line each:

| Date | What happened | Decision |
|---|---|---|
| 2026-08-10 | Grid search over `(t, steps)` using a two-observable error metric; smoothed schedule implemented | D-16 |
| 2026-08-11 | Schedule shown to be contaminated by accidental observable crossings; monotone form + sweep cap adopted | D-17 |
| 2026-08-11 | Metric replaced with noiseless state fidelity; D-17's numbers demoted to provisional | D-18 |

---

## The question

D-15 found that `NuronSim.py`'s time sweep — `dt = t / NumberOfTrotterSteps` — means step
count sets circuit depth (and therefore accumulated gate noise) *independently of `t`*,
while Trotter error at fixed step count grows *with* `t`. A single step count can't be
optimal at every point in a sweep. This note works out what the step count should be as a
function of `t`.

That framing is still correct and is not affected by anything below. What changed twice is
*how the optimum is measured*.

---

# Part 1 — The observable-metric analysis (2026-08-10) — **SUPERSEDED**

Retained for the method and for the aliasing finding, which Part 2 builds on. Its
*conclusions about which step counts are best* are wrong; see Part 2.

## Method

Grid search over 14 values of `t` across `[0.5, 20]` and 17 candidate step counts (`1..40`,
denser at the low end). For each, computed the Willow-mapped, Willow-compiled circuit's
**exact** expectation values — `cirq.DensityMatrixSimulator` with and without the
`willow_pink` noise model, `dtype=complex128` (the cirq default `complex64` fails
positive-semidefiniteness / trace-1 validation at higher step counts — hit directly, not
hypothetical). Exact rather than shot-sampled so the search isn't also fighting shot noise.

Error metric, combined in quadrature so boson occupancy and spin magnetization contribute
comparably despite their different natural ranges:

```
error(t, steps) = sqrt( ((occ - qutip_occ(t)) / (N-1))^2  +  ((mag - qutip_mag(t)) / 2)^2 )
```

`optimal_steps(t) = argmin_steps error(t, steps)` over the noisy error.

**This metric is the flaw.** See Part 2.

## Result 1: the trade-off is real and large

At `t = 6.5`, the fixed default (`NumberOfTrotterSteps = 7`) gives error `0.57`; the
per-point optimum (12 steps) gives `0.04`. Across the 14 sampled points, fixed steps=7 is
simultaneously too many steps (too much noise) at small `t` and too few (too much Trotter
error) at some larger `t`.

**Still stands.** The premise that step count must vary with `t` survives D-17 and D-18
intact. Only the resulting schedule was wrong.

## Result 2: the raw per-point optimum is jagged, and that's physical, not sampling noise

`optimal_steps(t)` over the 14 points: `1, 2, 6, 6, 12, 3, 2, 22, 26, 22, 26, 8, 5, 18` —
not monotonic, not smooth. These are **exact** expectation values (no shot noise), so the
jaggedness is a real feature. Cause: the system's own dynamics are oscillatory (Rabi-like
flopping), so error at a *fixed* step count doesn't grow monotonically with `t` — it
beats/aliases against the true oscillation. Confirmed directly: Trotter-only error (noise
off) at `steps=1` swings between ~0 and ~1 across the sweep rather than growing steadily
(`results/optimal_trotter_steps.png`, rightmost panel).

**Mechanism still stands; the mitigation did not.** Part 2 shows the artifact is narrow in
`steps`, not only in `t`, so smoothing along `t` cannot remove it.

## Result 3: smoothing along `t`, and the "hard window"

A 3-point moving window along `t` before taking argmin gave:

| t | 0.5 | 2.0 | 3.5 | 5.0 | 6.5 | 8.0 | 9.5 | 11.0 | 12.5 | 14.0 | 15.5 | 17.0 | 18.5 | 20.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| steps | 2 | 5 | 6 | 5 | 5 | 5 | 3 | 22 | 22 | 26 | 8 | 8 | 8 | 18 |

with a window around `t ≈ 12.5-15` where even the best available step count left error
~0.37-0.41 against ~0.02-0.22 elsewhere.

**Both superseded.** The dips at `t = 9.5` and `t = 17-18.5` are artifacts (Part 2). The
"hard window" is not a dynamical anomaly — it is the leading edge of the sweep cap (D-17).

---

# Part 2 — Why the observable metric fails (2026-08-11, D-17)

Prompted by Drew asking whether the low error at `t=9.5, r=2` was a coincidence rather than
a correct simulation, and arguing the schedule should be monotone in `t`. Both right.
Established by re-analysing the **existing** grid — no new circuit evaluations.

## Evidence 1 — the argmins are two populations

Scoring each argmin by `E*/mean(immediate neighbours in r)`; the 14 points separate at
0.40 vs 0.53 with a clear gap:

- **Basin** (8 pts, 0.53–0.93): `t = 0.5, 3.5, 5.0, 6.5, 11, 12.5, 14, 15.5`
- **Needle** (6 pts, 0.06–0.40): `t = 2.0, 8.0, 9.5, 17, 18.5, 20`

A "needle" is an isolated one-point dip — `E*` 4–17× below both neighbours, with no
supporting structure. A genuine optimum degrades gracefully when `r` changes by one; a
needle falls apart. Every non-monotone dip in the Part 1 schedule is driven by needles.

## Evidence 2 — the noiseless column is decisive

Trotter-only error at `t = 9.5`:

```
r  =    1     2     3     4     5     6     7     8    10    12  ...   40
E0 = 0.15  0.03  0.15  0.11  0.09  0.30  0.51  0.70  0.54  0.06  ...  0.02
```

Trotter error cannot be 23× *worse* at `r=8` than at `r=2`. The `r=2` state is badly wrong;
its two observables happen to cross the true values. Same at `t=8.0` (`r=4 → 0.04`,
`r=8 → 0.89`).

## Evidence 3 — noise "improves" accuracy at half the points

At 7 of 14 argmins the noisy error is *lower* than the noiseless error at the same `r`.
Worst: `t=20, r=18` — noiseless `0.199`, noisy `0.021`. Adding decoherence improved
agreement 10×, which is only possible if noise is cancelling Trotter error.

## Evidence 4 — the cap

Smallest `r` whose noiseless error stays below `0.15` for **all** larger `r`:

| t | 0.5 | 2 | 3.5 | 5 | 6.5 | 8 | 9.5 | 11 | 12.5 | 14 | 15.5 | 17 | 18.5 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 1 | 10 | 6 | 6 | 10 | 18 | 12 | 22 | 35 | 22 | 40 | — | 30 | — |

Past `t ≈ 12` the circuit isn't Trotter-converged anywhere in the tested range; at `t = 17`
and `20` it never converges by `r = 40`.

## Why monotone is right, with a derivation

First-order Trotter gives `ε_T ≈ A t²/r`; two-qubit gate count `∝ r` so `ε_N ≈ B r`.
Minimizing the sum:

```
r* = t · √(A/B)      linear in t, monotone
E* = 2t · √(A B)     grows linearly in t
```

Part 1's table violates the second prediction (`E* = 0.021` at `t=20` vs `0.223` at
`t=3.5`) — independent confirmation the argmin wasn't tracking physics.

Fitting `c = r*/t` on **basin points only**: `c = 1.76 ± 0.27` (range 1.20–2.08). Needle
points give `c = 0.54 ± 0.30`, all under-stepping — the signature of "accidentally right at
low depth."

---

# Part 3 — The replacement method (2026-08-11, D-18) — **TO IMPLEMENT**

## Metric

```
F(t, r) = |<ψ_qutip(t) | ψ_trotter(t, r)>|²        (noise off)
```

Fidelity compares the full state, so there is no low-dimensional projection for a wrong
state to hide behind. This eliminates the Part 2 failure mode rather than detecting it —
the basin/needle diagnostic becomes unnecessary, not merely automated.

## Fidelity alone is degenerate — do not argmax over it

`F` increases monotonically with `r` (`F → 1` as `r → ∞` by construction), so
`argmax_r F(t, r)` always returns the largest `r` tested and carries zero information. The
metric is **not** a drop-in replacement inside Part 1's argmin loop. The schedule must be
built in two stages:

1. **Trotter requirement, from fidelity.** `r_T(t)` = smallest `r` with `1 - F(t, r') < tol`
   for **all** `r' ≥ r`. The "for all larger `r`" quantifier is retained from Part 2 — it is
   what makes the threshold immune to accidental single-point dips.
2. **Noise budget, separately.** `r_max` from accumulated two-qubit gate error at the
   compiled per-step gate count. A property of the circuit and calibration, independent of
   `t` (D-15's mechanism).
3. **Schedule** = monotone fit to `r_T(t)`; **cap** the sweep at the largest `t` with
   `r_T(t) ≤ r_max`.

Noise enters as a *constraint*, not as a term in the objective. Mixing it into the
objective is precisely what let noise cancel Trotter error and produce the needles.

## Free correctness check to wire in at the same time

Every Hamiltonian term conserves excitation number within each boson column — unary
`H_boson` is `XX+YY`, `H_spin` is `XX+YY` on spin qubits, and `H_CNOT` is `|1><1| ⊗ X`
acting on a different qubit than the column it reads. So each Trotter factor conserves it
exactly, and **the noiseless Trotterized state must stay exactly in the unary sector for
all `r`**. Computing `F` requires projecting the `2^6`-dim qubit state onto the 10-dim
physical space anyway; the norm of the discarded component should be `0` to numerical
precision. Nonzero in a *noiseless* run ⇒ compilation or qubit-ordering bug, not physics.
Assert it.

## Implementation notes

- `|·|²` makes global phase irrelevant — no phase convention needed.
- Qubit ordering must match `willow_qubit_chain` (boson qubits per mode, then spin qubits —
  the same ordering `compute_observables_from_z_expectations` relies on). Getting this wrong
  gives a plausible-looking but wrong fidelity, so run the unary-sector check first as an
  ordering smoke test.
- Needs a statevector ⇒ the `UseNoiseModel=False` / `simulate` path. Cheaper than the noisy
  path (no sampling); D-15's flag already exposes it.
- `results/optimal_trotter_steps_grid.npz` contains observable-metric data only. The re-run
  is a fresh computation, not a re-analysis.

## What to check on the re-run

Whether the fidelity-derived schedule agrees with the observable-derived one at the **8
basin points**. If it does, the observable metric was adequate and D-18's machinery is
unnecessary — a free comparison worth making explicitly (D-18's "overturned by").

---

## Scope of validity — unchanged and still binding

Everything here is fitted to one config (`NumberOfBosonicModes=1, NumberOfFockStates=5,
D_list=[1.0], spin_interaction_coefficient=0.5`, `G≈2.03`) against one `willow_pink` median
calibration snapshot. Not a general law. `neuron_circuit.check_trotter_schedule_config(...)`
warns (print, not raise) on mismatch — it is print-only, so check for it.
`src/optimal_trotter_steps.py` takes ~5-6 min in the `GoogleQVM` env.

## Current code state

- `src/optimal_trotter_steps.py` — computes the **retired** observable metric.
- `neuron_circuit.recommended_trotter_steps` — returns the **superseded** D-16 schedule.
- `NuronSim.py` — sweeps to `Time = 20`, **uncapped**.

None of D-17 or D-18 is implemented yet.
