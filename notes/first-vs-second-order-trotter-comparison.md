# First- vs second-order Trotter: results comparison

Status: **complete**, 2026-08-11. Compares `src/NuronSim.py` (first order,
`NetworkOfNeuronsTrotterStep`) against `src/NeuronSim2ndOrderTrotter.py` (second order,
`build_second_order_trotter_circuit`, see `notes/second-order-trotter.md` for the
construction and the bug it caught). Same Hamiltonian, same `G` calibration, same Willow
qubit chain and compilation, same noise model, same post-selection, same adaptive step
schedule (`recommended_trotter_steps(t)`, fitted to first-order error, D-18/D-19, reused
unchanged for both — see "Method" below for why that matters to how to read this).

Four runs, all `NumberOfBosonicModes=1, NumberOfFockStates=5, J=0.5, G≈2.029`, swept
`t ∈ [0.1, 8.25]` (100 points, capped by the schedule's noise budget, D-19):

| Run | File |
|---|---|
| 1st order, noiseless | `results/NuronSim_noiseless.png` / `.npz` |
| 1st order, noisy + post-selected | `results/NuronSim_noisy.png` / `.npz` |
| 2nd order, noiseless | `results/NeuronSim2ndOrderTrotter_noiseless.png` / `.npz` |
| 2nd order, noisy + post-selected | `results/NeuronSim2ndOrderTrotter_noisy.png` / `.npz` |

---

## Headline result: it depends on whether noise is on

**Noiseless (isolates Trotter error only) — second order wins clearly:**

| | occupation RMS error vs qutip | spin magnetisation RMS error vs qutip |
|---|---|---|
| 1st order | 0.0693 | 0.1406 |
| 2nd order | 0.0406 | 0.0089 |
| improvement | **1.7x** | **15.7x** |

**Noisy, post-selected (the hardware-realistic case) — second order loses:**

| | occupation RMS error vs qutip | spin magnetisation RMS error vs qutip |
|---|---|---|
| 1st order | 0.382 | 0.408 |
| 2nd order | 0.554 | 0.557 |

Both scripts used the identical schedule (same `NumberOfTrotterSteps` at every `t`, mean
16.7 steps, same `t`-range), so this is not an artefact of comparing different step
counts — it is what happens when the *same* step count is spent on a circuit with ~1.5x
more two-qubit gates per step.

## Why: gate count, not accuracy, is what changed sign

Second order's extra accuracy is real and large — the noiseless numbers above and the
convergence-order check in `notes/second-order-trotter.md` (`1-F ~ r^-4` vs `r^-2`,
confirmed by a log-log fit, not assumed) both show it. What also changed is the gate
count: at matched `r`, the second-order circuit has **~1.51x** as many compiled two-qubit
gates as first order (`consistency_checks.py`: `r=12` gives 204 vs 308 gates — the extra
cost is the boson chain's odd-bond layer being applied twice per step instead of once,
see `notes/second-order-trotter.md`'s layer-merging section).

More gates per shot means more accumulated two-qubit gate error, which shows up directly
in the post-selection survival rate — the fraction of shots that stayed in the physical
unary subspace (D-4):

| | mean survival rate | min | max |
|---|---|---|---|
| 1st order | 53.2% | 29.5% | 94.7% |
| 2nd order | 45.3% | 22.9% | 92.9% |

D-19's schedule (`r*(t) = round(4.0*t)`, capped at `t≈8.25`) was already deliberately
pushed close to the noise budget `r_max=33` for *first*-order Trotter error — that's the
whole point of D-18's two-stage construction (Trotter requirement vs. noise budget,
D-17/D-18). Reusing that schedule unchanged for a circuit with 1.5x the gate count per
step doesn't just add a fixed overhead; it pushes the *already-tight* operating point
further into the noise-dominated regime the D-15 mechanism describes. First order was
closer to Trotter-error-limited at this schedule; second order, with the same steps but
more gates, is more noise-limited. Both circuits are evaluated at essentially the same
absolute Trotter error budget (same `r`, same `t`) but second order pays a higher noise
tax to get there and doesn't need to — it was already accurate enough at far fewer steps.

## Reframed: this is a scheduling problem, not a second-order problem

The comparison above is not "second order is worse for this circuit" — it's "an
apples-to-apples *step-count* comparison isn't the fair comparison once accuracy-per-step
differs this much between the two methods." The noiseless numbers show second order
reaches much higher accuracy per step; the sensible use of that is *fewer steps for the
same accuracy*, not *the same steps for more accuracy than needed while also paying more
noise*. Concretely: first order's `r=12` noiseless occupation error (0.069, close to the
overall RMS above) needs far more steps at second order's `r^-4` scaling to be *beaten*
by a wide margin, but needs far *fewer* steps to be *matched* — second order should be
able to hit first order's accuracy at a fraction of `r`, and therefore a fraction of the
gate count and noise exposure, comfortably beating first order on both accuracy and
survival rate simultaneously. That schedule was not derived here (see below) — this
comparison is only informative about the fixed-`r` case, which is a real but narrow
question ("if I don't change anything else, does switching product formulas at the same
step count help?" — noisy answer: no, not on this hardware/gate-count trade-off; that's
still a useful negative result, just not the interesting question).

## What this does and doesn't establish

**Established, with numbers:** second-order Trotterization, correctly implemented (see
the convergence-order bug in `notes/second-order-trotter.md`), gives dramatically better
noiseless accuracy per Trotter step (1.7–16x depending on observable) at ~1.5x the gate
cost per step — a good trade in isolation. Naively substituting it into an existing
schedule tuned for first order's noise/accuracy trade-off makes the noisy result *worse*,
because that schedule was already sitting at the edge of the noise budget and the extra
per-step gate cost pushes past it.

**Not established:** what happens with a schedule derived *for* second order. Since
`ε_T ~ r^-4` there instead of `r^-2`, the D-17-style derivation (`ε_T ≈ A/r^p`, `ε_N ≈ Br`,
minimize the sum) predicts a different, shallower `r*(t)` scaling and likely a *later*
noise-budget cap than first order's `t≈8.25`, not an earlier one — plausibly beating first
order on both accuracy and noise budget simultaneously once the schedule is chosen
correctly for the method actually being used. This is flagged as the natural next step in
`notes/second-order-trotter.md` and is not pursued here (would need its own
`src/trotter_fidelity_schedule.py`-style fidelity grid search, run against
`build_second_order_trotter_circuit` instead of `build_trotter_circuit` — mechanically
straightforward given the existing tooling, but a separate piece of work with its own
grid search and its own config-scope caveat, matching D-16 through D-19's pattern).

## Scope caveat (same shape as every schedule-related entry this session)

All of the above is one config (`L=1, N=5, D=1.0, J=0.5`), one `willow_pink` calibration
snapshot, one schedule (`c=4.0`, fitted to first-order error), and one noise-budget
threshold (`min_fidelity_budget=0.5`). The *sign* of the noisy comparison (2nd order
loses at matched `r`) is a consequence of the schedule being tuned for the other method's
error profile, not a general statement about second-order Trotter methods.
