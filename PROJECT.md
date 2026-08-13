# Quantum Network of Neurons — Project State

**Read this file and `DECISIONS.md` at the start of every new chat.**
Update both at the end of every chat.

Last updated: 2026-08-13

---

## Goal

Simulate a spin–boson Hamiltonian, interpretable as a network of neurons, on Google
Willow (105 qubits). Demonstration and benchmark — **not** a quantum-advantage claim
(see DECISIONS.md, D-6).

## The model

On a chain of `L` sites with Dirichlet boundaries, each site carrying a bosonic mode
truncated to `N` Fock states plus a spin:

```
H = H_boson + H_spin + H_CNOT

H_boson = D  Σ_j (a_j + a_j†)
H_spin  = J  Σ_j (σ+_j σ-_{j+1} + σ-_j σ+_{j+1})
H_CNOT  = G  Σ_j |N-1><N-1|_j X_j
```

Neuron reading: the boson is the membrane potential integrating upward; the Fock-space
ceiling is the firing threshold; the spin flip is the action potential; XY hopping is
the axon carrying the signal to the next site.

The truncation at `N` is **physical, not an approximation** — there is no phase-space
rotation term, so the mode's oscillation comes entirely from reflecting off the ceiling.

## Mapping to qubits

Unary encoding, `|n> = ⊗_i |δ_{i,n}>`, so `a† = Σ_n √(n+1) σ+_{n+1} σ-_n`.
Gives an `L × (N+1)` qubit grid — `N` qubits per boson column, plus one spin qubit.

```
H_boson = (D/2) Σ_j Σ_i √(i+1) (X_{i,j} X_{i+1,j} + Y_{i,j} Y_{i+1,j})
H_spin  = (J/2) Σ_j (X_{N,j} X_{N,j+1} + Y_{N,j} Y_{N,j+1})
H_CNOT  =  G   Σ_j |1><1|_{N-1,j} X_{N,j}
```

Every mapped term is nearest-neighbour on that grid, and Willow's lattice is degree-4,
so a good embedding should need **zero SWAPs**. Worth confirming in step 3.

**Frame: computational basis, controlled-X interaction (D-11).** A spin-sector rotation to
the ± basis was considered and rejected — no resource benefit, slightly negative on
single-qubit count, and `XX + YY` is already Willow's native iSWAP-like interaction. Initial
state is all modes `|0>` (unary `|100...0>`), all spins `|0>`; no two-qubit gates in prep.
The frame remains free at *compile* time, since the two descriptions are physically
identical.

**Conserved quantity: `Π_j X_j`** over the spin qubits — exact, verified numerically
(`src/symmetry_check.py`, D-9). Not diagonal in this frame, so reading it costs `L`
Hadamards before measurement, and it does not commute with the spin `Z` readout. See D-9
and D-11 for why it is a diagnostic rather than a per-shot filter.

---

## Outline

| # | Step | Status |
|---|------|--------|
| 1 | Classical exact simulation — fix `D_j`, `J`, `G`, `N`; confirm integrate-and-fire and propagation | in progress |
| 2 | Trotter error analysis — Childs et al. commutator bounds on our 3-way split; test term orderings | D-17 derives the `r* ∝ t` scaling; D-18/D-19 implement the fidelity-based schedule (`src/trotter_fidelity_schedule.py`); D-20 adds a second-order (Strang) alternative (`src/NeuronSim2ndOrderTrotter.py`) with an explicit convergence-order check; **D-21 re-fits the first-order schedule against the noisy, post-selected metric directly** (`src/adaptive_trotter_model_scan.py` + `src/linear_trotter_finegrid_scan.py`), confirming linear beats quadratic — `c=4.0` → `c=2.25`; **D-22 does the same for second order** (`src/second_order_trotter_linear_scan.py`, own `c=2.0`) — first order still wins head-to-head, narrowly |
| 3 | Circuit synthesis and Willow layout — confirm SWAP-free embedding, pick qubit patch from calibration | **D-1's zero-SWAP claim confirmed for `L=1–4`** — `neuron_circuit.find_low_error_qubit_embedding` (D-23) generalizes the old `L=1`-only chain search to a real subgraph embedding, needed because the `L≥2` logical topology is a comb, not a line (no Hamiltonian path exists at all once `L≥3`, proved not just patched). Coherence-aware (T1 + two-qubit error) qubit selection also required, not just gate-count — D-23. `L=3,N=5` exercised in `results/VMBenchmark/`; `L=3,N=6` target still not run |
| 4 | Noise simulation + post-selection — realistic error model, quantify unary-constraint recovery | noise model wired up (`src/NuronSim.py`, D-14); D-15→D-18/D-19 replaced the fixed/observable-metric step schedule with a fidelity-derived, capped, monotone one; **post-selection implemented (D-19)** — `neuron_circuit.sample_shots_with_postselection` filters shots on the unary one-hot constraint (D-4), mean survival rate `53.2%` in the first full sweep |
| 5 | Configuration sweep — (L, N, steps) vs. observable fidelity; justify final choice | not formally started, but `results/VMBenchmark/` (D-23) is a first pass at exactly this: `L=1→3` and `N=3→7` sweeps with noise+post-selection, including a per-`N` schedule re-fit. Early signal: `a` (the schedule constant) is flat at `2` for `N=3–6`, then drops to `1.5` at `N=7` alongside a notably higher error floor — a lead on real Trotter-error-vs-`N` scaling, not yet confirmed as a trend (one data point) |

Each step gets its own chat. This file is the handover.

## Target observable

Ranked by ambition; #3 is the ideal, #2 is the realistic target.

1. Single neuron fires — one site integrates, reflects, flips its spin
2. **Signal propagates** — site 0 fires, spin hops to site 1, perturbs its dynamics
3. Network behaviour — cascades, synchronisation (likely out of reach, see resource estimate)

---

## Open questions

- [ ] **`G(D,N)`** — currently fitted numerically; Drew has a script. Concern is whether the
      fit scales to larger systems. Two things to test: (a) does `G` calibrated on `L=1`
      transfer to `L=3`? First check (2026-08-10, `D_j = [1, 0.85, 0.72]`, `J = 0.2`):
      *mostly* — site 0 reaches `<Z> = -0.93` in the chain vs. `< -0.99` in isolation.
      Close but not exact; needs checking across a range of `J` and `D_j` gradients before
      calling it settled. (b) Closed-form route via the truncated `(a+a†)` spectrum —
      **investigated 2026-08-10 at Drew's request** (`notes/G-analytic-estimate.md`,
      `src/G_analytic.py`). Result: the truncated spectrum is exactly `√2 ×` the roots of
      Hermite `H_N` (exact), giving a first-order area-theorem estimate `G_area(D,N)` that
      undershoots the true `G` by a fairly stable ~1.17–1.25x (mean ~1.2) over `N=3–20`,
      `D=0.3–4`. Not a true closed form (the correction is fitted, small residual `N`/`D`
      trend) — now used as a search seed in `find_optimal_SpinBosonInteractionCoefficent`
      (~5x faster calibration at `N=6`, more at larger `N`); numerical fit is still ground
      truth. **Why it undershoots — back-action:** `|N-1><N-1|` does not commute with
      `(a+a†)`, so the spin flip feeds back on the mode and any impulse-approximation
      estimate of `G` is first-order only. A genuine closed form would need that correction
      at the operator level (the Magnus second-order term from `[H_CNOT, H_boson] ≠ 0`);
      parked unless asked for.
- [ ] **Per-site `G_j`, flagged 2026-08-10, not yet pursued.** With a single global `G`
      (`H_CNOT = G Σ_j |N-1><N-1|_j X_j`, D-9/D-11) calibrated against one reference `D`,
      other sites systematically fail to reach a full flip in isolation — confirmed at
      `J=0`, `D_list=[1, 0.85, 0.7225]`, `N=6`: site 0 `min<Z>=-0.998`, site 1 `-0.962`,
      site 2 `-0.826`. Mechanism: `G_area(D,N)` scales almost exactly linearly in `D`
      (`notes/G-analytic-estimate.md` — falls out of `p(t)` being a universal function of
      `D·t`), so a `G` tuned to `D_0` overshoots the π pulse area at smaller `D_j` by
      `~D_0/D_j`. Drew wants per-site `G_j` explored so every site can reach a full flip
      regardless of `D_j`. **This changes the Hamiltonian** from D-9/D-11's single-`G`
      form to `H_CNOT = Σ_j G_j |N-1><N-1|_j X_j` — worth a DECISIONS.md entry once
      pursued, not just a code tweak, since D-9's symmetry argument (`H_CNOT` commutes
      with `Π_j X_j` because it's `∝ X_j` on each site) is unaffected by making `G` site
      dependent, but it should be checked explicitly rather than assumed.
      **Tension to watch:** the current single-`G` mismatch is what makes a flip at site
      1/2 attributable to `J`-hopping rather than each site self-flipping (useful for the
      D_j-profile work above) — per-site `G_j` would remove that free diagnostic, so
      distinguishing self-flip from propagation would then rely on timing/lag rather than
      flip completeness alone.
- [ ] **Choice of `D_j` profile** — what spatial variation actually produces clean
      sequential firing rather than smeared simultaneous firing? First attempt
      (2026-08-10, geometric decay `0.85^j`, `J = 0.2 D_0`) still gives near-simultaneous
      dips across all three sites — too weak a gradient, or `J` too small to see a causal
      lag. Still open.
- [ ] **Spin-sector error mitigation** — an exact Z2 symmetry `Π_j X_j` *does* exist (D-9,
      superseding D-5), but two obstacles keep it from being a per-shot filter: the natural
      initial state sits in an equal superposition of both parity sectors, and in the
      computational basis it does not commute with the spin `Z` readout. Decide between
      using it diagnostically vs. changing the initial state to `|±>` — and separately,
      whether ZNE or an echo/mirror calibration is needed on the `L` spin qubits regardless.
- [ ] **Encoding choice is no longer settled** — Sawaya et al. find unary loses to Gray/SB
      on both qubits and gates for tridiagonal `q̂` at `d = 4, 7, 8`, and our candidates are
      `N = 6, 8`. D-4 amendment gives three reasons the decision still holds, but they need
      quantifying in step 3 rather than asserting.

## Next actions

Step 1 is running (`src/Classical Simulation.py`, macOS venv with qutip — see Environment
in `CLAUDE.md`). The five known issues from the previous session are fixed:

1. `D_list` is now per-site (geometric decay, `D_j = D_0 · 0.85^j`) — first guess, not tuned.
2. `find_optimal_SpinBosonInteractionCoefficent` does a coarse grid scan over `G` first, then
   refines around the earliest near-full-flip window, so it returns the smallest resonance
   rather than whatever `minimize_scalar`'s bounded search happened to land on.
3. `spin_interaction_coefficient` (`J`) is on, first guess `0.2 · D_0`.
4. `G` is calibrated once at `L=1` on the reference site and reused across the chain. Transfer
   check added directly in the script output.
5. `mesolve` → `sesolve` (no collapse operators, so this is pure Schrödinger evolution); the
   dead `args={...}` dict is gone.

**First `L=3, N=6` run:** `G` calibrated at `L=1` (`D=1`) is `2.241`, reaching `<Z> < -0.99`
in isolation. In the `L=3` chain, site 0's minimum `<Z>` is `-0.93` — the transfer mostly
holds but is not exact; `J` measurably perturbs the flip. Worth characterising rather than
assuming clean transfer for larger `J` or steeper `D_j` gradients (open question (a) below).

**Diagnosed why that run looked like lockstep firing rather than propagation.** At
`J=0` (isolating the effect), site 0 reaches `min<Z>=-0.998`, site 1 `-0.962`, site 2
`-0.826` — each site's own boson dynamics partially self-flip because the single global
`G` was tuned to `D_0` and `D_1, D_2` weren't different enough to fully suppress that (see
the new per-site-`G` open question below for the mechanism). So the earlier near-simultaneous
dips across all three sites were largely each site independently climbing to its own
ceiling, not signal hopping from site 0 via `J`. **Next, once back on `L=3`:** widen the
`D_j` gradient so sites 1/2 clearly cannot self-flip in isolation, then turn `J` back on —
a flip appearing there would then be an unambiguous propagation signal.

**Script currently set to `L=1` (single site)** at Drew's request, to work the `G`
calibration in isolation before returning to the chain.

**Independent agreement worth noting:** the script's `t_hit = √(N-1)/D` matches the
firing-time estimate derived independently for `notes/resource-estimate.md`. The two were
arrived at separately, so the step-budget analysis there rests on the same footing as the
existing numerics.

**`src/NuronSim.py` brought in line with `Classical Simulation.py`, and mapped to hardware
(2026-08-10).** `NuronSim.py` is the qutip-vs-Cirq-Trotter-circuit comparison script (kept
separate from `Classical Simulation.py`, which is the `D_j`/propagation study). Two changes:

1. **G-finding.** Ported the same method as `Classical Simulation.py`: `D_list` (per-site,
   even though this script currently only runs `NumberOfBosonicModes=1`), the
   `G_area_theorem`-seeded calibration search in place of the old blind 400-point scan, and
   `sesolve` instead of `mesolve` (pure Schrödinger evolution, no collapse operators). Also
   fixed a latent bug this exposed: `NetworkOfNeuronsTrotterStep` built one
   `BosonicDisplacementGate` from a single scalar `D` and reused it for every site — replaced
   with one gate per site off `D_list[i]`, matching `QutipHamiltonian`. No behaviour change
   at `L=1`; matters once this script is run at `L>1`.
2. **Willow mapping + noise model.** The Cirq sim's circuit is now explicitly mapped onto
   real `willow_pink` `GridQubit`s (`find_low_error_qubit_chain` — greedy low-error chain
   search over the device connectivity graph, SWAP-free by construction, D-13) and compiled
   to Willow's native gateset (`map_and_compile_for_willow`, via
   `device.metadata.compilation_target_gatesets[0]`), then run through
   `cirq_google`'s calibrated noise model.

**Verified working end-to-end:** `L=1, N=5`, 100-point time sweep, 7 Trotter steps/point,
6 qubits mapped onto Willow — completes in under a minute and produces a sensible plot: the
noisy, hardware-mapped Cirq points track the ideal qutip curve closely at early times and
diverge more at later times. Not yet run at larger `L`/`N`.

**Backend switched from `DensityMatrixSimulator` to qsimcirq (2026-08-10, same day).**
Drew set up a second environment, `GoogleQVM` (conda, Python 3.12.13), with a working
qsimcirq install — `NuronSim.py`'s noisy sim now runs on it via
`sample_expectation_values` (shot-sampled trajectories, matching real hardware execution
and `QVMSetup.ipynb`'s pattern) instead of the exact-but-`4^n` density-matrix stopgap.
**Anything running `NuronSim.py`'s Cirq/noise path now needs the `GoogleQVM` env, not
`.venv`** — `.venv` still has no qsimcirq wheel for macOS arm64 + Python 3.14. `.venv`
remains right for qutip-only work (`Classical Simulation.py`). See D-14.

**Finding, not a bug: noisy circuits don't get better by raising `NumberOfTrotterSteps`
alone.** Drew ran `NumberOfTrotterSteps=100` (vs. the default 7) expecting Trotter error to
vanish and match qutip closely; instead got a flat line untethered from the qutip curve.
Confirmed via two independent noisy backends (`DensityMatrixSimulator` and qsimcirq) that
this is real: the sweep's `dt = t / NumberOfTrotterSteps` design means every point in the
sweep has the same gate count, set directly by `NumberOfTrotterSteps` — raising it from 7
to 100 multiplies accumulated noise at *every* point by ~14x, overwhelming the signal well
before Trotter error would matter. This is D-2's "depth 40, XEB 0.1%" argument playing out
live on the 6-qubit demo circuit. Full writeup, and how it was confirmed (noise-off control
run, cross-backend agreement): D-15.

**Adaptive `NumberOfTrotterSteps` per sweep point (2026-08-10) — does the D-15 crossover
sweep.** `src/optimal_trotter_steps.py` grid-searched 14 `t` values × 17 step counts (exact
expectation values, ~5-6 min, `GoogleQVM` env) and confirmed the trade-off is large: at
`t=6.5`, fixed `NumberOfTrotterSteps=7` gives error `0.57` against qutip; the per-point
optimum (12 steps) gives `0.04`. The raw per-point optimum is jagged because the system's
own dynamics are oscillatory, so a fixed step count aliases against them — a window-smoothed
version of the error surface was used to separate trend from artifact. `NuronSim.py`'s loop
uses `neuron_circuit.recommended_trotter_steps(t)` by default (`UseAdaptiveTrotterSteps=True`)
instead of one fixed count. See D-16.

**⚠ That schedule was retired the next day — the code has not caught up (2026-08-11,
D-17 + D-18).** Drew asked whether the low error at `t=9.5, r=2` was a coincidence rather
than a correct simulation, and argued the schedule should be monotone in `t`. Both correct,
confirmed by re-analysing the existing grid with no new circuit runs:

- **6 of the 14 argmins are artifacts** — isolated one-point dips, `E*` 4–17× below both
  neighbours in `r`. Every non-monotone dip in the schedule is driven by one.
- **Decisive check:** Trotter-only error at `t=9.5` is `0.03` at `r=2` but `0.70` at `r=8`.
  Trotter error cannot be 23× worse with 4× more steps — the `r=2` state is badly wrong and
  its two observables merely cross the true values.
- **At 7 of 14 points, adding noise *improved* the score** (`t=20, r=18`: noiseless `0.199`,
  noisy `0.021`). Noise was cancelling Trotter error.
- The `t≈12.5-15` "hard window" is therefore **not** a dynamical anomaly — it is the leading
  edge of a sweep cap. Past `t≈12` the circuit isn't Trotter-converged at any tested `r`.

**Consequences, both decided and neither implemented:** (D-17) replace the interpolated
schedule with the monotone one-parameter form `r*(t) = round(1.8·t)` — derived from
`ε_T ≈ A t²/r` plus `ε_N ≈ B r` giving `r* = t√(A/B)`, and fitted on the 8 clean points to
`c = 1.76 ± 0.27` — and **cap the sweep** where the required depth exceeds the noise budget
rather than reporting fitted artifacts. (D-18) choose the schedule by **noiseless state
fidelity** `|<ψ_qutip|ψ_trotter>|²` instead of the two-observable error metric, which is
what allowed the artifacts: two real numbers matched in a 10-dim Hilbert space leaves ample
room for accidental agreement. **Note fidelity is monotone in `r`, so it must not be
argmax'd** — it supplies a sustained-convergence threshold `r_T(t)`, with noise entering
separately as a budget constraint `r_max`. Full method: `notes/optimal-trotter-steps.md`.

**Scope caveat (unchanged, still binding):** all of this is fitted to one model config and
one Willow calibration snapshot — re-run if
`NumberOfFockStates`/`D_list`/`spin_interaction_coefficient`/the target processor change;
`check_trotter_schedule_config` warns (doesn't block) if they no longer match. D-17's
numbers (`c=1.76`, the `0.15` tolerance, the `t≈12` cap) are **metric-dependent and
provisional** until the D-18 fidelity re-run replaces them.

**Refactor alongside this (2026-08-10):** the Hamiltonian/circuit/Willow-mapping functions
that used to live directly in `NuronSim.py` moved to `src/neuron_circuit.py`, importable
without triggering NuronSim's own top-level sim+plot run. `optimal_trotter_steps.py` uses
the same module, so both scripts build circuits identically — no duplicated logic to drift
out of sync.

**D-17/D-18 implemented, plus post-selection and a second-order variant (2026-08-11).**
Full detail in D-19/D-20 and their linked notes files; summary here for the handover:

1. **Fidelity-derived schedule, implemented.** `src/trotter_fidelity_schedule.py`
   replaces the retired observable metric with `F(t,r)=|<psi_qutip|psi_trotter>|^2`,
   asserts the noiseless unary-leakage correctness check (`~0` everywhere, confirmed), and
   fits the monotone schedule `r*(t) = round(4.0 * t)`, capped at `t≈8.25` by the noise
   budget (`r_max=33`). `neuron_circuit.recommended_trotter_steps` now returns this; the
   old D-16 interpolation table is gone. `NuronSim.py` caps its own `Time` to the schedule
   cap automatically (one printed warning, not a warning per sweep point). D-19.
2. **Post-selection, implemented.** `neuron_circuit.sample_shots_with_postselection`
   switches the noisy path to per-shot sampling and discards shots where any boson column
   isn't exactly one-hot (D-4's mitigation, now real rather than argued-for). First full
   sweep: mean survival rate `53.2%`. D-19.
3. **Second-order (Strang) Trotter, implemented.** `src/NeuronSim2ndOrderTrotter.py` +
   `neuron_circuit.build_second_order_trotter_circuit`, with explicit boundary-layer
   merging. Caught and fixed a real bug along the way (an inner first-order-only building
   block silently capped the outer symmetric splitting at first-order convergence) via a
   log-log convergence-order check added to `src/consistency_checks.py` — worth reading
   as a case study in why "verify numerically" catches things inspection doesn't. D-20.
4. **`src/consistency_checks.py` added** — 15 executable checks (unary conservation,
   convergence order, gate-count structure, post-selection sanity, schedule monotonicity),
   all passing. Complements `src/symmetry_check.py` (D-9, unaffected, still passing).
5. Results in `results/`: `NuronSim_{noisy,noiseless}.png/.npz`,
   `NeuronSim2ndOrderTrotter_{noisy,noiseless}.png/.npz`, `trotter_fidelity_schedule.png`
   (+ cached grid `.npz`). Comparison writeup: `notes/first-vs-second-order-trotter-comparison.md`.

**Still open from this session:** no second-order-specific step schedule was derived (the
2nd-order script deliberately reuses the first-order-fitted schedule for a fair
matched-step comparison, D-20) — deriving one would let it use substantially fewer steps
for the same accuracy, a natural next step if the 2nd-order path becomes the default.

**Trotter schedule re-fit against the real (noisy, post-selected) metric, and linear
confirmed over quadratic (2026-08-12, D-21).** Prompted by a chat discussion of whether
`r` should scale as `t` or `t²` (Childs et al. predicts `t²` for a *fixed Trotter
tolerance*; D-17/D-18/D-19's schedule is `t`-linear because it solves a different problem
— minimizing total error under a noise cost that grows with `r`). Rather than re-deriving
this, ran a direct scan: `src/adaptive_trotter_model_scan.py` swept `r=a*t` and `r=b*t²`
with noise ON + post-selection ON against qutip, then `src/linear_trotter_finegrid_scan.py`
refined around the winner. **Linear won outright** (best RMS `0.157` @ `a=2` vs best
quadratic RMS `0.292` @ `b=0.6` — quadratic's required depth crushes post-selection
survival past `t≈7`). Fine grid found a clean unimodal optimum at **`a=2.25`** (parabolic
fit: `a≈2.26`) — about half of D-18/D-19's noiseless-fidelity-derived `c=4.0`, and close
to D-17's own (discarded) noisy-metric fit `c=1.76±0.27`. `neuron_circuit.py`'s
`TROTTER_SCHEDULE_C` is now `2.25` (`R_MAX=33` unchanged, re-verified; `T_CAP` moves
`8.25→14.67`, though that extension itself is unvalidated past the scan's tested `t≤10` —
see D-21's caveat). Outputs: `results/linear trotter/`, `results/quadratic trotter/`,
`results/linear_vs_quadratic_trotter_comparison.png`. **Confirmed end-to-end:** re-ran
`NuronSim.py`'s default sweep (`Time=10`) with the new constant —
`results/NuronSim_noisy_adaptive_postselect.png` tracks the qutip curve closely across
the whole sweep, mean post-selection survival improved to `68.2%` (was `53.2%` under
`c=4.0`), and the sweep no longer hits the cap warning at `Time=10` (new `t_cap≈14.67`).
**Still not done:** extending the scan past `t=10` to check whether `a=2.25` still holds
out to the new `t_cap≈14.67`, rather than needing to grow there.

**Second-order (Strang) gets its own fitted schedule, and re-loses to first order fairly
(2026-08-12, D-22).** Same day, same method as D-21, applied to
`build_second_order_trotter_circuit` (`src/second_order_trotter_linear_scan.py`):
coarse-then-fine scan, noise ON + post-selection ON. Result: optimal coefficient
`a=2.0` (parabolic fit `a≈2.15`) — close to first order's `a=2.25`, not markedly smaller,
because second order's faster `~r^-2` convergence is largely offset by its `~1.6x`
higher per-step gate cost (16 vs 10 two-qubit gates/step). Head-to-head at each order's
own optimum: first order RMS `0.1565` vs second order RMS `0.1657` — first order still
wins, but by only `~6%`, and second order is actually *better* for `t≲7` (first order has
a local error bump there) and worse only past `t≳7.5`, which is what decides the RMS
verdict. So D-20's "second order loses under noise" finding **survives** even once the
schedule-mismatch confound D-20 itself flagged is removed — this was a real result, not
an artefact of reusing first order's schedule. `neuron_circuit.py` now has parallel
second-order schedule constants/functions
(`TROTTER_SCHEDULE_C_2ND_ORDER=2.0`, `recommended_trotter_steps_2nd_order`, etc.,
`r_max=20`, `t_cap=10.0` — not extrapolated past the tested range, unlike D-21's cap);
`NeuronSim2ndOrderTrotter.py` uses them and was verified end-to-end (correctly caps its
default `Time=20` sweep to `10.00`,
`results/NeuronSim2ndOrderTrotter_noisy.png` tracks qutip sanely). Outputs:
`results/2nd order linear trotter/`,
`results/first_vs_second_order_optimized_trotter_comparison.png`.

**`NuronSim.py` fixed for `NumberOfBosonicModes>1` (2026-08-12).** Running it at `L=2`
crashed (`ValueError: Qubit pair is not valid on device`). Root cause: the qubit-chain
search only ever looked for a straight 1D device chain and mapped the circuit onto it in
`LineQubit` index order — correct only at `L=1`, where the logical topology genuinely is
a line. At `L≥2` the real topology is a comb (each site's boson chain, joined only via a
spine of spin-spin links at one end), which has no Hamiltonian path at all once `L≥3`
(provably — every site's far boson qubit is forced to be a path endpoint, and a path has
only two). Fixed by `neuron_circuit.find_low_error_qubit_embedding`, a real subgraph
embedding search onto Willow's actual 2D connectivity, replacing the old chain-only
search everywhere it was used (`NuronSim.py`, `NeuronSim2ndOrderTrotter.py`, and every
analysis script). Verified valid at `L=1–4`.

**VM benchmark figure suite produced, and it surfaced a real qubit-selection quality bug
in the fix above (2026-08-12, D-23).** `results/VMBenchmark/`: an `L=1,N=5`
step-count/noise/post-selection ablation (4 figures), an `L=1→3` sweep at `N=5,J=0.1` (3
figures), and an `N=3→7` sweep at `L=1` with a fresh per-`N` linear-schedule re-fit (5
figures + 5 schedule-fit diagnostics) — `src/vm_benchmark.py`, now with
`--part1`/`--part2`/`--part3`/`--n=` flags for targeted re-runs.

Drew noticed one figure was visibly worse than a nominally-identical existing result,
which led to finding that the embedding search above was structurally correct but
*low-quality*: it returned the first valid chain found rather than a good one, and even
once fixed to compare alternatives, scoring by two-qubit gate error alone missed a
qubit with `T1=39us` against a chip-wide `~70us` median (post-selection can't catch
T1-driven decoherence — it only checks the boson registers' one-hot constraint). Took
three rounds to fix properly: (1) compare multiple candidate chains instead of returning
the first valid one, (2) score by percentile rank of *both* two-qubit error and `1/T1`
(unit-free combination), (3) a hard floor excluding the chip-wide worst T1 decile
outright, since percentile scoring alone still let the same bad qubit through into two
configs where its neighbourhood's gate errors were good enough to outweigh it in the sum.
Full diagnostic trail in D-23.

This also explained an apparent physics anomaly Drew caught in the `N`-sweep: an
`N=5` outlier and a suspiciously flat `a`-vs-`N` trend both turned out to be artifacts of
each `N` getting an independently-unoptimized (and inconsistently noisy) chain, not real
signal — confirmed directly by replaying both candidate schedules on the actual chain
used and showing the "wrong" one really did win *on that specific chain*. After the fix
and two re-runs, `a` is flat at `2` across `N=3–6` (now trustworthy) and drops to `1.5`
at `N=7` — checked and NOT a repeat of the same bug (no T1 outlier), instead consistent
with `N=7` needing 7 required two-qubit edges per Trotter step vs `4–6` for smaller `N`,
i.e. possibly the first real sign of Trotter-error growth with `N` showing up. One data
point, not yet a confirmed trend — extending past `N=7` or adding resolution around
`N=6–7` would be the natural next step if this becomes load-bearing.

**Left deliberately unregenerated:** Part 2 (`results/VMBenchmark/05`–`07`, the `L=1→3`
sweep) still reflects the pre-fix embedding search and should be treated as stale if
examined closely — not redone since it wasn't this investigation's focus and its `L=3`
case alone costs ~3 hours to re-run.

**`NuronSim.py` feature additions, and a clean second entry point (2026-08-13).** Three
small features added to `src/NuronSim.py`/`src/neuron_circuit.py`, then the whole
pipeline copied into a decluttered `src_clean/` (D-24):

1. The D-17/D-18 sweep cap (auto-truncating `Time` to `TROTTER_SCHEDULE_T_CAP`) is
   disabled at Drew's request — the sweep now always runs to the full requested `Time`;
   the noise-budget warning still prints, it just no longer truncates.
2. `ShowQubitEmbeddingOverlay` toggle: `neuron_circuit.plot_qubit_embedding_overlay`
   draws the `willow_pink` calibration grid (T1 per qubit, two-qubit CZ error per bond)
   with the run's actual chosen qubits/edges highlighted in red, saved alongside the
   results plot — a visual sanity check on `find_low_error_qubit_embedding`'s choice.
   `src/willow_calibration_grid.py` (the plain, unhighlighted version) added alongside.
3. `ShowPostSelectionRemovedPoints` toggle: `sample_shots_with_postselection` now also
   returns `occ_removed`/`mag_removed` — the same per-point estimator as `*_post`, but
   over exactly the shots post-selection discarded — plotted as greyed-out points beneath
   the kept ones, so you can see what's actually being thrown away, not just the
   survival-rate number.

**`src_clean/`** (`neuron_circuit.py`, `NuronSim.py`, `G_analytic.py`) is a second,
minimal entry point: same run path and current best-known config (D-21's schedule,
D-23's embedding, D-4/D-19's post-selection, all three features above), with the
second-order circuit, the schedule-derivation tooling, and the R&D comment history
stripped out — see D-24 for the full accounting of what was kept/dropped and why.
Verified to reproduce `src/NuronSim.py`'s output end-to-end. Outputs go to
`results/clean/`, kept separate from `src/`'s outputs in `results/`. `src/` remains the
full historical record and the place to re-derive any constant; changes there don't
auto-propagate to `src_clean/`.

## Environment note

Shell access to this folder **works**: read, write, execute, delete.

The Linux sandbox (Python 3.10) is separate from the Mac and **resets between sessions**.
Only `numpy` ships by default.

**At the start of any chat that runs code:**

```
bash setup_env.sh
```

~15 s cold, ~1 s warm. Installs numpy, scipy, matplotlib, sympy, cirq-core, cirq-google,
qiskit, qiskit-aer and verifies every import. Pinned versions in `requirements.txt`.

No virtual environment is kept in this folder, deliberately — the stack is 286 MB of Linux
binaries that macOS cannot use, and reinstalling takes 15 s. Not worth the disk or the
path fragility (the sandbox mount path changes between sessions).

**qutip is not installed.** No cp310 manylinux wheel is reachable from this sandbox's
index, so pip falls back to a source build that did not finish in >6 minutes. It is also
unnecessary: `(2N)^L = 1728` at `L=3, N=6`, so dense numpy/scipy is ample. Revisit only if
step 4 needs qutip's open-system solvers — in which case the fix is to build the wheel once
and cache the `.whl` in this folder, not to build it every session.

Always keep `--only-binary=:all:` on pip commands. Without it pip silently falls back to
source builds that can hang for many minutes.

## References

Attached in project knowledge:

- **Model.md** — the source definition of `H_neuron` and the unary mapping
- **Willow spec sheet** (Google Quantum AI, Dec 2024) — qubit count, gate and readout
  errors, T1. Note the headline "depth 40" is quoted at 0.1% XEB; see D-2.
- **Childs, Su, Tran, Wiebe, Zhu — *A Theory of Trotter Error*** (PRX 11, 011020, 2021).
  Commutator-scaling bounds. Basis for step 2.
- **Sawaya, Menke, Kyaw, Johri, Aspuru-Guzik, Guerreschi — *Resource-efficient digital
  quantum simulation of d-level systems*** (npj QI, 2020). Unary vs. Gray vs. standard
  binary encodings of truncated bosonic operators; Trotter gate counts vs. `d`. Directly
  challenges D-4 — see the amendment there. Key figure for us is **Fig. 6, top row**
  (`q̂`, tridiagonal).

## Layout

```
CLAUDE.md      auto-loaded by Claude Code at session start; points here
PROJECT.md     this file — outline, status, open questions
DECISIONS.md   running log of what we chose and why
requirements.txt / setup_env.sh   sandbox environment (see above)
notes/         per-topic analysis
src/           code — full R&D history, both Trotter orders, all derivation scripts
src_clean/     clean first-order-only pipeline (current best-known config only, D-24)
results/       generated output — disposable, regenerable from src/ (gitignored)
results/clean/ output from src_clean/ — kept separate from src/'s outputs
```

## Which tool for which job

This folder is worked on from two places, sharing `PROJECT.md` + `DECISIONS.md` as state.

| | Claude Code (terminal, on the Mac) | Claude desktop app |
|---|---|---|
| Best for | steps 1–5: simulation, circuits, noise models | design discussion, reading papers, written deliverables |
| Shell | native macOS, no timeout, persistent venv | sandboxed Linux, 45 s tool timeout, resets each session |
| qutip | installs from an arm64 wheel, stays installed | no wheel available, source build does not complete |
| git | native | none |

Rule of thumb: **if it runs for more than a minute or needs qutip, use Claude Code.**
