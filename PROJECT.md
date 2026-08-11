# Quantum Network of Neurons — Project State

**Read this file and `DECISIONS.md` at the start of every new chat.**
Update both at the end of every chat.

Last updated: 2026-08-11

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
| 2 | Trotter error analysis — Childs et al. commutator bounds on our 3-way split; test term orderings | D-17 derives the `r* ∝ t` scaling; D-18/D-19 implement the fidelity-based schedule (`src/trotter_fidelity_schedule.py`, `c=4.0`, capped `t≈8.25`); D-20 adds a second-order (Strang) alternative (`src/NeuronSim2ndOrderTrotter.py`) with an explicit convergence-order check |
| 3 | Circuit synthesis and Willow layout — confirm SWAP-free embedding, pick qubit patch from calibration | prototyped at small scale (`src/NuronSim.py`, `L=1,N=5`); not yet run at the `L=3,N=6` target |
| 4 | Noise simulation + post-selection — realistic error model, quantify unary-constraint recovery | noise model wired up (`src/NuronSim.py`, D-14); D-15→D-18/D-19 replaced the fixed/observable-metric step schedule with a fidelity-derived, capped, monotone one; **post-selection implemented (D-19)** — `neuron_circuit.sample_shots_with_postselection` filters shots on the unary one-hot constraint (D-4), mean survival rate `53.2%` in the first full sweep |
| 5 | Configuration sweep — (L, N, steps) vs. observable fidelity; justify final choice | not started |

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
src/           code
results/       generated output — disposable, regenerable from src/ (gitignored)
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
