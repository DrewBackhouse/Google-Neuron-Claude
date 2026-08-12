# Decisions Log

Append-only. Each entry: what we decided, why, and what would overturn it.

---

## D-1 — Do not use the full 105-qubit grid

**Decided:** target `L=3–4`, `N=6–8` (21–36 qubits). Use a small patch of Willow.

**Why:** the binding constraint is total two-qubit gate count against per-gate error, not
qubit count. `L=5, N=20` fills the chip exactly (105 qubits) but costs ~208 two-qubit
gates *per Trotter step*, giving ~0.75 fidelity for a single step before readout — and
105-qubit readout at 0.67%/qubit is ~0.49 on its own. One Trotter step lands near 0.37.
Unusable. See `notes/resource-estimate.md`.

Corollary: the unused qubits are worth more as headroom for choosing the highest-fidelity
patch of the chip than as larger `L` or `N`.

**Overturned by:** substantially better two-qubit error rates, or an encoding that cuts
the per-step gate count (see D-4).

---

## D-2 — "Depth 40" from the spec sheet is not our budget

**Decided:** budget against cumulative two-qubit gate count and measured error rates, not
against the spec sheet's depth figure.

**Why:** the 105-qubit / depth-40 number is the random-circuit-sampling benchmark, and at
that depth XEB fidelity is **0.1%** — that is where the signal has decayed to nothing, not
a depth at which expectation values are trustworthy. For an observable-based experiment the
relevant quantity is `(1 - ε_2q)^(gate count)`.

---

## D-3 — Break site symmetry via `D_j`, not the initial state

**Decided:** vary `D` per site. Keep the initial state as vacuum.

**Why:** with uniform `D` and all modes in `|0>`, the state is permutation-symmetric across
sites — every neuron climbs in lockstep and fires simultaneously, and `H_spin` acting on a
uniform spin configuration transports nothing. No causal propagation to observe. Symmetry
must be broken somehow.

Varying `D_j` is free: different rotation angles in an otherwise identical circuit, zero
extra gates. Vacuum is the cheapest initial state in unary — `|0> = |100...0>`, one X gate
per column, no two-qubit gates.

**Note:** coherent-state prep in unary is cheap if we later want it — an arbitrary
single-excitation state on `N` qubits is a Givens cascade, `N-1` two-qubit gates, depth
`N-1`, with real angles for real α. Available, just not needed by default. (This is one of
the places unary earns its qubit cost; the same prep in Gray code is far worse.)

---

## D-4 — Unary encoding, despite the qubit cost

**Decided:** keep unary. Do not switch to binary/Gray-code Fock encoding.

**Why:** unary confines each boson column to its single-excitation subspace — a hard
constraint that noise does not respect. Every shot with ≠1 excitation in a column is
detectably corrupt and can be discarded, for free, with no extra circuitry. It covers
`L·N` of `L·(N+1)` qubits: at `N=6` that is 6/7 of the register.

Unary also keeps every mapped Pauli string nearest-neighbour on the `L × (N+1)` grid,
which is what buys the SWAP-free embedding. Gray code would give `log₂ N` qubits per mode
but non-local Pauli strings, no symmetry to post-select on, and expensive state prep.

**Caveat to quantify in step 4:** post-selection catches errors that leave the
single-excitation subspace (bit-flip / amplitude-damping type). Phase errors *within* the
subspace pass through undetected. So this is not a free 6/7 fidelity recovery — closer to
half the error channels.

### Amended 2026-07-29 — challenged by Sawaya et al.

Sawaya et al. (`refs/`, npj QI 2020) find that for the tridiagonal position operator `q̂`
— i.e. our `(a+a†)` up to `√2` — **unary is inferior to Gray and SB in both qubit count and
gate count at d = 4, 7, 8**. They flag this explicitly as counterintuitive. Our candidate
`N = 6, 8` sits in that region.

Decision **stands**, but now rests on three arguments rather than on locality alone:

1. **Their counts ignore connectivity.** Gate counts assume free qubit interaction. Willow
   is fixed degree-4. Unary is nearest-neighbour and SWAP-free; compact codes produce
   non-local strings needing SWAP networks at 3 CNOTs each. That overhead is absent from
   their figures and could erase the compact-code margin entirely.
2. **`H_CNOT` is not an operator they study.** `|N-1><N-1| ⊗ X` projects onto a *single*
   level: in unary it is one qubit's `|1><1|`, so a plain controlled rotation (~2 gates).
   In Gray/SB it is a multi-controlled gate over `⌈log₂ N⌉` qubits — for `N=8` a
   3-controlled operation, and non-local. Their studied operators (`q̂`, `q̂²`,
   `a†_i a_j + h.c.`) do not include anything of this form.
3. **Post-selection is unavailable in compact codes.** Every bitstring is a legal Fock
   state in Gray/SB, so there is no unphysical subspace to detect leakage into. D-4's
   mitigation argument has no compact-code analogue.

**To verify in step 3:** read actual gate counts off their Fig. 6 at our exact `N`, add
realistic SWAP overhead for a degree-4 lattice, and add the `H_CNOT` cost. Only then is the
comparison honest. If unary loses even after all three corrections, revisit.

**Trap to avoid:** the paper notes that in compact encodings it is often worth *raising* `d`
to a power of two, since gate count falls while qubit count does not. **Not available to
us** — `N` sets the firing threshold and is a physical parameter, not a convergence knob.
We are in the same position as their spin-`s` case, where padding causes leakage into
unphysical states.

**Option held in reserve:** they show inter-encoding conversion is cheap enough to be worth
doing *within* a Trotter step (compact for some terms, unary for others). Probably not
worthwhile for us given argument 2, but revisit if `H_boson` turns out to dominate the
per-step gate count.

---

## D-5 — No exact symmetry available in the spin sector

**Decided:** accept that the `L` spin qubits are not covered by post-selection. Plan for a
generic method there (ZNE, or echo/mirror calibration).

**Why:** `H_spin` conserves `Σ_j Z_j`; `H_CNOT`, being `∝ X_j` on the spin, conserves
`Σ_j X_j` instead. Different U(1)s, no common conserved quantity. Rotating the spin qubits
to the ± basis does not fix this — under Hadamard conjugation `H_spin → (J/2)(Z_jZ_{j+1} +
Y_jY_{j+1})` and `H_CNOT → G|N-1><N-1| ⊗ Z_j`, and the two still share no symmetry.

*(Flagged as a quick derivation — worth re-checking before relying on it.)*

### CORRECTED 2026-07-29 — the claimed compilation win does not exist

The original entry claimed the ± rotation gives "fewer gates and lower Trotter error via
merging `H_CNOT` with the `ZZ` part of `H_spin`". **This is wrong.** A global single-qubit
basis change is a unitary equivalence: `[UAU†, UBU†] = U[A,B]U†`, so Trotter error is
identically preserved, and entangling-gate counts are preserved because two-qubit local
equivalence classes are.

The specific error: `H_CNOT ∝ Z_a X_b` *already* commutes with the `XX` part of `H_spin` in
the original frame and fails against `YY`. After rotation it commutes with `ZZ` and fails
against `YY`. Same structure, relabelled.

The conclusion that the spin sector has **no exact U(1)** still stands — but see D-9, which
supersedes the pessimism.

---

## D-6 — Frame this as demonstration + benchmark, not quantum advantage

**Decided:** do not claim classical intractability anywhere in the writeup.

**Why:** the accessible Hilbert space is `(2N)^L`. At the sizes that are hardware-viable
(`L=3, N=6`) that is ~1.7×10³ states — trivial. Even the ruled-out `L=5, N=20` is ~10⁸,
which is well within reach of sparse methods.

The genuine contributions are: a structured spin–boson simulation with a real
non-quadratic element (`H_boson` and `H_spin` are each free/quadratic on their own — the
`H_CNOT` term is what makes the model non-trivial), a SWAP-free hardware-native mapping,
and a symmetry-protected error-mitigation scheme. That is a good paper on its own terms.

---

## D-7 — Project state lives in files, not chat history

**Decided:** one chat per outline step; `PROJECT.md` + `DECISIONS.md` carry context between
them.

**Why:** context does not transfer automatically between chats. Prior-session transcripts
can technically be read but it is fragile and noisy. Explicit files are durable, and Drew
can read them too.

---

## D-8 — Local working folder, not iCloud

**Decided:** use `~/Documents/Claude/Google Neuron`. The earlier iCloud folder is retired.

**Why:** iCloud's "Optimise Mac Storage" evicts local file copies and leaves dataless
placeholders that the shell cannot read; iCloud also races against `.git` internals and
corrupts repos. A plain local folder avoids both.

**Resolved 2026-07-29:** shell access to the new folder is working after an app restart.
Read, write, execute and delete all verified. The old iCloud folder is still connected in
the app but is not to be used.

---

## D-9 — Exact Z2 symmetry in the spin sector: `Π_j X_j`

**Found 2026-07-29.** Supersedes the pessimism in D-5.

**The result:** `Π_j X_j` — the product of `X` over all `L` spin qubits — commutes exactly
with the full Hamiltonian.

Verified numerically to machine precision at `(L,N) = (2,4), (3,3), (3,5), (4,3)` with
site-dependent `D_j` (`src/symmetry_check.py`). `||[H, Π X]|| = 0` in every case;
`||[H, Π Z]||` is large, as expected.

**Why:**

- `H_spin`: the `XX` term commutes with every `X`; the `YY` term picks up two sign flips
  that cancel.
- `H_CNOT`: `∝ X` on the spin, so commutes. The `|N-1><N-1|` projector acts on boson
  qubits and is inert.
- `H_boson`: acts on boson qubits only.

Because the projector plays no role, the symmetry survives **any** `D_j`, any `G_j`, and
any boundary condition. It is frame-independent — it was always present.

**What the ± rotation buys:** in the rotated frame the symmetry becomes `Π_j Z_j`, i.e.
plain bitstring parity in the computational basis. Readable for free alongside the readout
already needed for unary post-selection, with no extra basis-change gates. Bookkeeping
rather than resources — but it is what makes the symmetry practically usable.

**The catch — open decision.** The natural initial state (all modes `|0>`, all spins down)
has `<Π_j X_j> = 0` *exactly*: an equal superposition of both parity sectors. There is no
expected value to post-select on. Options:

1. **Initialise spins in `|±>`** (`|0>`/`|1>` in the rotated frame). Definite sector, clean
   post-selection, zero gate cost. But the neuron starts in a superposition of fired and
   not-fired — an interpretational cost that needs defending.
2. **Keep `|↓>`, use the symmetry diagnostically.** Separate calibration runs in a definite
   sector to characterise spin-row error rates, applied as a correction. No shot-by-shot
   mitigation.
3. **GHZ-like initial state** `(|↓..↓> ± |↑..↑>)/√2`. Definite sector but costs `L-1`
   entangling gates, fragile, and the interpretation is worse.

**Strength:** one Z2 bit across `L` spin qubits — catches only odd numbers of bit-flips on
that row. Much weaker than the unary constraint (`L` independent checks over `LN` qubits).
Free, but modest. Do not over-sell it.

---

## D-10 — Adopt the ± form as the canonical Hamiltonian  ~~[SUPERSEDED by D-11]~~

> **Superseded 2026-07-29.** Retained for the reasoning only. We stay in the computational
> basis with a controlled-X interaction. See D-11.

**Decided:** write the model in the spin-rotated frame from here on. `U = ⊗_j H_{spin,j}`
(Hadamard on every spin qubit; boson qubits untouched).

```
H̃_boson = (1/2) Σ_j D_j Σ_i √(i+1) (X_{i,j} X_{i+1,j} + Y_{i,j} Y_{i+1,j})     [unchanged]
H̃_spin  = (J/2) Σ_j (Z_{N,j} Z_{N,j+1} + Y_{N,j} Y_{N,j+1})
H̃_CNOT  =  G  Σ_j |1><1|_{N-1,j} Z_{N,j}  =  (G/2) Σ_j (Z_{N,j} − Z_{N-1,j} Z_{N,j})
```

Initial state: spins `|0> → |+>`. Symmetry: `Π_j Z_{N,j}`.

**Why:** not for resources — see the D-5 correction, there are none. Because it puts both
`H_CNOT` and the conserved parity **diagonal in the measurement basis**, which makes
readout and post-selection reasoning direct rather than requiring a mental basis change at
every step.

**Cost — corrected 2026-07-29.** Entangling-gate count and Trotter error are *identically*
preserved (unitary equivalence). But the single-qubit overhead moves slightly **against**
the rotation:

- state prep costs `L` extra Hadamards (`|0>` was free, `|+>` is not);
- measurement is a wash — parity `Π Z` becomes free to read, but the fired/not-fired spin
  `Z` now costs `L` Hadamards. The rotation chooses *which* observable is free;
- `XX + YY` is the better native match. Willow's RCS entangler is iSWAP-like and `XX + YY`
  *is* that interaction directly; `ZZ + YY` is the same Weyl class but needs an
  axis-permutation Clifford dressing on each spin–spin gate, which is not a virtual-Z and
  costs real pulses (~`2(L-1)` extra single-qubit gates per Trotter step).

Single-qubit error is 0.036% vs 0.14% two-qubit — **~4× cheaper, not ~10× as first written**.
Over ~40 steps the extra gates cost a few percent of fidelity. Small, but real, and in the
wrong direction.

**Therefore: the frame is a presentation choice, not a physics choice.** Use ± for the
write-up and analysis, where having `H_CNOT` and the conserved parity both diagonal makes
the reasoning direct. Do **not** bake it into the circuit — at compile time let the
transpiler pick whichever frame minimises gates, since the two are physically identical.

**Note:** `H̃_CNOT` splits into a single-qubit `Z` rotation on the spin plus a `ZZ` coupling
between the top boson qubit `(N-1, j)` and the spin qubit `(N, j)` — nearest-neighbour on
the grid, so still SWAP-free.


---

## D-11 — Stay in the computational basis, controlled-X interaction

**Decided:** keep the original form. Supersedes D-10.

```
H_boson = Σ_j D_j (a_j + a_j†)
H_spin  =  J  Σ_j (σ+_j σ-_{j+1} + σ-_j σ+_{j+1})
H_CNOT  =  G  Σ_j |N-1><N-1|_j X_j
```

Mapped to qubits (unary):

```
H_boson = (1/2) Σ_j D_j Σ_i √(i+1) (X_{i,j} X_{i+1,j} + Y_{i,j} Y_{i+1,j})
H_spin  = (J/2) Σ_j (X_{N,j} X_{N,j+1} + Y_{N,j} Y_{N,j+1})
H_CNOT  =  G   Σ_j |1><1|_{N-1,j} X_{N,j}
```

Initial state: all modes `|0>` (unary `|100...0>`), all spins `|0>` (down). No two-qubit
gates in state prep.

**Why:**

1. The ± rotation has **no resource benefit** and is slightly negative — see the D-10 cost
   correction. Entangling count and Trotter error are identical by unitary equivalence;
   single-qubit overhead goes the wrong way.
2. `XX + YY` **is** Willow's native iSWAP-like interaction, so `H_spin` compiles with
   minimal dressing. `ZZ + YY` needs an axis-permutation Clifford on every spin–spin gate.
3. State prep is free: spins start in `|0>`, not `|+>`.
4. The primary observable — has neuron `j` fired? — is the spin's computational-basis
   value. Directly readable, no basis change.
5. The neuron interpretation is cleanest here: `|↓>` is unambiguously "resting",
   `H_CNOT` is unambiguously "flip the spin".

**Consequence for D-9.** The conserved quantity is `Π_j X_j`. In this frame it is *not*
diagonal, so measuring it costs `L` Hadamards on the spin qubits immediately before
readout — once per circuit, single-qubit, negligible. The symmetry itself is unaffected;
only its readout cost changes, and the change is trivial.

Note this makes the two readouts **mutually exclusive in a single shot**: the spin's `Z`
(fired/not-fired) and the parity `Π X` do not commute. Either split shots between two
measurement settings, or accept that parity is a diagnostic run rather than a per-shot
filter. Combined with the initial-state issue in D-9, this reinforces option 2 there — use
the symmetry diagnostically, not as a post-selection filter.

**The frame stays free at compile time.** The two descriptions are physically identical, so
the transpiler may rotate internally if that reduces gate count. This decision fixes how we
*write and reason about* the model, not how it is compiled.

---

## D-12 — `NuronSim.py`'s noisy Cirq sim: `DensityMatrixSimulator`, not qsimcirq  ~~[SUPERSEDED by D-14]~~

> **Superseded 2026-08-10, same day.** Drew set up a second environment (`GoogleQVM`
> conda env, Python 3.12) with a working qsimcirq install. Retained for the reasoning
> only — see D-14 for the current backend and D-15 for what this decision's stopgap
> backend accidentally made hard to notice.

**Decided (2026-08-10):** `src/NuronSim.py`'s Cirq simulation now explicitly maps its
circuit onto real Willow (`willow_pink`) `GridQubit`s and runs it through
`cirq_google`'s calibrated noise model, but via `cirq.DensityMatrixSimulator(noise=...)`
rather than the qsimcirq-backed QVM engine pattern shown in `src/QVMSetup.ipynb`.

**Why:** `pip install qsimcirq` fails a from-source build in this project's macOS venv
(no prebuilt wheel for arm64 + Python 3.14; the `cmake` build step errors out). `cirq`'s
own `DensityMatrixSimulator` accepts the identical
`cirq_google.NoiseModelFromGoogleNoiseProperties` object, so the noise model itself is
unaffected — only the execution backend differs. It also gives exact expectation values
(diagonal of the density matrix) rather than shot-sampled ones, which is strictly better
at the small qubit counts `NuronSim.py` currently runs (6 qubits, `L=1, N=5`: ~58 s for a
100-point time sweep, 7 Trotter steps/point).

**What this does not solve:** density-matrix simulation costs `4^n`. The step-4 target
size (`L=3, N=6` → 21 qubits) is far beyond it — that would need qsimcirq's trajectory
(Monte Carlo) sampling instead, run via `sim.run(circuit, repetitions=...)` with
measurement gates, exactly as `QVMSetup.ipynb` does. `NuronSim.py`'s current pipeline
(qubit-chain search in `find_low_error_qubit_chain`, mapping + native-gateset compile in
`map_and_compile_for_willow`) is backend-agnostic — swapping in qsimcirq once a wheel is
available (a different Python version, Linux, or a prebuilt binary) only changes the
`noisy_sim = ...` line and the observable extraction (diagonal-of-density-matrix →
measurement statistics), not the mapping/compilation logic.

**Overturned by:** a working qsimcirq install in this environment, or moving step 4's
noise work to a machine where one is available.

---

## D-13 — Willow qubit-patch choice: greedy low-error chain search, not a fixed patch

**Decided (2026-08-10):** `NuronSim.py` picks its hardware qubit patch at runtime via
`find_low_error_qubit_chain` — a backtracking DFS over the device's connectivity graph
that greedily prefers edges with the lowest calibrated two-qubit XEB Pauli error
(`cirq_google`'s median calibration, `two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle`),
constrained to follow real device edges so the result is SWAP-free by construction.

**Why:** the model's own qubit graph is a simple path (each boson column is a line of
`N` qubits, capped by one spin qubit) — exactly the corollary in D-1: unused capacity on
the 105-qubit chip is worth more as a choice of high-fidelity patch than as larger `L`/`N`.
A fixed hand-picked patch would drift out of date as calibration changes between runs;
searching against the *current* median calibration each run keeps the choice current
without hand-tuning.

**Caveat:** this is a greedy heuristic (cheapest local edge, backtrack on dead ends), not
a proof of the globally lowest-error chain of that length — fine at the sizes tried so far
(6 qubits, near-instant search), not verified to stay fast or near-optimal at the L=3, N=6
target (21 qubits).

---

## D-14 — `NuronSim.py`'s noisy Cirq sim: qsimcirq, via the `GoogleQVM` conda env

**Decided (2026-08-10).** Supersedes D-12. Drew set up a second environment —
`GoogleQVM`, a conda env at `/opt/miniconda3/envs/GoogleQVM`, Python 3.12.13 — with a
working `qsimcirq` install (qutip and cirq-google are in it too). `NuronSim.py`'s noisy
Cirq sim now runs on `qsimcirq.QSimSimulator(noise=willow_noise_model)`, replacing D-12's
`cirq.DensityMatrixSimulator` stopgap.

**What changed mechanically:** `QSimSimulator` doesn't expose a single "final state" for
a noisy circuit (each run is one Monte Carlo trajectory), so exact expectation values
aren't available the way `DensityMatrixSimulator.simulate().final_density_matrix` gave
them. Observable extraction moved to `sample_expectation_values(circuit, observables,
num_samples=...)`, which is exactly the shot-sampled, trajectory-noise pipeline real
hardware execution uses (matching `QVMSetup.ipynb`'s `sampler.run(..., repetitions=...)`
pattern). `compute_observables_from_probs` (needed the full `2^n` joint diagonal) was
replaced with `compute_observables_from_z_expectations` (needs only the single-qubit
`<Z_j>` marginals `sample_expectation_values` returns directly) — both bosonic occupation
and spin magnetisation were always sums of single-qubit marginals, so nothing was lost by
dropping the joint distribution, and it's what makes the estimate cheap.

**New tuning knob:** `NumberOfNoiseSamples` (shots per time point, default 2000 — chosen
because it matched the exact `DensityMatrixSimulator` values closely in testing, see D-15).
Runtime scales ~linearly with it; shot noise scales ~`1/√samples`. Cross-checked against
D-12's exact density-matrix values at `NumberOfTrotterSteps=7`: 2000-shot qsimcirq
estimates agreed to within a few percent.

**Why keep both environments rather than fixing `.venv`:** `.venv` (macOS arm64, Python
3.14) still has no reachable `qsimcirq` wheel and fails a from-source build (D-12) —
that's a platform/Python-version gap, not something pinning a requirement fixes. `.venv`
remains fine for qutip-only work (`Classical Simulation.py`); anything running
`NuronSim.py`'s Cirq/noise path needs `GoogleQVM`.

**Overturned by:** a working qsimcirq wheel landing for macOS arm64 + Python 3.14 (at
which point the two environments could plausibly be merged).

---

## D-15 — The "flat line at high `NumberOfTrotterSteps`" is noise physics, not a bug

**Found 2026-08-10.** Drew ran `NuronSim.py` with `NumberOfTrotterSteps=100` (up from the
default 7, intending to make Trotter error negligible) and got a Cirq output that was
flat across the whole time sweep and did not track the qutip curve at all. Investigated
and confirmed **not a software bug** — it's the correct output of the noise model given
how the sweep is structured.

**Mechanism:** the execution loop is "fixed Trotter-step count, varying total time `t`"
(`dt = t / NumberOfTrotterSteps`, comment already in the code). That means **every** point
in the time sweep runs a circuit with the same gate count — `NumberOfTrotterSteps` sets
circuit depth directly and independently of `t`. Raising it from 7 to 100 doesn't just
shrink Trotter error at large `t`; it multiplies the two-qubit gate count (and therefore
accumulated noise) at *every* point in the sweep by ~14×. At 100 steps the accumulated
gate error overwhelms the signal well before the noise model's fixed point is reached
(all boson qubits' population settling near a T1/mixed-state value, spin `<Z>` → 0),
*regardless of `t`* — hence a flat line. This is D-2's "depth 40, XEB 0.1%" finding,
reproduced live on this 6-qubit demo circuit instead of the 105-qubit RCS benchmark it
was originally measured on.

**How this was confirmed, not just argued:** ran the same compiled circuit through
`cirq.DensityMatrixSimulator(noise=None)` (noise off) at `NumberOfTrotterSteps=7` and
`=100` — both matched the qutip curve closely, and `=100` matched *better* than `=7`
(Trotter error shrinking as expected). Then independently reproduced the *noisy* flat-line
collapse via two different simulation methods — `DensityMatrixSimulator` (exact) and
qsimcirq `sample_expectation_values` (Monte Carlo trajectories, D-14) — and got the same
answer from both. Two independent backends agreeing rules out a backend-specific bug.

**Consequence — the two knobs trade against each other, they don't compose freely.**
`NumberOfTrotterSteps` cannot be raised freely to kill Trotter error on real hardware (or
in a hardware-realistic noisy simulation): raising it directly spends noise budget at
every sweep point. The default (`NumberOfTrotterSteps=7`) sits closer to a usable
trade-off; `=100` is well past where signal survives on Willow's calibrated error rates at
this gate count. This is a genuine, useful demonstration of the D-1/D-2 gate-count
argument, not a defect to fix — the code is behaving correctly.

**Not yet done:** finding where the crossover actually sits (some intermediate step count
where Trotter error and noise are both small) — would need a step-count sweep with the
noise model on, comparing against the qutip curve. Worth doing before treating any
particular `NumberOfTrotterSteps` choice as "the demo config."

**Update 2026-08-10, same day.** `NuronSim.py` now has a `UseNoiseModel` input flag
(default `True`). `False` runs the same Willow-mapped/compiled circuit through
`qsimcirq`'s *exact* `simulate_expectation_values` instead of the shot-sampled noisy path
— fast (no sampling overhead) and isolates Trotter error from noise, which is exactly what
was used above to confirm this decision (and is the tool the still-open crossover sweep
needs).

**Update 2026-08-10, later same day — crossover found.** The "not yet done" step-count
sweep is done; see D-16.

---

## D-16 — Adaptive `NumberOfTrotterSteps` per sweep point, not one fixed value  ~~[SUPERSEDED by D-17]~~

> **Superseded 2026-08-11.** The *premise* survives — a fixed step count is a poor
> compromise across the sweep, and step count should vary with `t`. What does not survive
> is the **schedule**: the per-point argmin it interpolates is contaminated by accidental
> observable crossings, and six of its fourteen points are artifacts. Retained for the
> reasoning and for the aliasing finding, which D-17 builds on directly. See D-17.

**Decided (2026-08-10).** `NuronSim.py`'s execution loop now picks `NumberOfTrotterSteps`
per time point via `neuron_circuit.recommended_trotter_steps(t)` (`UseAdaptiveTrotterSteps`,
default `True`) instead of using one fixed value for the whole sweep.

**Why:** does the crossover sweep D-15 flagged as open. Grid search (exact expectation
values, not shot-sampled — `src/optimal_trotter_steps.py`, full method and results in
`notes/optimal-trotter-steps.md`) over 14 `t` values × 17 step counts confirmed the
trade-off is real and large: at `t=6.5`, fixed `NumberOfTrotterSteps=7` gives error `0.57`
against the qutip curve; the per-point optimum (12 steps) gives `0.04`. A single fixed
count is a poor compromise across most of the sweep — simultaneously too many steps (too
much noise) at small `t` and too few (too much Trotter error) at some larger `t`.

**Finding along the way, worth keeping separate from the schedule itself:** the raw
per-point-argmin step count vs `t` is jagged, not smooth — `1, 2, 6, 6, 12, 3, 2, 22, 26,
22, 26, 8, 5, 18`. This is not statistical noise (these are exact expectation values). The
system's own dynamics are oscillatory (Rabi-like flopping), so error at a *fixed* step
count doesn't grow monotonically with `t` — it beats/aliases against the true oscillation
period, confirmed directly by checking that trotter-only error (noise off) at a fixed low
step count swings between ~0 and ~1 across the sweep rather than climbing steadily. A
3-point moving-window smoothing of the error surface along `t` (reusing the already
computed grid, no extra circuit evaluations) separates the systematic trend from this
artifact; `recommended_trotter_steps` interpolates the *smoothed* schedule, not the raw one
— using the raw jagged curve would have baked in each sampled point's lucky/unlucky phase
alignment into the other ~86 points of the real 100-point sweep it wasn't computed at.

**Also found, not resolved:** a genuinely hard window around `t ≈ 12.5-15` where even the
best available step count in the tested range (up to 40) leaves error around `0.37-0.41` —
markedly worse than the rest of the sweep (`~0.02-0.22` elsewhere). Mechanism not
investigated (plausibly a high-sensitivity region of the true trajectory); flagged as open
in the notes file rather than asserting a cause without checking.

**Scope of validity — this is a fitted result, not a general law.** The schedule was
derived for one specific config (`NumberOfBosonicModes=1, NumberOfFockStates=5,
D_list=[1.0], spin_interaction_coefficient=0.5`, `G≈2.03`) against one noise snapshot
(`willow_pink`'s median calibration at analysis time). `neuron_circuit.
check_trotter_schedule_config(...)` warns (print, not raise) if the live config no longer
matches what the schedule was fit to. Changing `N`, `D_list`, `J`, or the target processor
needs a re-run of `src/optimal_trotter_steps.py` (~5-6 min, `GoogleQVM` env) — this is not
yet automated.

**Overturned by:** re-running the analysis at a different config or after Willow's
calibration drifts — the schedule is a snapshot, not something that stays correct on its
own.

---

## D-17 — Monotone one-parameter Trotter schedule, and cap the sweep where it's infeasible

**Decided (2026-08-11).** Supersedes D-16's schedule (not its premise). Two changes:

1. Replace `recommended_trotter_steps(t)`'s interpolated per-point argmin with the
   monotone one-parameter form **`r*(t) = round(1.8 · t)`**.
2. **Cap the sweep** at the largest `t` where the required step count is actually
   affordable under the noise budget, rather than reporting a fitted low-error point
   beyond it.

**Why — Drew's objection, checked and confirmed.** Drew asked whether the low error at
`t=9.5, r=2` in D-16's grid was a coincidence rather than a correct simulation, and
argued that a physically consistent schedule should be monotonically increasing in `t`.
Both parts are right. Re-analysed the *existing* grid
(`results/optimal_trotter_steps_grid.npz`) — no new circuit evaluations needed.

**Evidence 1 — the argmins are two distinct populations.** Scoring each argmin by
`E*/mean(immediate neighbours in r)`, the 14 points separate cleanly at 0.40 vs 0.53 with
a gap (the 0.5 cutoff is descriptive, not tuned):

- **Basin** (8 pts, ratio 0.53–0.93): `t = 0.5, 3.5, 5.0, 6.5, 11, 12.5, 14, 15.5`
- **Needle** (6 pts, ratio 0.06–0.40): `t = 2.0, 8.0, 9.5, 17, 18.5, 20` — `E*` is 4–17×
  below both neighbours, i.e. an isolated one-point dip with no supporting structure.

Every non-monotone dip in D-16's smoothed schedule is driven by needles (the drop to 3
steps at `t=9.5` comes from the `t=8.0` and `t=9.5` needles; the drop to 8 at
`t=17–18.5` from those two). The 3-point smoothing was designed to suppress aliasing in
`t`; it does not suppress this, because the artifact is narrow in `r`, not in `t`.

**Evidence 2 — the noiseless column is decisive.** Trotter-only error at `t=9.5`:

```
r  =    1     2     3     4     5     6     7     8    10    12  ...   40
E0 = 0.15  0.03  0.15  0.11  0.09  0.30  0.51  0.70  0.54  0.06  ...  0.02
```

Trotter error cannot be 23× *worse* at `r=8` than at `r=2`. The `r=2` state is badly
wrong; its two observables happen to cross the true values. Confirms Drew's hypothesis
directly. Same pattern at `t=8.0` (`r=4 → 0.04`, `r=8 → 0.89`).

**Evidence 3 — noise "improves" accuracy at half the points.** At 7 of 14 argmins the
noisy error is *lower* than the noiseless error at the same `r`. Worst case `t=20, r=18`:
noiseless `0.199`, noisy `0.021` — adding decoherence improved agreement 10×. Only
possible if noise is cancelling Trotter error, which is the plainest available statement
that the metric was scoring coincidences rather than accuracy.

**Why monotone is the right constraint, with a derivation rather than an intuition.**
First-order Trotter gives `ε_T ≈ A t²/r`; two-qubit gate count is `∝ r` so `ε_N ≈ B r`.
Minimizing the sum over `r`:

```
r* = t · √(A/B)      linear in t, monotone
E* = 2t · √(A B)     grows linearly in t
```

The D-16 table violates the second prediction badly (`E* = 0.021` at `t=20` but `0.223`
at `t=3.5`), which is independent confirmation the argmin was not tracking physics.

**The fit.** `c = r*/t` on the **basin points only**: `c = 1.76 ± 0.27` (range 1.20–2.08 —
tight for a one-parameter fit, and consistent with the linear law above). The needle
points give `c = 0.54 ± 0.30`, all *under*-stepping and scattered — the expected signature
of "accidentally right at low depth." Hence `r*(t) = round(1.8 · t)`.

**Evidence 4 — the cap, which is the more important half of this entry.** Smallest `r`
whose noiseless error stays below `0.15` **for all larger `r`** (sustained convergence,
immune to crossings by construction):

| t | 0.5 | 2 | 3.5 | 5 | 6.5 | 8 | 9.5 | 11 | 12.5 | 14 | 15.5 | 17 | 18.5 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r | 1 | 10 | 6 | 6 | 10 | 18 | 12 | 22 | 35 | 22 | 40 | — | 30 | — |

Past `t ≈ 12` the circuit is not Trotter-converged anywhere in the tested range, and at
`t = 17` and `t = 20` it never converges by `r = 40`. So the honest schedule says *you
need more depth than the noise budget allows* beyond `t ≈ 12` — D-1/D-2's gate-count
argument and D-15's depth argument arriving for a third time, now as a hard limit on the
usable time window rather than a tuning problem. Reporting the fitted needle at `t=20` as
`error 0.021` was the failure mode this cap exists to prevent.

**What this means for the D-16 "hard window at `t ≈ 12.5–15`".** Not a special dynamical
feature after all, or at least not established as one. It is where the required step count
first exceeds what noise permits — the leading edge of the cap, not an anomaly inside an
otherwise-good region. The `t ≈ 12.5–15` points are simply the last ones where a
best-available-`r` was still reported honestly (large `E*`, 0.37–0.59) before the needles
at `t ≥ 17` started reporting small ones dishonestly.

**Status: decided, not yet implemented.** `neuron_circuit.recommended_trotter_steps` still
returns D-16's interpolated schedule and `NuronSim.py` still sweeps to `Time = 20`.

**What is *not* claimed.** That `c = 1.8` is exact — it is an 8-point fit to one config
against one `willow_pink` calibration snapshot, and D-16's config-scope caveat and
`check_trotter_schedule_config` warning carry over unchanged. That `0.15` is the right
convergence tolerance, or `12` the exact cap — both are read off a 14-point `t` grid and a
17-point `r` grid, and a finer grid would move them. That the needles are *only* an
artifact of accidental crossings: the underlying oscillatory-aliasing mechanism D-16
identified is real and is the cause; what changed is the conclusion that argmin-ing
through it cannot be repaired by smoothing in `t`.

**Overturned by:** a state-fidelity metric replacing the 2-observable error metric. The
whole failure mode here exists because two real numbers are being matched in a 10-dim
physical Hilbert space, leaving ample room for accidental agreement. Computing
`|<ψ_qutip(t)|ψ_trotter(t)>|²` in the noiseless case cannot be fooled this way and would
make the needle/basin distinction unnecessary rather than merely detectable — that is the
better fix, and this entry should be revisited if it is done.

---

## D-18 — Determine the Trotter schedule from noiseless state fidelity, not observable error

**Decided (2026-08-11).** Drew's call, agreed. The metric used to choose
`NumberOfTrotterSteps` becomes the **noiseless state fidelity**

```
F(t, r) = |<ψ_qutip(t) | ψ_trotter(t, r)>|²
```

replacing the two-observable error metric that D-16 and D-17 were both built on:

```
error(t, r) = sqrt( ((occ - occ_qutip)/(N-1))² + ((mag - mag_qutip)/2)² )      [retired]
```

D-17's *conclusions* (monotone schedule, cap the sweep) are expected to survive and are
not withdrawn. D-17's *numbers* — `c = 1.76 ± 0.27`, the `0.15` convergence tolerance,
the `t ≈ 12` cap, the basin/needle classification — are all metric-dependent and must be
recomputed. Treat them as provisional until the fidelity re-run replaces them.

**Why.** D-17 established that the observable metric can be fooled: it matches two real
numbers in a 10-dimensional physical Hilbert space (`N=5` Fock × 2 spin), so a badly wrong
state can score well by accident. Six of fourteen argmins were such accidents, and at
seven of fourteen points *adding noise improved the score* — the clearest possible sign the
metric was not measuring accuracy. Fidelity compares the full state vector, so there is no
low-dimensional projection for a wrong state to hide behind. This removes the failure mode
rather than detecting it, which is why it supersedes D-17's needle/basin diagnostic instead
of complementing it.

**Critical structural point — fidelity alone is degenerate, and the schedule must not be
an argmax over it.** Noiseless Trotter fidelity increases monotonically with `r`
(asymptotically; `F → 1` as `r → ∞` by construction). So `argmax_r F(t, r)` always returns
the largest `r` tested and carries no information. The metric change therefore does *not*
slot in as a drop-in replacement inside the old argmin loop — the schedule must be built
in the two-stage form D-17 already adopted, with fidelity supplying only the first stage:

1. **Trotter requirement, from fidelity.** `r_T(t)` = smallest `r` such that
   `1 - F(t, r') < tol` for **all** `r' ≥ r` (sustained convergence, not first crossing —
   the "for all larger `r`" quantifier is what makes it immune to accidental single-point
   dips, and it is retained from D-17 for exactly that reason).
2. **Noise budget, separately.** `r_max` from accumulated two-qubit gate error at the
   compiled gate count per step — a property of the circuit and the calibration, and
   independent of `t` (D-15's mechanism).
3. **Schedule** = monotone fit to `r_T(t)`; **cap** the sweep at the largest `t` where
   `r_T(t) ≤ r_max`.

Stage 2 is where noise enters, and it enters as a constraint rather than as a term in the
objective. This is deliberate: mixing noise into the objective is what allowed noise to
cancel Trotter error and produce the D-17 needles.

**Cheap correctness check this makes available, worth wiring in at the same time.** Every
term in the Hamiltonian conserves excitation number within each boson column — the unary
`H_boson` terms are `XX+YY`, `H_spin` is `XX+YY` on the spin qubits, and `H_CNOT` is
`|1><1| ⊗ X` acting on a different qubit than the column it reads. So each Trotter factor
conserves it exactly, and **the noiseless Trotterized state must remain exactly in the
unary sector for all `r`**. Computing `F` requires projecting the `2^6`-dim qubit state
onto the 10-dim physical space anyway; the norm of the discarded component should be `0`
to numerical precision. Any nonzero value in the *noiseless* run is a compilation or
qubit-ordering bug, not physics. Assert it rather than assume it (project convention).

**Implementation notes for the re-run.** `|·|²` makes global phase irrelevant, so no phase
convention is needed. Qubit ordering must match `willow_qubit_chain`'s convention (boson
qubits per mode, then spin qubits — same ordering
`compute_observables_from_z_expectations` relies on); getting this wrong produces a
plausible-looking but wrong fidelity, so the unary-sector check above should be run first
as an ordering smoke test. Requires a statevector, so this is the `UseNoiseModel=False` /
`simulate` path — cheaper than the noisy path, no sampling, and D-15's `UseNoiseModel`
flag already exposes it.

**Status: decided, not yet implemented.** `src/optimal_trotter_steps.py` still computes the
retired observable metric; `neuron_circuit.recommended_trotter_steps` still returns D-16's
interpolated schedule; `NuronSim.py` still sweeps to `Time = 20` uncapped.
`results/optimal_trotter_steps_grid.npz` holds observable-metric data only and does not
contain fidelities — the re-run is a fresh computation, not a re-analysis of the existing
grid (unlike D-17, which needed no new circuit evaluations).

**What is *not* claimed.** That fidelity is the right *final* figure of merit for the
project — it is not, and cannot be measured on hardware; the deliverable observables remain
the boson occupation and spin magnetisation (see "Target observable" in `PROJECT.md`).
Fidelity is being used as a *diagnostic to choose `r`*, where it is available for free in
simulation and where the observable metric is demonstrably unreliable. That distinction
should survive into the writeup: the schedule is chosen by fidelity, the results are
reported in observables.

**Overturned by:** finding that the fidelity-derived schedule and the observable-derived
one agree closely at the basin points — which would mean the observable metric was adequate
and the extra machinery is unnecessary. Worth checking explicitly on the re-run, since the
8 basin points give a direct comparison for free.

---

## D-19 — D-18 implemented: fidelity-derived schedule (`c=4.0`, capped at `t≈8.25`) and shot-level post-selection

**Decided (2026-08-11).** Implements D-18 end to end. `src/trotter_fidelity_schedule.py`
computes `F(t,r) = |<psi_qutip(t)|psi_trotter(t,r)>|^2` on the logical (uncompiled)
circuit (see `neuron_circuit.trotter_final_statevector`'s docstring for why this is
equivalent to the Willow-compiled path and avoids the qubit-ordering pitfall D-18 flagged),
projects the Trotter statevector onto the unary subspace
(`neuron_circuit.unary_subspace_projection`), and asserts leaked probability is `~0`
(max `1.8e-7` over the whole grid — the free correctness check D-18 specified).

**Stage 1 result — trotter requirement `r_T(t)`.** Grid: `t ∈ [0.5, 20]` (14 points),
`r ∈ {1..60}` (20 points). Sustained-convergence threshold `1 - F < 0.05` for all larger
`r`. 10/14 points converge within the tested range; `r_T` grows with `t` roughly linearly
but with real scatter (`r_T/t` ranges 2.0–4.3). The scatter is worse at small `t`
(quantization: `steps_grid`'s spacing is coarse relative to small `r_T`, so a ±1-grid-point
rounding error swings the ratio a lot) — confirmed by comparing the whole-range fit
(median `c=3.84`) against a tail-only fit restricted to `t≥8` (median `c=4.00`, tighter,
less quantization-sensitive). **Adopted `c=4.0`** — the tail fit, not the whole-range one.

**Stage 2 result — noise budget `r_max`.** Compiled circuit has 10 two-qubit gates per
Trotter step on the chosen `willow_pink` chain; mean two-qubit XEB Pauli error on that
chain's edges is `0.2064%`. A naive `(1-eps)^(gates*r) >= 0.5` budget gives `r_max = 33`.

**Schedule: `r*(t) = round(4.0 * t)`, capped at `t ≈ 8.25`** (`r_max/c`). This is
noticeably *tighter* than D-17's own cap estimate (`t≈12`, on the retired observable
metric) — expected, not a regression: fidelity is a strictly harder bar to clear (full
10-dim state overlap vs. two projected numbers), so it correctly demands more Trotter
depth for the same `t`, which brings the noise-budget crossover forward. Comparing
`r_T(t)` against D-17's basin-point cap table directly (same `t` values) confirms this:
`r_T` is at or above D-17's cap at every comparable point (e.g. `t=6.5`: D-17 cap `10`,
this schedule's `r_T = 26`). D-18's own "overturned by" clause — checking whether the two
metrics agree at the basin points — is therefore answered **no, they don't agree, and the
disagreement is in the direction fidelity's stricter bar predicts.** The observable metric
was not merely occasionally fooled (D-17's needles); it was systematically permissive.

**Wired into `NuronSim.py`:** `UseAdaptiveTrotterSteps=True` now caps `Time` to the
schedule's `t_cap` once, with one printed warning, rather than sweeping past it and
reporting fitted-but-meaningless points (D-17's original complaint). Verified end-to-end:
noiseless Cirq points track the qutip curve closely out to the cap; the sweep stops there
by construction (`results/NuronSim_noiseless.png`, `results/NuronSim_noisy.png`).

**Post-selection (D-4), implemented alongside.** `neuron_circuit.sample_shots_with_postselection`
switches the noisy path from `sample_expectation_values` (marginals only, D-14) to
`.run()` with measurement gates on every qubit, keeping the joint per-shot bitstrings.
Each mode's boson register is checked for being exactly one-hot; shots that aren't are
discarded before averaging (the detectable-leakage argument from D-4, now actually
implemented rather than just argued for). Survival rate is tracked and printed per run —
first full sweep gave mean `53.2%` (range `29.5%–94.7%`, falling as step count/circuit
depth grows with `t`, consistent with accumulated two-qubit gate error). Both post-selected
(`*_post`, primary) and uncorrected marginal (`*_raw`, diagnostic) estimates are saved.

**Scope caveat, unchanged in kind from D-16/D-17/D-18:** fitted to one config
(`NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=[1.0], spin_interaction_coefficient=0.5`,
`G≈2.03`) against one `willow_pink` calibration snapshot and one `min_fidelity_budget=0.5`
choice for `r_max` (a round, defensible, but not derived number — see
`estimate_max_affordable_trotter_steps`'s docstring). Re-run
`src/trotter_fidelity_schedule.py` if any of these change;
`neuron_circuit.check_trotter_schedule_config` still only warns, doesn't block.

---

## D-20 — Second-order (Strang) Trotterization added; caught and fixed a real convergence-order bug

**Decided (2026-08-11).** `src/NeuronSim2ndOrderTrotter.py` (duplicate of `NuronSim.py`)
and `neuron_circuit.build_second_order_trotter_circuit` implement symmetric/Strang
splitting — `A(dt/2) B(dt/2) C(dt) B(dt/2) A(dt/2)` per step instead of first order's
`A(dt) B(dt) C(dt)` — with explicit boundary-layer merging across repeated steps
(`e^{A dt/2} e^{A dt/2} = e^{A dt}`) so the extra accuracy doesn't cost a naive ~1.7x
(5 layers/step vs 3) gate overhead. Full derivation and layer-count accounting:
`notes/second-order-trotter.md`.

**Bug found and fixed during implementation, worth logging on its own.** The first
version treated `BosonicDisplacementGate` — itself an even-bond/odd-bond first-order
Trotter split of `H_boson` — as one atomic operator in the outer symmetric formula. This
compiles, runs, and produces plausible-looking (better than first order at most sampled
points) numbers, but is **not actually second order**: an `O(dt^2)`-accurate inner
building block caps the whole circuit's convergence at first order regardless of outer
symmetrization. Caught by extending `consistency_checks.py` with a log-log
convergence-order fit (`1-F` vs `r`) rather than trusting the construction by inspection —
before the fix, both circuits' fitted slopes were `≈-2` (first-order scaling); after
splitting the boson term's even and odd bonds into separate atomic terms in the outer
composition (`_boson_even_odd_layers`), the second-order slope moved to `≈-4.3`, matching
the `O(r^-4)` prediction for infidelity under a genuinely second-order method. This is
exactly the project's "verify numerically before asserting" convention catching something
inspection would not have: the circuit *looked* like a correct Strang split.

**Verified (`src/consistency_checks.py`, all 15 checks pass):** noiseless unary-subspace
leakage `~0` for both circuit orders (same D-18 free correctness check); convergence
order `-1.93` (1st) vs `-4.28` (2nd) at `t=3`; gate-count overhead of 2nd order vs 1st at
matched `r` is `~1.51x` — matching the derivation (the boson chain's odd-bond layer is
applied twice per step and can never merge across a boundary, since it never sits at one;
only the even-bond layer, the true outermost term, merges) — not the `~1x` an
(incorrectly) fully-atomic boson-layer treatment would predict, and well under a naive
unmerged composition's `~2x`.

**Not second-order-specific:** `NeuronSim2ndOrderTrotter.py` deliberately reuses
`recommended_trotter_steps(t)` (D-18/D-19, fitted to *first*-order error) rather than
deriving its own schedule, so the two scripts are directly comparable at matched step
counts. A schedule exploiting second order's faster convergence to use fewer steps for
the same accuracy is flagged as a natural follow-up, not pursued (see D-16 through D-18's
pattern of one-schedule-per-config — this would be another one).

**Results and comparison (`notes/first-vs-second-order-trotter-comparison.md`):
noise sign-flips the answer.** At matched step count (both scripts share D-18/D-19's
schedule, fitted for first order): **noiseless**, second order wins clearly — RMS error
vs. qutip drops `1.7x` (occupation) to `15.7x` (spin magnetisation), consistent with the
`O(r^-4)` vs `O(r^-2)` convergence-order difference confirmed above. **Noisy,
post-selected** (the hardware-realistic case), second order is *worse* (occupation RMS
0.554 vs first order's 0.382) and its post-selection survival rate is lower (45.3% vs
53.2% mean) — because it costs `~1.51x` the two-qubit gates per step (D-20 above), and
D-19's schedule was already tuned close to the noise budget edge for *first*-order error,
so the extra per-step gate cost pushes an already-tight operating point further into the
noise-dominated regime (D-15's mechanism, now demonstrated across circuit variants, not
just across step counts within one). **Not a verdict against second order** — it is a
mismatched-schedule artefact: the fair comparison is fewer steps at higher per-step
accuracy, not the same steps at higher per-step cost. Deriving a second-order-specific
D-18-style schedule (predicted to allow a *later* cap than `t≈8.25`, not an earlier one,
given the shallower `r*(t)` scaling `O(r^-4)` implies) is flagged as the natural next
step and not pursued here.

---

## D-21 — Linear Trotter schedule confirmed over quadratic on the real (noisy,
post-selected) metric; `TROTTER_SCHEDULE_C` revised `4.0` → `2.25`

**Decided (2026-08-12).** Supersedes D-19's schedule constant (not D-18's method of
using fidelity as a convergence *diagnostic*, which stands). `TROTTER_SCHEDULE_C` in
`neuron_circuit.py` changes from `4.0` to `2.25`; `TROTTER_SCHEDULE_R_MAX` stays `33`
(re-verified numerically, unchanged — it's a property of the compiled circuit and
`willow_pink`'s calibration, not of how `c` is fit); `TROTTER_SCHEDULE_T_CAP` moves from
`8.25` to `14.67` (`= r_max / c`, algebraic consequence — see caveat below).

**Motivating question.** Standard first-order Trotter error theory (Childs, Su, Tran,
Wiebe, Zhu, *A Theory of Trotter Error*, PRX 11 011020 — commutator-scaling bound, Eq. 2
with `p=1`) gives a single formula application over duration `t` error `O(t^2)`; chopping
total time `T` into `r` steps and summing via the triangle inequality gives total error
`~O(T^2/r)`. Holding a **fixed** error tolerance as `T` grows therefore requires `r ~ T^2`,
not `r ~ T`. Yet D-17's own trade-off derivation used exactly this `T^2/r` form and
produced a **linear** schedule (`r* = t*sqrt(A/B)`), and D-18/D-19 also adopted a linear
form (`c=4.0`) — fit to `r_T(t)`, defined as a *fixed-tolerance* threshold on noiseless
fidelity, which by the same theory should have come out quadratic. Raised in
conversation 2026-08-12 (not itself checked against the paper before D-17); this entry
resolves it empirically rather than re-deriving further.

**Resolution — these are two different questions, and theory predicts different scaling
for each.** `r ~ t^2` is the answer to "what `r` holds *Trotter-only* error to a fixed
tolerance" (D-18/D-19's `r_T(t)`, in principle). `r ~ t` is the answer to "what `r`
minimizes *total* error when every extra step also costs noise" (D-17's actual
optimization, `ε_T + ε_N = At^2/r + Br`) — the operationally relevant question here, since
every Trotter step is a real gate on real (simulated) hardware. D-18/D-19's mistake was
not the switch to fidelity as a *diagnostic* — it remains a strictly better way to detect
Trotter-only convergence than D-16/D-17's two-observable metric, immune to the "needle"
contamination D-17 found (isolated lucky argmin points, 6 of 14). The mistake was using a
**noiseless** metric to set a schedule meant to also account for noise: fidelity cannot
see the noise cost of extra steps, so `r_T(t)` will always ask for more depth than is
actually optimal once post-selection survival is factored in.

**Method — measure both functional forms directly against the metric that matters.**
Rather than re-deriving `r_T(t)` under a different metric, this entry scans `r=a*t` and
`r=b*t^2` directly against noisy, post-selected Cirq observable error vs. qutip (D-16/
D-17's normalized combined-error metric — boson occupation error `/(N-1)`, spin
magnetization error `/2`, combined in quadrature — RMS'd over a time sweep), which is
the metric NuronSim.py's actual output is judged on. Two scripts, same model config as
NuronSim.py's defaults (`L=1, N=5, D=1.0, J=0.5`, matching `_SCHEDULE_CONFIG`):

1. `src/adaptive_trotter_model_scan.py` — coarse scan, `t ∈ [0.5,10]` (20 points, 800
   shots/point), noise ON + post-selection ON throughout. `a ∈ {0.5,1,2,3,4,6,8,10}`,
   `b ∈ {0.02,0.05,0.1,0.2,0.3,0.4,0.6,0.8}` (b-range scaled down from a-range so both
   sweep a comparable step-count range at `t=10`). Outputs in `results/linear trotter/`
   and `results/quadratic trotter/` (one plot+`.npz` per coefficient, plus a per-model
   summary), and `results/linear_vs_quadratic_trotter_comparison.png`.
2. `src/linear_trotter_finegrid_scan.py` — fine scan around the coarse winner,
   `a ∈ {1.5,1.75,2.0,2.25,2.5,2.75,3.0}`, doubled resolution (`t ∈ [0.5,10]`, 40 points,
   1200 shots/point). Output: `results/linear trotter/linear_finegrid_a*.png/.npz` +
   `summary_finegrid.png/.npz`.

**Result 1 — linear beats quadratic outright, not just at one coefficient.** Coarse scan:
best linear `a=2` gives RMS `0.157`; best quadratic `b=0.6` gives RMS `0.292` — linear
wins by ~2x, and every tested linear coefficient in `{1,2,3,4}` beats every tested
quadratic coefficient in `{0.3,0.4,0.6}` (the respective near-optimal neighbourhoods).
Mechanism, visible in the comparison plot: quadratic tracks qutip about as well as linear
out to `t≈7`, then its required step count (60 at `t=10`, hitting `R_CAP`) crushes
post-selection survival and error spikes to `~0.6`; linear needs only 20 steps at `t=10`
and stays flat around `0.2`. This is D-15's noise-vs-depth mechanism again, now showing
up as a difference between *functional forms* rather than between fixed step counts.

**Result 2 — fine grid gives a clean, non-lucky optimum.** `a`: `1.5→0.222, 1.75→0.174,
2.0→0.162, 2.25→0.157 (min), 2.5→0.179, 2.75→0.189, 3.0→0.202`. A parabola fit through
all 7 points gives a minimum at `a≈2.26` — matching the grid optimum to within `0.01`,
and the measured curve is visibly a clean unimodal parabola (`summary_finegrid.png`), not
jagged the way D-16's raw per-point argmin was. Adopted `TROTTER_SCHEDULE_C = 2.25` (the
grid-verified value).

**Cross-check against prior work.** `a≈2.25` sits close to D-17's own noisy-metric fit
(`c=1.76±0.27`, range `1.20–2.08`) — the same regime, not a contradiction — and is about
half of D-18/D-19's noiseless-fidelity `c=4.0`. D-17's needles were noted there as "all
*under*-stepping" relative to the true trend; a slightly higher, needle-free estimate here
(2.25 vs 1.76) is consistent with that, not surprising.

**What this does not overturn.** D-18's core methodological point stands: a metric that
can be accidentally fooled (D-16/D-17's raw two-observable argmin) is a bad way to *detect
Trotter convergence*, and fidelity fixed that. What was wrong was conflating "detect
Trotter convergence" with "choose the noise-aware schedule" — those need different
metrics, and D-18/D-19 used the convergence-detection one for both jobs. D-21's direct
scan sidesteps the two-stage decomposition entirely (no separate `r_T(t)` / `r_max`
stages) by optimizing the real metric end-to-end — the noise constraint falls out of the
objective automatically (error gets worse past the optimum) rather than needing to be
bolted on as a separate cap, though `r_max=33` remains a useful, independently-derived
sanity bound (re-confirmed unchanged above).

**Caveat — `T_CAP=14.67` is algebra, not itself validated.** The D-21 scan only tested
`t ∈ [0.5, 10]`. `T_CAP = r_max/c` is a correct statement about where the schedule
`r*(t)=round(2.25t)` would first exceed the `r_max=33` noise budget if extrapolated, but
whether `a=2.25` *remains* the error-minimizing coefficient all the way out to `t≈14.67`
(rather than needing to grow, if quadratic-like effects start to matter at larger `t`) is
unverified — re-run the scan with `t` extended past 10 before leaning on the extended cap
for anything load-bearing.

**Scope caveat, unchanged in kind from D-16 through D-19:** fitted to one config
(`NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=[1.0], spin_interaction_coefficient=0.5`,
`G≈2.03`) against one `willow_pink` calibration snapshot. Re-run
`src/adaptive_trotter_model_scan.py` (+ a finer follow-up grid if needed) if any of these
change; `check_trotter_schedule_config` still only warns, doesn't block.

**Overturned by:** a re-run at extended `t` showing `a=2.25` stops being optimal past
`t≈10`, or a config/calibration change per the scope caveat above.

---

## D-22 — Second-order gets its own fitted linear schedule (`c=2.0`); still loses to
first order even on a fair, order-specific schedule

**Decided (2026-08-12).** Same day as D-21, same method, applied to the second-order
(Strang) circuit. `NeuronSim2ndOrderTrotter.py` previously reused
`recommended_trotter_steps(t)` — the schedule fitted to *first*-order error (D-18/D-19,
then D-21) — deliberately, for an apples-to-apples matched-step comparison (D-20). That
comparison's own writeup flagged this as leaving second order's faster convergence
unexploited: Strang splitting's operator-norm error shrinks as `~r^-2` (vs first order's
`~r^-1`), so a schedule derived for first order should, in principle, over-step second
order. This entry derives second order's own schedule and re-runs the comparison to check
whether that headroom is real.

**Method — identical to D-21, on `build_second_order_trotter_circuit`.**
`src/second_order_trotter_linear_scan.py`: coarse scan over `a ∈
{0.25,0.5,0.75,1,1.5,2,3,4,6,8}` (`t∈[0.5,10]`, 20 points, 800 shots/point, noise ON +
post-selection ON), then a fine scan around the coarse winner (`a ∈
{1.25,...,2.75}` step 0.25, 40 points, 1200 shots/point) — same model config, same
combined-error metric, same `t`-range as D-21, so directly comparable. Head-to-head
comparison reuses D-21's saved fine-grid `a=2.25` result rather than re-running first
order (same config, same resolution, no reason to re-sample).

**Result 1 — the optimal coefficient is close to first order's, not smaller.** Coarse
winner `a=2` (RMS `0.167`); fine grid gives a clean unimodal minimum, best grid point
`a=2` (RMS `0.1657`), parabolic fit `a≈2.15`
(`results/2nd order linear trotter/summary_finegrid.png`). This is close to — not
markedly below — first order's `a=2.25`. The naive expectation from `~r^-2` vs `~r^-1`
convergence was that second order should need noticeably fewer steps; that expectation
is largely cancelled by its higher per-step gate cost: one (unmerged) Strang step compiles
to 16 two-qubit gates vs first order's 10 (`~1.6x`, consistent with D-20's `~1.51x`
amortized-large-`r` estimate). Re-deriving the trade-off (`ε_T ~ A t^3/r^2`, `ε_N ~ B' r`
with `B'` scaled up `~1.6x`) still predicts a *linear* schedule (`r*=t·(2A/B')^(1/3)`),
just with a smaller constant than a naive `r^-2`-convergence-only argument would suggest —
consistent with what was measured.

**Result 2 — first order still wins, but narrowly and not everywhere.**
`results/first_vs_second_order_optimized_trotter_comparison.png`: at each order's own
optimum, first order (`a=2.25`) gets RMS `0.1565`; second order (`a=2.0`, mean steps
`10.5` vs first order's `11.8`) gets RMS `0.1657` — about `6%` higher, not a blowout.
The per-`t` error trace shows this isn't uniform: second order tracks qutip *better* than
first order for `t≲7` (first order has a pronounced local error bump around `t≈2` that
second order doesn't share), then *worse* for `t≳7.5`, where second order's error spikes
while first order's stays flatter. The RMS verdict is decided by that late-time region.

**Conclusion.** D-20's original finding — second order loses once real noise and
post-selection are in the picture, despite winning noiselessly — **survives** even after
removing the schedule-mismatch confound D-20 itself flagged as a possible artefact. Both
scripts are now on their own fair footing (own coefficient, own noise-budget cap), and
first order still comes out ahead overall, though closely enough, and unevenly enough
across `t`, that "second order is worse" is not the full story — it is a small
net-negative average over a trade that runs the other way part of the time.

**Own noise budget, not reused.** `TROTTER_SCHEDULE_R_MAX_2ND_ORDER=20` (vs first order's
`33`), recomputed via `two_qubit_gates_per_trotter_step` against a single (`r=1`,
unmerged-boundary) Strang step rather than reusing first order's `r_max` — a circuit
property, so it has to be its own number given the ~1.6x gate-count difference.
`TROTTER_SCHEDULE_T_CAP_2ND_ORDER = r_max/c = 10.0`. Unlike D-21's `t_cap=14.67` (which
extrapolates past the tested `t≤10`), this cap lands almost exactly at the edge of what
D-22 actually scanned — a coincidence, not designed, but it means this cap is not an
extrapolation the way D-21's is.

**Code changes:** `neuron_circuit.py` gains `TROTTER_SCHEDULE_C_2ND_ORDER`,
`TROTTER_SCHEDULE_R_MAX_2ND_ORDER`, `TROTTER_SCHEDULE_T_CAP_2ND_ORDER`,
`recommended_trotter_steps_2nd_order`, `trotter_schedule_cap_message_2nd_order` — parallel
to, not replacing, the first-order versions. `NeuronSim2ndOrderTrotter.py` now calls
these instead of the first-order schedule. Verified end-to-end: default run (`Time=20`)
correctly warns and caps to `t=10.00`, produces a sane plot tracking qutip,
`results/NeuronSim2ndOrderTrotter_noisy.png`.

**Scope caveat, unchanged in kind from D-16 through D-21:** fitted to one config
(`NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=[1.0], spin_interaction_coefficient=0.5`,
`G≈2.03`) against one `willow_pink` calibration snapshot. Re-run
`src/second_order_trotter_linear_scan.py` if any of these change.

**Overturned by:** a config change per the scope caveat, or evidence the late-`t` region
driving second order's RMS deficit is itself a measurement artefact (e.g. shot-noise
outliers) rather than real — worth a higher-shot re-check at `t∈[7,10]` specifically if
this comparison becomes load-bearing for a paper claim.

**Results and comparison (`notes/first-vs-second-order-trotter-comparison.md`):
noise sign-flips the answer.** At matched step count (both scripts share D-18/D-19's
schedule, fitted for first order): **noiseless**, second order wins clearly — RMS error
vs. qutip drops `1.7x` (occupation) to `15.7x` (spin magnetisation), consistent with the
`O(r^-4)` vs `O(r^-2)` convergence-order difference confirmed above. **Noisy,
post-selected** (the hardware-realistic case), second order is *worse* (occupation RMS
0.554 vs first order's 0.382) and its post-selection survival rate is lower (45.3% vs
53.2% mean) — because it costs `~1.51x` the two-qubit gates per step (D-20 above), and
D-19's schedule was already tuned close to the noise budget edge for *first*-order error,
so the extra per-step gate cost pushes an already-tight operating point further into the
noise-dominated regime (D-15's mechanism, now demonstrated across circuit variants, not
just across step counts within one). **Not a verdict against second order** — it is a
mismatched-schedule artefact: the fair comparison is fewer steps at higher per-step
accuracy, not the same steps at higher per-step cost. Deriving a second-order-specific
D-18-style schedule (predicted to allow a *later* cap than `t≈8.25`, not an earlier one,
given the shallower `r*(t)` scaling `O(r^-4)` implies) is flagged as the natural next
step and not pursued here.
