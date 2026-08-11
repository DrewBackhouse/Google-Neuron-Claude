# Google Neuron — quantum simulation of a neuron network Hamiltonian

## Read these first, every session

1. `PROJECT.md` — outline, current status, open questions
2. `DECISIONS.md` — what has been decided and why (entries `D-1` … `D-16`)

**Update both before finishing.** They are the handover between sessions and between
tools (this project is worked on from both Claude Code and the Claude desktop app).

## What the project is

Simulate a spin–boson Hamiltonian, readable as a network of integrate-and-fire neurons,
on Google Willow (105 qubits). Chain of `L` sites; each site has a bosonic mode truncated
to `N` Fock states plus a spin.

```
H_boson = Σ_j D_j (a_j + a_j†)                        boson integrates upward
H_spin  =  J  Σ_j (σ+_j σ-_{j+1} + σ-_j σ+_{j+1})     signal hops along the chain
H_CNOT  =  G  Σ_j |N-1><N-1|_j X_j                    hitting the ceiling flips the spin
```

Unary encoding → an `L × (N+1)` qubit grid, every term nearest-neighbour.

**This is a demonstration and benchmark, not a quantum-advantage claim** (D-6). The state
space is `(2N)^L` — 1728 at `L=3, N=6`. Classical simulation is easy and is the point of
comparison, not the enemy.

## Environment

**On macOS (Claude Code), use a real venv — it persists, unlike the desktop app sandbox:**

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install qutip          # installs fine from an arm64 wheel here
```

`.venv/` is gitignored and must never be committed.

Note `requirements.txt` and `setup_env.sh` are pinned for the *desktop app's Linux
sandbox* (cp310, no qutip wheel available). On macOS the versions may float; do not
"fix" the pins to match macOS, since the sandbox still needs them.

**For anything running `src/NuronSim.py`'s Cirq/noise path, use the `GoogleQVM` conda env
instead** (`/opt/miniconda3/envs/GoogleQVM`, Python 3.12.13 — activate via
`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate GoogleQVM`, or invoke
`/opt/miniconda3/envs/GoogleQVM/bin/python3` directly since `conda activate` doesn't
reliably take effect in a non-interactive shell). It has a working `qsimcirq` install
that `.venv` doesn't (no wheel for macOS arm64 + Python 3.14, D-14) — `.venv` is still
right for qutip-only work (`Classical Simulation.py`).

## Conventions

- `src/` — code. `results/` — generated output, disposable, must be regenerable from
  `src/`. Never hand-edit anything in `results/`.
- `notes/` — analysis writeups, one topic per file.
- **Decisions are append-only.** If new evidence overturns an earlier decision, add a new
  entry that supersedes it and mark the old one — do not silently rewrite. D-5 → D-9 and
  D-10 → D-11 are worked examples of this.
- **Verify numerically before asserting.** Two claims in `DECISIONS.md` were wrong on first
  writing and caught by explicit checks. `src/symmetry_check.py` is the pattern: make the
  claim executable, run it, record the output.

## Gotchas that have already caught us

- **`N` is a physical parameter, not a convergence knob.** It sets the firing threshold.
  Encoding tricks that pad `d` to a power of two do not apply (D-4).
- **The Fock truncation is physical, not an approximation.** There is no phase-space
  rotation term; the mode oscillates by reflecting off the ceiling. Do not "improve" the
  model by raising `N` until it converges — that changes the physics.
- **Frame is the computational basis with a controlled-X interaction** (D-11). A ± -basis
  rotation was analysed and rejected: no resource benefit, since a global single-qubit
  basis change is a unitary equivalence and cannot alter entangling-gate count or Trotter
  error.
- **Uniform `D` gives no propagation.** All sites fire simultaneously and the XY hopping
  transports nothing. Symmetry must be broken via `D_j` (D-3).
- **The binding constraint is two-qubit gate count, not qubit count** (D-1, D-2). The
  spec sheet's "depth 40" is quoted at 0.1% XEB and is not a working budget.
- **Raising `NumberOfTrotterSteps` on a noisy sim doesn't just shrink Trotter error — it
  spends noise budget too, at every point in the sweep** (D-15). `NuronSim.py`'s time
  sweep varies total time `t` at a per-point step count, so step count sets circuit depth
  directly. Past some step count the accumulated gate noise overwhelms the signal (flat
  output, untethered from the qutip curve) well before Trotter error would have mattered —
  this is D-2's depth argument, not a bug. `NuronSim.py` now uses a per-point *adaptive*
  step count by default (`UseAdaptiveTrotterSteps`, D-16) derived from a grid search
  (`src/optimal_trotter_steps.py`) rather than one fixed value for the whole sweep — but
  that schedule is fit to one specific model config + Willow calibration snapshot, not a
  general law; re-run the analysis if either changes.

## Do not pursue without asking

- Further work on a full closed-form `G(D,N)` beyond `notes/G-analytic-estimate.md`. That
  note (investigated 2026-08-10 at Drew's request) found the truncated `(a+a†)` spectrum
  is exactly the Hermite-polynomial roots, giving a first-order area-theorem estimate that
  is a good *search seed* but not a replacement for the numerical fit — a real closed form
  would need the back-action correction at the operator level (Magnus expansion), which is
  parked again unless asked for.
