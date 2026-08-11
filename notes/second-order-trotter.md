# Second-order (Strang) Trotterization

Status: **implemented and verified**, 2026-08-11. `src/NeuronSim2ndOrderTrotter.py`,
`build_second_order_trotter_circuit` in `src/neuron_circuit.py`.

---

## What this is

A second-order (Strang/symmetric) product-formula alternative to the first-order
Trotterization `NuronSim.py` has always used. Same Hamiltonian, same qubit mapping, same
Willow compilation, same noise model, same post-selection — the only thing that changes
is how one `dt`-step's unitary is assembled from the three non-commuting Hamiltonian
pieces (`H_boson`, `H_spin`, `H_CNOT`).

First order (existing, `NetworkOfNeuronsTrotterStep`):

```
U1(dt) = A(dt) B(dt) C(dt)          A = H_boson, B = H_spin, C = H_CNOT
```

Local error per step is `O(dt^2)`; global error after `r = t/dt` steps is `O(dt) = O(t/r)`.

Second order (new, `build_second_order_trotter_circuit`):

```
U2(dt) = A(dt/2) B(dt/2) C(dt) B(dt/2) A(dt/2)
```

Local error per step is `O(dt^3)`; global error is `O(dt^2) = O(t^2/r^2)` — the same
total time `t` needs far fewer steps `r` for a given error, or the same `r` gives much
better accuracy. Standard leapfrog/Strang splitting, nothing novel about the formula
itself; what took the actual work was implementing it *correctly and efficiently* for
this specific circuit, which had one non-obvious subtlety (see below).

## The layer-merging optimization (what Drew asked for specifically)

Naively concatenating `r` independent copies of `U2(dt)` costs 2 `A`-layers and 2
`B`-layers per step (5 layers/step vs first order's 3) — roughly 60% more gates for the
boson term alone. But `e^{A dt/2} e^{A dt/2} = e^{A dt}`, so **the trailing half-strength
layer of one step and the leading half-strength layer of the next step are the same
operator and combine into one full-strength layer**. Only the outermost term in the
symmetric ordering sits at a step boundary, so only it merges:

```
A(dt/2) [B(dt/2) C(dt) B(dt/2) A(dt)]^(r-1) B(dt/2) C(dt) B(dt/2) A(dt/2)
```

Layer counts: `A: r+1` (down from `2r`), `B: 2r` (doesn't merge — see below), `C: r`
(same as first order, it's always full-strength). This is implemented explicitly in
`build_second_order_trotter_circuit` (the `even_full if step < r - 1 else even_half`
line) rather than left for the Cirq compiler to discover — Cirq's `optimize_for_target_gateset`
does local gate fusion but has no reason to know two ISwapPowGate applications separated
by other circuit structure are the same accumulated rotation; explicit construction
guarantees the merge happens.

## The bug this caught, and why it mattered

The first attempt treated `H_boson`'s `BosonicDisplacementGate` — which is *itself* a
first-order Trotter split of the boson chain's nearest-neighbour couplings into
even-bond and odd-bond layers (`even(τ) then odd(τ)`, `_decompose_`) — as one atomic
"A" operator in the outer `U2` formula above. That's wrong: even and odd bonds don't
commute, so `BosonicDisplacementGate(τ)` is only an `O(τ^2)`-accurate approximation of
`exp(-i H_boson τ)`, the same order as a first-order step. Embedding an `O(τ^2))`-accurate
building block into the "exact" slot of a symmetric outer composition caps the *whole*
circuit at first-order convergence, no matter how carefully the outer `A/B/C` ordering is
symmetrized — the inner approximation is the bottleneck.

This was not caught by inspection — it was caught by `consistency_checks.py`'s
convergence-order test, which fits `log(1-F)` vs `log(r)` and checks the slope. Before the
fix: 1st order slope ≈ `-1.9` (expected `-2`, correct), 2nd order slope ≈ `-2.0` (expected
`-4` — **wrong**, indistinguishable from first order despite the extra gates and
complexity). After the fix, 2nd order slope ≈ `-4.3`. See "Verification" below for the
numbers; the point here is that a plausible-looking, bug-free-*running* implementation
was silently not doing what it claimed, and the only way to know was to check the
convergence rate directly rather than trust the construction by eye. This is the project
convention (`CLAUDE.md`: "verify numerically before asserting") working as intended.

**The fix:** expose the boson term's even-bond and odd-bond halves as separate atomic
terms (`_boson_even_odd_layers`, alongside the existing `_boson_layer` which is now only
used by the first-order path) and symmetrize *them* individually in the outer
composition, rather than treating "the boson layer" as a black box:

```
Even(dt/2) Odd(dt/2) B(dt/2) C(dt) B(dt/2) Odd(dt/2) Even(dt/2)
```

Only `Even` — now the true outermost term — merges across step boundaries; `Odd` sits
between two `B`/`C`-adjacent slots on both sides of every step and never touches a step
boundary, so it stays at `2r` applications regardless (`Odd` cannot benefit from the same
trick `Even` does). Net boson-gate overhead vs. first order is therefore ~1.5x
(confirmed empirically below), not the ~1x an (incorrect) fully-atomic-A treatment
suggested, and well under the ~2x a fully unmerged construction would cost.

At `NumberOfBosonicModes == 1` (the config both scripts actually run), `B` (`H_spin`) is
identically empty — there's no second site to hop to, so `_spin_layer`'s decomposition
yields nothing — and the construction above simplifies toward the classic
"kick-drift-kick" leapfrog form built from `Even`, `Odd`, and `C`.

## Verification (`src/consistency_checks.py`)

All 15 checks pass, including (added specifically for the 2nd-order work):

- **Unary-subspace conservation, noiseless**: both circuits leak `< 3e-7` probability out
  of the physical subspace at every tested `(t, r)` — exact conservation to floating-point
  precision, the D-18 "free correctness check" (every Hamiltonian term conserves the
  unary constraint, so a noiseless Trotter circuit must too, regardless of order).
- **Convergence order**: log-log fit of `1-F(t,r)` vs `r` at `t=3.0`, `r ∈ {16,32,64,128}`
  gives slope `-1.93` (1st order, matches the `O(r^-2)` prediction) vs `-4.28` (2nd order,
  matches the `O(r^-4)` prediction). This is the check that caught the bug above and the
  one that actually certifies "this is a second-order method," as opposed to "this circuit
  merely does better at one arbitrarily chosen `(t,r)`."
- **Gate-count structure**: at `r=12`, `gates_2nd / gates_1st ≈ 1.51`, matching the `Odd`
  doubling argument above (not ≈1, not ≈2).
- **Accuracy at matched step count**: at `t=3.0, r=128`, `1-F` is `3.1e-4` (1st order) vs
  `1.2e-7` (2nd order) — roughly 2500x more accurate for ~1.5x the gates, once `r` is
  large enough that the asymptotic scaling has kicked in (see the comparison notes for
  what happens at small `r`, where this is not guaranteed pointwise).

## What wasn't done (scope, matching `CLAUDE.md`'s "do not pursue without asking")

- **No second-order-specific step schedule.** `NeuronSim2ndOrderTrotter.py` reuses
  `recommended_trotter_steps(t)`, which was fitted to *first*-order Trotter error (D-18).
  This is deliberate — it keeps the two scripts' step counts identical at every `t`, so
  the comparison notes measure "same gate budget, which method is more accurate" rather
  than conflating two different schedules. A schedule that exploits second order's faster
  convergence to use substantially fewer steps for the same fidelity is a natural
  follow-up (worth roughly `sqrt` fewer steps for the same error, given the different
  power laws) but is a separate D-18-style fidelity-grid exercise, not done here.
- **`H_spin` was not exercised** (`NumberOfBosonicModes == 1` in both scripts' default
  config, same as `NuronSim.py`) — the general `L`-site construction is implemented and
  the layer counts above account for it, but only the `L=1` simplification has actually
  been run.
