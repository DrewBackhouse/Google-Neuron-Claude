# Resource Estimate — first pass

Status: **back-of-envelope.** Superseded by proper analysis in steps 2 and 4.
Purpose is to fix the order of magnitude and rule out the full-grid configuration.

---

## Willow parameters used

From the Dec 2024 spec sheet. Chip 2 (random circuit sampling) has the better two-qubit
numbers, Chip 1 (error correction) the worse:

| Quantity | Chip 1 (QEC) | Chip 2 (RCS) |
|---|---|---|
| Two-qubit gate error | 0.33% (CZ) | 0.14% (iswap-like) |
| Single-qubit gate error | 0.035% | 0.036% |
| Measurement error | 0.77% | 0.67% |
| T1 | 68 µs | 98 µs |

Estimates below use the optimistic 0.14% and 0.67%. Halve the step counts for 0.33%.

Note the headline "103 qubits, depth 40" is quoted at **XEB fidelity 0.1%** — that is the
noise floor, not a working budget. See DECISIONS.md D-2.

## Counting

Qubits: `L(N+1)` — `N` per boson column (unary) plus one spin qubit per site.

Two-qubit gates per **first-order** Trotter step, assuming each `XX+YY` rotation and each
controlled-RX compiles to 2 native two-qubit gates:

```
H_boson :  L(N-1) XY rotations   → 2L(N-1)
H_spin  :  (L-1)  XY rotations   → 2(L-1)
H_CNOT  :  L      cRX            → 2L
                                  ─────────────
                          total  =  2LN + 2L - 2
```

Second-order (Suzuki) roughly doubles this.

## Configurations

`F/step` is `(1-0.0014)^gates`. "Steps to 1/e" is where cumulative circuit fidelity hits
0.368, i.e. a budget of ~714 two-qubit gates. Readout is `(1-0.0067)^qubits`.

| L | N | qubits | 2q gates/step | F/step | steps to 1/e | readout F |
|---|---|--------|---------------|--------|--------------|-----------|
| 3 | 6  | 21  | 40  | 0.946 | ~18 | 0.87 |
| 3 | 8  | 27  | 52  | 0.930 | ~14 | 0.83 |
| 4 | 6  | 28  | 54  | 0.927 | ~13 | 0.83 |
| 4 | 8  | 36  | 70  | 0.906 | ~10 | 0.78 |
| 5 | 20 | 105 | 208 | 0.747 | ~3  | 0.49 |

The last row is the "fill the chip" configuration. It gets ~3 Trotter steps and loses half
its signal to readout alone.

## Does the physics fit in the step budget?

Under pure displacement the mode is coherently driven, `⟨n⟩ = (Dt)²`, so reaching the
ceiling `n = N-1` takes roughly

```
D t*  ≈  √(N-1)
```

Spreading that over `M` Trotter steps gives a per-step rotation angle `D·δt ≈ √(N-1)/M`.
Trotter error grows with this angle, so we want it small:

| config | steps to 1/e (M) | √(N-1)/M | verdict |
|---|---|---|---|
| L=3, N=6  | 18 | 0.12 | comfortable |
| L=3, N=8  | 14 | 0.19 | workable |
| L=4, N=8  | 10 | 0.26 | marginal |
| L=5, N=20 | 3  | 1.45 | hopeless |

This is the crux. The full-grid configuration fails twice over: not enough steps to reach
the firing threshold, and each step so coarse that Trotter error dominates anyway.

**`L=3, N=6` reaches first firing with room to spare.** Observing *propagation* needs
further evolution after the flip, so realistically 30–40 steps, pushing raw fidelity to
~0.2 before post-selection. Whether that is enough is the main open question for step 4.

## Caveats

- Gate-count-based fidelity ignores crosstalk, leakage, and idling decoherence. Expect the
  real numbers to be worse.
- Assumes 2 native gates per `XX+YY` and per controlled-RX. Confirm against the actual
  Willow native gate set in step 3; the `√iSWAP` decomposition may do better.
- Assumes SWAP-free embedding. If the layout needs routing, all counts inflate.
- The ± -basis merge of `H_CNOT` with the `ZZ` part of `H_spin` (D-5) should shave gates off
  every step — not yet included above.
- Readout fidelity is quoted over the whole register, but we may only need a few qubits
  read out. Post-selection on the unary constraint, however, *does* require reading every
  boson column, so the full-register figure is roughly right.
