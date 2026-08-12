"""Executable consistency checks for the D-18 fidelity schedule, post-selection, and the
second-order Trotter circuit added 2026-08-11. Project convention (CLAUDE.md): "verify
numerically before asserting" -- this makes each claim executable rather than assumed.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/consistency_checks.py
(GoogleQVM env -- needs qutip, cirq, cirq_google, qsimcirq)

Does NOT duplicate src/symmetry_check.py (Pi_j X_j commutes with H, D-9) -- that check is
independent of everything touched today and still passes on its own; run it separately.
"""
import sys
import numpy as np
import cirq
import cirq_google
import qsimcirq
from qutip import basis, tensor, sesolve, fock

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    find_low_error_qubit_embedding,
    map_and_compile_for_willow,
    build_trotter_circuit,
    build_second_order_trotter_circuit,
    trotter_final_statevector,
    noiseless_state_fidelity,
    unary_subspace_projection,
    sample_shots_with_postselection,
    recommended_trotter_steps,
    TROTTER_SCHEDULE_C,
    TROTTER_SCHEDULE_R_MAX,
    TROTTER_SCHEDULE_T_CAP,
)

N = 5
L = 1
D_list = [1.0]
J = 0.5

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


print("Calibrating G...")
G = find_optimal_SpinBosonInteractionCoefficent(N, D_list[0])
print(f"G = {G:.6g}\n")

psi0 = tensor(basis(2, 0), fock(N, 0))
H, n_list, sz_list = QutipHamiltonian(L, N, D_list, J, G)


# ---------------------------------------------------------------------------
# 1. G calibration is deterministic / reproducible
# ---------------------------------------------------------------------------
G2 = find_optimal_SpinBosonInteractionCoefficent(N, D_list[0])
check("G calibration is reproducible", abs(G - G2) < 1e-12, f"G={G:.6g}, G2={G2:.6g}")


# ---------------------------------------------------------------------------
# 2. First-order circuit: noiseless unary-subspace leakage is exactly 0
# ---------------------------------------------------------------------------
max_leak_1st = 0.0
for t, r in [(0.5, 1), (3.0, 5), (7.0, 20), (10.0, 40)]:
    sv = trotter_final_statevector(D_list, J, N, L, t / r, r, G, circuit_builder=build_trotter_circuit)
    _, leaked = unary_subspace_projection(sv, N, L)
    max_leak_1st = max(max_leak_1st, abs(leaked))
check("1st-order circuit conserves the unary subspace exactly (noiseless)",
      max_leak_1st < 1e-6, f"max leaked probability = {max_leak_1st:.2e}")


# ---------------------------------------------------------------------------
# 3. Second-order circuit: same conservation law, same tolerance
# ---------------------------------------------------------------------------
max_leak_2nd = 0.0
for t, r in [(0.5, 1), (3.0, 5), (7.0, 20), (10.0, 40)]:
    sv = trotter_final_statevector(D_list, J, N, L, t / r, r, G, circuit_builder=build_second_order_trotter_circuit)
    _, leaked = unary_subspace_projection(sv, N, L)
    max_leak_2nd = max(max_leak_2nd, abs(leaked))
check("2nd-order circuit conserves the unary subspace exactly (noiseless)",
      max_leak_2nd < 1e-6, f"max leaked probability = {max_leak_2nd:.2e}")


# ---------------------------------------------------------------------------
# 4. Fidelity trend: more steps should get closer to the qutip state, at fixed t
# ---------------------------------------------------------------------------
t_probe = 6.0
qutip_state = sesolve(H, psi0, [0, t_probe]).states[-1]
fids_1st = []
for r in [1, 5, 20, 60]:
    sv = trotter_final_statevector(D_list, J, N, L, t_probe / r, r, G, circuit_builder=build_trotter_circuit)
    f, _ = noiseless_state_fidelity(qutip_state, sv, N, L)
    fids_1st.append(f)
check("1st-order fidelity trends toward 1 as r grows (t=6, r=1,5,20,60)",
      fids_1st[-1] > fids_1st[0] and fids_1st[-1] > 0.9,
      f"F={[round(f,4) for f in fids_1st]}")


# ---------------------------------------------------------------------------
# 5. Second order should converge faster (higher order in r) than first order -- this is
# the actual second-order claim, checked as a log-log convergence-rate fit rather than
# "2nd order beats 1st order pointwise at every r" (not guaranteed at large step size /
# small r, where prefactors can dominate -- see notes/second-order-trotter.md).
# ---------------------------------------------------------------------------
t_probe3 = 3.0
qs3 = sesolve(H, psi0, [0, t_probe3]).states[-1]
rs_fit = [16, 32, 64, 128]
e1_fit, e2_fit = [], []
for r in rs_fit:
    dt = t_probe3 / r
    sv1 = trotter_final_statevector(D_list, J, N, L, dt, r, G, circuit_builder=build_trotter_circuit)
    sv2 = trotter_final_statevector(D_list, J, N, L, dt, r, G, circuit_builder=build_second_order_trotter_circuit)
    f1, _ = noiseless_state_fidelity(qs3, sv1, N, L)
    f2, _ = noiseless_state_fidelity(qs3, sv2, N, L)
    e1_fit.append(1 - f1)
    e2_fit.append(1 - f2)
slope1 = float(np.polyfit(np.log(rs_fit), np.log(e1_fit), 1)[0])
slope2 = float(np.polyfit(np.log(rs_fit), np.log(e2_fit), 1)[0])
check("1st-order infidelity ~ r^-2 (log-log slope near -2)",
      -2.5 < slope1 < -1.5, f"slope={slope1:.2f}")
check("2nd-order infidelity ~ r^-4, i.e. converges strictly faster than 1st order",
      slope2 < slope1 - 1.0, f"slope1={slope1:.2f}, slope2={slope2:.2f}")
check("2nd order is more accurate than 1st order at the largest tested r (t=3, r=128)",
      e2_fit[-1] < e1_fit[-1], f"1-F: 1st={e1_fit[-1]:.2e}, 2nd={e2_fit[-1]:.2e}")


# ---------------------------------------------------------------------------
# 6. Gate-count structure of the merged 2nd-order circuit: boundary merging should keep
# per-step overhead close to (r+1)/r * (1st order's boson-gate count), not the ~2x a
# naive (unmerged) symmetric composition would cost. cirq.decompose() expands each
# logical ISwapPowGate/controlled-rx into several native primitives, so raw gate counts
# aren't simply r*(N-1)+r -- the merge claim is about the *ratio* between two step
# counts (isolates the boundary-layer overhead from decomposition's constant factor).
# ---------------------------------------------------------------------------
def two_qubit_gate_count(D_list, J, N, L, r, G, builder):
    circuit, _ = builder(D_list, J, N, L, 1.0, r, G)
    return sum(1 for op in circuit.all_operations() if len(op.qubits) == 2)


gates_1st_r6 = two_qubit_gate_count(D_list, J, N, L, 6, G, build_trotter_circuit)
gates_1st_r12 = two_qubit_gate_count(D_list, J, N, L, 12, G, build_trotter_circuit)
gates_2nd_r6 = two_qubit_gate_count(D_list, J, N, L, 6, G, build_second_order_trotter_circuit)
gates_2nd_r12 = two_qubit_gate_count(D_list, J, N, L, 12, G, build_second_order_trotter_circuit)
# Doubling r should ~double both circuits' gate counts (linear in r, no fixed overhead
# dominating) -- sanity check that merging isn't secretly still O(r) but with a huge
# per-step constant hiding elsewhere.
ratio_1st = gates_1st_r12 / gates_1st_r6
ratio_2nd = gates_2nd_r12 / gates_2nd_r6
check("1st-order gate count is ~linear in r (doubling r ~doubles gate count)",
      1.8 < ratio_1st < 2.1, f"gates(r=6)={gates_1st_r6}, gates(r=12)={gates_1st_r12}, ratio={ratio_1st:.3f}")
check("2nd-order gate count is ~linear in r (merge keeps per-step overhead constant, not compounding)",
      1.8 < ratio_2nd < 2.1, f"gates(r=6)={gates_2nd_r6}, gates(r=12)={gates_2nd_r12}, ratio={ratio_2nd:.3f}")
# 2nd order's per-step overhead vs 1st order: merging collapses the Even-bond boundary
# layers (cost ~unchanged vs 1st order), but the Odd-bond layer is applied TWICE per step
# (never sits at a step boundary, so it can't merge) -- so the expected overhead is ~1.5x
# (one extra Odd application per step, roughly half of the boson gate budget), clearly
# below a naive fully-unmerged symmetric composition's ~2x, but not as low as ~1x.
overhead_ratio = gates_2nd_r12 / gates_1st_r12
check("2nd-order gate count overhead matches the derivation (~1.5x from Odd applied twice per step), well under a naive 2x",
      1.3 < overhead_ratio < 1.75, f"gates_2nd(r=12)/gates_1st(r=12) = {overhead_ratio:.3f}")


# ---------------------------------------------------------------------------
# 7. Post-selection: noiseless circuit -> always one-hot (survival_rate == 1); noisy
#    circuit -> some detectable leakage (survival_rate < 1).
# ---------------------------------------------------------------------------
processor_id = "willow_pink"
willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
willow_noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(
    cirq_google.engine.load_device_noise_properties(processor_id)
)
willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]
total_qubits = L * N + L
willow_qubit_chain = find_low_error_qubit_embedding(willow_device, willow_calibration, N, L)

full_circuit, all_qubits = build_trotter_circuit(D_list, J, N, L, 6.0 / 7, 7, G)
compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)

noiseless_sim = qsimcirq.QSimSimulator(noise=None)
noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model)

noiseless_shots = sample_shots_with_postselection(noiseless_sim, compiled_circuit, willow_qubit_chain, N, L, 500)
noisy_shots = sample_shots_with_postselection(noisy_sim, compiled_circuit, willow_qubit_chain, N, L, 2000)

check("noiseless post-selection survival rate is exactly 1.0",
      noiseless_shots['survival_rate'] == 1.0, f"got {noiseless_shots['survival_rate']}")
check("noisy post-selection survival rate is strictly less than 1.0 (detects real leakage)",
      0.0 < noisy_shots['survival_rate'] < 1.0, f"got {noisy_shots['survival_rate']:.4f}")
check("post-selected and raw estimates agree when survival rate is 1.0 (noiseless)",
      np.allclose(noiseless_shots['occ_post'], noiseless_shots['occ_raw'], atol=1e-9) and
      np.allclose(noiseless_shots['mag_post'], noiseless_shots['mag_raw'], atol=1e-9),
      f"occ_post={noiseless_shots['occ_post']}, occ_raw={noiseless_shots['occ_raw']}")


# ---------------------------------------------------------------------------
# 8. Schedule sanity: monotone, cap is internally consistent, matches config used above
# ---------------------------------------------------------------------------
schedule_vals = [recommended_trotter_steps(t) for t in np.linspace(0.1, 20, 50)]
check("recommended_trotter_steps(t) is monotone non-decreasing in t",
      all(b >= a for a, b in zip(schedule_vals, schedule_vals[1:])),
      f"C={TROTTER_SCHEDULE_C}")
check("TROTTER_SCHEDULE_T_CAP is consistent with R_MAX / C",
      abs(TROTTER_SCHEDULE_T_CAP - TROTTER_SCHEDULE_R_MAX / TROTTER_SCHEDULE_C) < 1e-9)


# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
else:
    print("All consistency checks passed.")
