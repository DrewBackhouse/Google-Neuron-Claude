"""D-22: fit a second-order-specific linear Trotter schedule r=a*t (mirroring D-21's
method for first order) and compare the optimized second-order result against D-21's
optimized first-order result (a=2.25, RMS=0.1565).

Context: NeuronSim2ndOrderTrotter.py currently reuses recommended_trotter_steps(t) -- the
FIRST-order-fitted schedule -- for an apples-to-apples matched-step comparison (D-20).
That was deliberate at the time, but flagged as leaving real second-order headroom on the
table: Strang splitting converges as ~r^-2 in operator-norm error (vs first order's ~r^-1),
so for the same Trotter accuracy it needs far fewer steps -- even though each step costs
~1.51x the two-qubit gates (D-20). D-17's own trade-off derivation, redone for a
second-order error term A*t^3/r^2 instead of first-order's A*t^2/r (minimizing against
noise ~B*r), still gives a schedule LINEAR in t (r* = t*(2A/B)^(1/3)) -- just a different,
data-determined constant. This script finds that constant directly against the operational
metric (noisy, post-selected Cirq error vs qutip), same method as D-21:
  1. Coarse scan over a broad `a` grid.
  2. Fine scan (more time points, more shots) around the coarse winner.
  3. Head-to-head comparison against D-21's saved first-order-optimal result.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/second_order_trotter_linear_scan.py
(GoogleQVM env -- qsimcirq/cirq_google, D-14)
Pass --smoke-test for a fast, tiny-grid sanity run before the full scan.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sesolve, expect, fock
import cirq_google
import qsimcirq

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    find_low_error_qubit_embedding,
    map_and_compile_for_willow,
    build_second_order_trotter_circuit,
    sample_shots_with_postselection,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "2nd order linear trotter")
FIRST_ORDER_DIR = os.path.join(RESULTS_DIR, "linear trotter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOKE_TEST = "--smoke-test" in sys.argv

#------------------------
# CONFIG -- same model as D-21 / NuronSim.py defaults, so directly comparable
#------------------------
NumberOfBosonicModes = 1
NumberOfFockStates = 5
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]
spin_interaction_coefficient = 0.5
total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes

T_MIN, T_MAX = 0.5, 10.0
R_CAP = 60

# D-21's saved fine-grid result for the first-order optimum (a=2.25) -- reused rather than
# recomputed, since it's the exact same config/t-range and already the higher-resolution
# (40 pts, 1200 shots) run.
FIRST_ORDER_BEST_A = 2.25
FIRST_ORDER_NPZ = os.path.join(FIRST_ORDER_DIR, f"linear_finegrid_a{FIRST_ORDER_BEST_A:g}.npz")

if SMOKE_TEST:
    COARSE_N_TIME_POINTS, COARSE_SHOTS = 3, 100
    COARSE_A_VALUES = [1.0]
    FINE_N_TIME_POINTS, FINE_SHOTS = 3, 100
    FINE_A_STEP, FINE_A_HALF_WIDTH = 0.5, 0.5
else:
    # Matches D-21's coarse scan settings exactly (20 pts/800 shots) for comparability.
    COARSE_N_TIME_POINTS, COARSE_SHOTS = 20, 800
    COARSE_A_VALUES = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
    # Matches D-21's fine scan settings exactly (40 pts/1200 shots).
    FINE_N_TIME_POINTS, FINE_SHOTS = 40, 1200
    FINE_A_STEP, FINE_A_HALF_WIDTH = 0.25, 0.75  # 7 points centered on the coarse winner

#------------------------
# Ground truth (qutip) -- shared by coarse + fine scans
#------------------------
print("Calibrating G and solving the qutip ground truth...")
SpinBosonInteractionCoefficent = find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, D_list[0])
print(f"G = {SpinBosonInteractionCoefficent:.4g}")

psi0 = tensor(basis(2, 0), fock(NumberOfFockStates, 0))
H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent)
qutip_times = np.linspace(0, T_MAX, int(T_MAX * 20) + 1)
qutip_result = sesolve(H, psi0, qutip_times)
qutip_occ_curve = expect(n_list[0], qutip_result.states)
qutip_mag_curve = expect(sz_list[0], qutip_result.states)


def qutip_ground_truth(t):
    occ = np.interp(t, qutip_times, qutip_occ_curve)
    mag = np.interp(t, qutip_times, qutip_mag_curve)
    return occ, mag


def combined_error(occ, mag, t):
    """Same normalized combined-error metric as D-16/D-17/D-21."""
    qutip_occ, qutip_mag = qutip_ground_truth(t)
    boson_err = (occ - qutip_occ) / (NumberOfFockStates - 1)
    spin_err = (mag - qutip_mag) / 2.0
    return float(np.sqrt(boson_err**2 + spin_err**2))


#------------------------
# Willow device + noise model
#------------------------
print("Setting up Willow device, calibration, and noise model...")
processor_id = "willow_pink"
willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
willow_noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(
    cirq_google.engine.load_device_noise_properties(processor_id)
)
willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]
willow_qubit_chain = find_low_error_qubit_embedding(willow_device, willow_calibration, NumberOfFockStates, NumberOfBosonicModes)
noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model)
print(f"Mapped {total_qubits} qubits onto Willow ({processor_id}): " + " - ".join(str(q) for q in willow_qubit_chain))


def run_schedule_sweep(a, time_data, num_shots):
    """Second-order (Strang) circuit, r(t) = clip(round(a*t), 1, R_CAP)."""
    n = len(time_data)
    occ_arr = np.zeros(n)
    mag_arr = np.zeros(n)
    err_arr = np.zeros(n)
    survival_arr = np.zeros(n)
    steps_arr = np.zeros(n, dtype=int)

    for i, t in enumerate(time_data):
        r = int(min(max(round(a * t), 1), R_CAP))
        steps_arr[i] = r
        dt = t / r
        full_circuit, all_qubits = build_second_order_trotter_circuit(
            D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
            dt, r, SpinBosonInteractionCoefficent
        )
        compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
        shots = sample_shots_with_postselection(
            noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes, num_shots
        )
        occ_arr[i] = shots['occ_post'][0]
        mag_arr[i] = shots['mag_post'][0]
        survival_arr[i] = shots['survival_rate']
        err_arr[i] = combined_error(occ_arr[i], mag_arr[i], t)

    rms_error = float(np.sqrt(np.mean(err_arr ** 2)))
    return dict(occ=occ_arr, mag=mag_arr, err=err_arr, survival=survival_arr, steps=steps_arr, rms_error=rms_error)


def plot_run(result, a, time_data, num_shots, tag):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(qutip_times, qutip_occ_curve, color='#d62728', linewidth=1.5, label='QuTiP occupation')
    ax2.plot(qutip_times, qutip_mag_curve, color='#1f77b4', linewidth=1.5, label='QuTiP magnetisation')
    ax1.scatter(time_data, result['occ'], color='#d62728', marker='o', edgecolors='black',
                linewidths=0.5, s=18, zorder=5, label='Cirq 2nd-order occupation (noisy, post-selected)')
    ax2.scatter(time_data, result['mag'], color='#1f77b4', marker='s', edgecolors='black',
                linewidths=0.5, s=18, zorder=5, label='Cirq 2nd-order magnetisation (noisy, post-selected)')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Bosonic occupation', color='#d62728')
    ax2.set_ylabel('Spin magnetisation', color='#1f77b4')
    boson_max = max(np.max(qutip_occ_curve), 1e-9)
    ax1.set_ylim(-0.1 * boson_max, 1.1 * boson_max)
    ax2.set_ylim(-1.1, 1.1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')
    ax1.set_title(
        f"2nd-order, linear r=a*t: a = {a:g}   |   RMS error = {result['rms_error']:.4f}   "
        f"|   mean survival = {np.mean(result['survival']):.1%}",
        fontsize=10,
    )

    ax3.plot(time_data, result['err'], marker='o', markersize=3, color='tab:green', label='combined error vs qutip')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('combined error', color='tab:green')
    ax3b = ax3.twinx()
    ax3b.plot(time_data, result['steps'], marker='s', markersize=3, color='tab:gray', alpha=0.6, label='steps r(t)')
    ax3b.set_ylabel('Trotter steps r', color='tab:gray')
    ax3.set_title('Per-point error and step count', fontsize=10)

    fig.suptitle(
        f"N={NumberOfFockStates}, D={D_list[0]:.3g}, J={spin_interaction_coefficient}, "
        f"G={SpinBosonInteractionCoefficent:.3g}, noise ON, post-selection ON, willow_pink "
        f"({len(time_data)} t-points, {num_shots} shots/point)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = os.path.join(OUTPUT_DIR, f"{tag}.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


#------------------------
# Stage 1: coarse scan
#------------------------
coarse_time_data = np.linspace(T_MIN, T_MAX, COARSE_N_TIME_POINTS)
print(f"\n=== COARSE scan: 2nd-order linear r=a*t over a = {COARSE_A_VALUES} "
      f"({COARSE_N_TIME_POINTS} t-points, {COARSE_SHOTS} shots/point) ===")
coarse_results = {}
t_start = time.time()
for a in COARSE_A_VALUES:
    result = run_schedule_sweep(a, coarse_time_data, COARSE_SHOTS)
    coarse_results[a] = result
    fname = plot_run(result, a, coarse_time_data, COARSE_SHOTS, f"coarse_a{a:g}")
    np.savez(os.path.join(OUTPUT_DIR, f"coarse_a{a:g}.npz"),
             time_data=coarse_time_data, **result, a=a)
    print(f"  a={a:g}: RMS error={result['rms_error']:.4f}, mean steps={np.mean(result['steps']):.1f}, "
          f"mean survival={np.mean(result['survival']):.1%}  ({time.time()-t_start:.0f}s elapsed) -> {fname}")

coarse_rms = np.array([coarse_results[a]['rms_error'] for a in COARSE_A_VALUES])
coarse_best_a = COARSE_A_VALUES[int(np.argmin(coarse_rms))]
print(f"\nCoarse winner: a = {coarse_best_a:g} (RMS = {coarse_results[coarse_best_a]['rms_error']:.4f})")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(COARSE_A_VALUES, coarse_rms, marker='o', color='tab:green')
ax.axvline(coarse_best_a, color='tab:red', linestyle='--', label=f'coarse best a={coarse_best_a:g}')
ax.set_xlabel('a')
ax.set_ylabel('RMS combined error vs qutip')
ax.set_title('2nd-order coarse scan: error vs a')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "summary_coarse.png"), dpi=130)
plt.close(fig)

#------------------------
# Stage 2: fine scan around the coarse winner
#------------------------
fine_a_values = sorted(set(
    round(coarse_best_a + k * FINE_A_STEP, 4)
    for k in range(-int(round(FINE_A_HALF_WIDTH / FINE_A_STEP)), int(round(FINE_A_HALF_WIDTH / FINE_A_STEP)) + 1)
    if coarse_best_a + k * FINE_A_STEP > 0
))
fine_time_data = np.linspace(T_MIN, T_MAX, FINE_N_TIME_POINTS)
print(f"\n=== FINE scan: 2nd-order linear r=a*t over a = {fine_a_values} "
      f"({FINE_N_TIME_POINTS} t-points, {FINE_SHOTS} shots/point) ===")
fine_results = {}
t_start = time.time()
for a in fine_a_values:
    result = run_schedule_sweep(a, fine_time_data, FINE_SHOTS)
    fine_results[a] = result
    fname = plot_run(result, a, fine_time_data, FINE_SHOTS, f"finegrid_a{a:g}")
    np.savez(os.path.join(OUTPUT_DIR, f"finegrid_a{a:g}.npz"),
             time_data=fine_time_data, **result, a=a)
    print(f"  a={a:g}: RMS error={result['rms_error']:.4f}, mean steps={np.mean(result['steps']):.1f}, "
          f"mean survival={np.mean(result['survival']):.1%}  ({time.time()-t_start:.0f}s elapsed) -> {fname}")

fine_rms = np.array([fine_results[a]['rms_error'] for a in fine_a_values])
fine_best_a = fine_a_values[int(np.argmin(fine_rms))]
if len(fine_a_values) >= 3:
    coeffs = np.polyfit(fine_a_values, fine_rms, 2)
    a_fit_min = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] > 0 else fine_best_a
else:
    coeffs, a_fit_min = None, fine_best_a

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(fine_a_values, fine_rms, marker='o', color='tab:green', label='measured RMS error')
if coeffs is not None:
    a_dense = np.linspace(min(fine_a_values), max(fine_a_values), 200)
    ax.plot(a_dense, np.polyval(coeffs, a_dense), color='tab:red', linestyle='--',
            label=f'parabolic fit (min at a={a_fit_min:.3f})')
ax.axvline(fine_best_a, color='tab:gray', linestyle=':', label=f'best grid point a={fine_best_a:g}')
ax.set_xlabel('a')
ax.set_ylabel('RMS combined error vs qutip')
ax.set_title(f'2nd-order fine scan: error vs a\n({FINE_N_TIME_POINTS} t-points, {FINE_SHOTS} shots/point)', fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
summary_fine_path = os.path.join(OUTPUT_DIR, "summary_finegrid.png")
fig.savefig(summary_fine_path, dpi=130)
plt.close(fig)

np.savez(os.path.join(OUTPUT_DIR, "summary_finegrid.npz"),
          fine_a_values=np.array(fine_a_values), fine_rms=fine_rms,
          fine_best_a=fine_best_a, a_fit_min=a_fit_min)

print(f"\n=== 2ND-ORDER FINE-GRID RESULT ===")
for a in fine_a_values:
    print(f"  a={a:g}: RMS={fine_results[a]['rms_error']:.4f}, mean steps={np.mean(fine_results[a]['steps']):.1f}, "
          f"mean survival={np.mean(fine_results[a]['survival']):.1%}")
print(f"Best grid point: a = {fine_best_a:g} (RMS = {fine_results[fine_best_a]['rms_error']:.4f})")
print(f"Parabolic-fit minimum: a ~= {a_fit_min:.3f}")
print(f"Saved {summary_fine_path}")

#------------------------
# Stage 3: head-to-head vs D-21's optimized first-order result
#------------------------
best_second_order = fine_results[fine_best_a]

if os.path.exists(FIRST_ORDER_NPZ):
    fo = np.load(FIRST_ORDER_NPZ)
    fo_time_data = fo["time_data"]
    fo_err = fo["err"]
    fo_steps = fo["steps"]
    fo_rms = float(fo["rms_error"])
else:
    print(f"\nWARNING: {FIRST_ORDER_NPZ} not found -- skipping first-vs-second-order comparison plot.")
    fo_time_data = fo_err = fo_steps = None
    fo_rms = None

if fo_time_data is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(fo_time_data, fo_err, marker='o', color='tab:blue',
             label=f'1st order, a={FIRST_ORDER_BEST_A:g} (RMS={fo_rms:.4f})')
    ax1.plot(fine_time_data, best_second_order['err'], marker='s', color='tab:orange',
             label=f'2nd order, a={fine_best_a:g} (RMS={best_second_order["rms_error"]:.4f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('combined error vs qutip')
    ax1.set_title('Optimized 1st vs 2nd order: error vs t')
    ax1.legend(fontsize=8)

    ax2.plot(fo_time_data, fo_steps, marker='o', color='tab:blue', label=f'1st order, a={FIRST_ORDER_BEST_A:g}')
    ax2.plot(fine_time_data, best_second_order['steps'], marker='s', color='tab:orange', label=f'2nd order, a={fine_best_a:g}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Trotter steps r(t)')
    ax2.set_title('Resource cost (step count) vs t')
    ax2.legend(fontsize=8)

    winner = "1st order" if fo_rms < best_second_order['rms_error'] else "2nd order"
    fig.suptitle(
        f"Optimized 1st order (a={FIRST_ORDER_BEST_A:g}, RMS={fo_rms:.4f}) vs "
        f"optimized 2nd order (a={fine_best_a:g}, RMS={best_second_order['rms_error']:.4f})  --  "
        f"winner: {winner}  (noise ON, post-selection ON)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    comparison_path = os.path.join(RESULTS_DIR, "first_vs_second_order_optimized_trotter_comparison.png")
    fig.savefig(comparison_path, dpi=130)
    plt.close(fig)

    np.savez(
        os.path.join(RESULTS_DIR, "first_vs_second_order_optimized_trotter_comparison.npz"),
        first_order_a=FIRST_ORDER_BEST_A, first_order_rms=fo_rms,
        first_order_time=fo_time_data, first_order_err=fo_err, first_order_steps=fo_steps,
        second_order_a=fine_best_a, second_order_rms=best_second_order['rms_error'],
        second_order_time=fine_time_data, second_order_err=best_second_order['err'],
        second_order_steps=best_second_order['steps'],
    )

    print(f"\n=== FINAL COMPARISON: OPTIMIZED 1ST vs 2ND ORDER ===")
    print(f"1st order (a={FIRST_ORDER_BEST_A:g}): RMS error = {fo_rms:.4f}, mean steps = {np.mean(fo_steps):.1f}")
    print(f"2nd order (a={fine_best_a:g}): RMS error = {best_second_order['rms_error']:.4f}, "
          f"mean steps = {np.mean(best_second_order['steps']):.1f}")
    print(f"Winner (lower RMS error): {winner}")
    print(f"Saved {comparison_path}")
