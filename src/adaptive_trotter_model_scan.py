"""Scan linear (r = a*t) vs quadratic (r = b*t^2) adaptive Trotter-step schedules,
noise ON + post-selection ON throughout, sweeping each model's coefficient to find the
value that minimizes error against the qutip ground truth -- then compares the two
functional forms directly on that metric.

Context (2026-08-12 discussion, not yet a DECISIONS.md entry): D-18/D-19 adopted a
*noiseless-fidelity*-derived schedule r*(t) = round(4.0*t), linear in t. Standard
first-order Trotter error theory (Childs, Su, Tran, Wiebe, Zhu, "A Theory of Trotter
Error", commutator-scaling bound, Eq. 2 with p=1) gives per-application error O(t^2) for
a single formula application over duration t; chopping total time T into r steps and
summing via triangle inequality gives total error ~O(T^2/r) -- so holding a FIXED error
tolerance as T grows requires r ~ T^2, not r ~ T. D-17's own derivation used exactly this
T^2/r form, but that schedule optimizes a *trade-off* (Trotter error + noise error, noise
~B*r) rather than fixed accuracy, and its result (r*=t*sqrt(A/B)) is linear by construction
of that optimization -- not evidence against the T^2/r fixed-tolerance law. D-18/D-19's
r_T(t), by contrast, IS a fixed-tolerance quantity, and was fit as linear without testing
a quadratic form against the same data. This script sidesteps the theory argument and
just measures both forms directly, on the metric that actually matters operationally:
noisy, post-selected Cirq observable error vs qutip (not noiseless fidelity).

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/adaptive_trotter_model_scan.py
(GoogleQVM env -- qsimcirq/cirq_google, D-14)
Pass --smoke-test for a fast, tiny-grid sanity run before the full scan.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sesolve, expect, fock
import cirq
import cirq_google
import qsimcirq

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    find_low_error_qubit_embedding,
    map_and_compile_for_willow,
    build_trotter_circuit,
    sample_shots_with_postselection,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
LINEAR_DIR = os.path.join(RESULTS_DIR, "linear trotter")
QUADRATIC_DIR = os.path.join(RESULTS_DIR, "quadratic trotter")
os.makedirs(LINEAR_DIR, exist_ok=True)
os.makedirs(QUADRATIC_DIR, exist_ok=True)

SMOKE_TEST = "--smoke-test" in sys.argv

#------------------------
# CONFIG -- matches NuronSim.py's current defaults / _SCHEDULE_CONFIG (D-19), so this is
# directly comparable to the existing fidelity-derived schedule.
#------------------------
NumberOfBosonicModes = 1
NumberOfFockStates = 5
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]
spin_interaction_coefficient = 0.5
total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes

T_MIN, T_MAX = 0.5, 10.0
R_CAP = 60  # hard cap on step count per point -- bounds circuit depth/runtime; matches the
            # largest step count tested elsewhere (steps_grid in optimal_trotter_steps.py)

if SMOKE_TEST:
    N_TIME_POINTS = 3
    NUM_SHOTS = 100
    A_VALUES = [4]
    B_VALUES = [0.2]
else:
    N_TIME_POINTS = 20
    NUM_SHOTS = 800
    A_VALUES = [0.5, 1, 2, 3, 4, 6, 8, 10]
    B_VALUES = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]

time_data = np.linspace(T_MIN, T_MAX, N_TIME_POINTS)

#------------------------
# Ground truth (qutip)
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
    """Same normalized combined-error metric as optimal_trotter_steps.py (D-16/D-17)."""
    qutip_occ, qutip_mag = qutip_ground_truth(t)
    boson_err = (occ - qutip_occ) / (NumberOfFockStates - 1)
    spin_err = (mag - qutip_mag) / 2.0
    return float(np.sqrt(boson_err**2 + spin_err**2))


#------------------------
# Willow device + noise model (shared across both scans)
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


def run_schedule_sweep(schedule_fn, param_value):
    """Runs the noisy, post-selected Cirq sim across time_data with
    r(t) = clip(round(schedule_fn(t, param_value)), 1, R_CAP)."""
    occ_arr = np.zeros(N_TIME_POINTS)
    mag_arr = np.zeros(N_TIME_POINTS)
    err_arr = np.zeros(N_TIME_POINTS)
    survival_arr = np.zeros(N_TIME_POINTS)
    steps_arr = np.zeros(N_TIME_POINTS, dtype=int)

    for i, t in enumerate(time_data):
        r = int(min(max(round(schedule_fn(t, param_value)), 1), R_CAP))
        steps_arr[i] = r
        dt = t / r
        full_circuit, all_qubits = build_trotter_circuit(
            D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
            dt, r, SpinBosonInteractionCoefficent
        )
        compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
        shots = sample_shots_with_postselection(
            noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes, NUM_SHOTS
        )
        occ_arr[i] = shots['occ_post'][0]
        mag_arr[i] = shots['mag_post'][0]
        survival_arr[i] = shots['survival_rate']
        err_arr[i] = combined_error(occ_arr[i], mag_arr[i], t)

    rms_error = float(np.sqrt(np.mean(err_arr ** 2)))
    return dict(occ=occ_arr, mag=mag_arr, err=err_arr, survival=survival_arr, steps=steps_arr, rms_error=rms_error)


def plot_run(result, model_label, param_symbol, param_value, output_dir, tag):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(qutip_times, qutip_occ_curve, color='#d62728', linewidth=1.5, label='QuTiP occupation')
    ax2.plot(qutip_times, qutip_mag_curve, color='#1f77b4', linewidth=1.5, label='QuTiP magnetisation')
    ax1.scatter(time_data, result['occ'], color='#d62728', marker='o', edgecolors='black',
                linewidths=0.5, zorder=5, label='Cirq occupation (noisy, post-selected)')
    ax2.scatter(time_data, result['mag'], color='#1f77b4', marker='s', edgecolors='black',
                linewidths=0.5, zorder=5, label='Cirq magnetisation (noisy, post-selected)')

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
        f"{model_label}: {param_symbol} = {param_value:g}   |   RMS error = {result['rms_error']:.4f}   "
        f"|   mean survival = {np.mean(result['survival']):.1%}",
        fontsize=10,
    )

    ax3.plot(time_data, result['err'], marker='o', color='tab:green', label='combined error vs qutip')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('combined error', color='tab:green')
    ax3b = ax3.twinx()
    ax3b.plot(time_data, result['steps'], marker='s', color='tab:gray', alpha=0.6, label='steps r(t)')
    ax3b.set_ylabel('Trotter steps r', color='tab:gray')
    ax3.set_title('Per-point error and step count', fontsize=10)

    fig.suptitle(
        f"N={NumberOfFockStates}, D={D_list[0]:.3g}, J={spin_interaction_coefficient}, "
        f"G={SpinBosonInteractionCoefficent:.3g}, noise ON, post-selection ON, willow_pink",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = os.path.join(output_dir, f"{tag}.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


def scan_model(schedule_fn, param_values, model_label, param_symbol, tag_prefix, output_dir):
    results = {}
    t_start = time.time()
    for pv in param_values:
        result = run_schedule_sweep(schedule_fn, pv)
        results[pv] = result
        tag = f"{tag_prefix}_{param_symbol}{pv:g}"
        fname = plot_run(result, model_label, param_symbol, pv, output_dir, tag)
        np.savez(
            os.path.join(output_dir, f"{tag}.npz"),
            time_data=time_data, occ=result['occ'], mag=result['mag'], err=result['err'],
            survival=result['survival'], steps=result['steps'], rms_error=result['rms_error'],
            param_value=pv, qutip_time=qutip_times, qutip_occ=qutip_occ_curve, qutip_mag=qutip_mag_curve,
        )
        print(f"  [{model_label}] {param_symbol}={pv:g}: RMS error={result['rms_error']:.4f}, "
              f"mean steps={np.mean(result['steps']):.1f}, mean survival={np.mean(result['survival']):.1%}  "
              f"({time.time()-t_start:.0f}s elapsed) -> {fname}")
    return results


#------------------------
# Linear model: r = a * t
#------------------------
print(f"\n=== Scanning LINEAR model r = a*t over a = {A_VALUES} ===")
linear_results = scan_model(
    lambda t, a: a * t, A_VALUES, "Linear r=a*t", "a", "linear", LINEAR_DIR
)

#------------------------
# Quadratic model: r = b * t^2
#------------------------
print(f"\n=== Scanning QUADRATIC model r = b*t^2 over b = {B_VALUES} ===")
quadratic_results = scan_model(
    lambda t, b: b * t ** 2, B_VALUES, "Quadratic r=b*t^2", "b", "quadratic", QUADRATIC_DIR
)

#------------------------
# Per-model summary plots (RMS error vs coefficient)
#------------------------
def summary_plot(results, param_values, param_symbol, model_label, output_dir):
    rms_errors = [results[pv]['rms_error'] for pv in param_values]
    mean_steps = [float(np.mean(results[pv]['steps'])) for pv in param_values]
    best_pv = param_values[int(np.argmin(rms_errors))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(param_values, rms_errors, marker='o', color='tab:green')
    ax1.axvline(best_pv, color='tab:red', linestyle='--', label=f'best {param_symbol}={best_pv:g}')
    ax1.set_xlabel(param_symbol)
    ax1.set_ylabel('RMS combined error vs qutip')
    ax1.set_title(f'{model_label}: error vs {param_symbol}')
    ax1.legend(fontsize=8)

    ax2.plot(param_values, mean_steps, marker='s', color='tab:gray')
    ax2.set_xlabel(param_symbol)
    ax2.set_ylabel('mean Trotter steps over sweep')
    ax2.set_title(f'{model_label}: resource cost vs {param_symbol}')

    fig.suptitle(f"Noise ON, post-selection ON, t in [{T_MIN},{T_MAX}], {NUM_SHOTS} shots/point", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fname = os.path.join(output_dir, "summary.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"Saved {fname}")
    return best_pv, dict(zip(param_values, rms_errors)), dict(zip(param_values, mean_steps))


best_a, linear_rms_by_a, linear_steps_by_a = summary_plot(linear_results, A_VALUES, "a", "Linear r=a*t", LINEAR_DIR)
best_b, quadratic_rms_by_b, quadratic_steps_by_b = summary_plot(quadratic_results, B_VALUES, "b", "Quadratic r=b*t^2", QUADRATIC_DIR)

print(f"\nBest linear coefficient: a = {best_a:g}  (RMS error = {linear_rms_by_a[best_a]:.4f}, "
      f"mean steps = {linear_steps_by_a[best_a]:.1f})")
print(f"Best quadratic coefficient: b = {best_b:g}  (RMS error = {quadratic_rms_by_b[best_b]:.4f}, "
      f"mean steps = {quadratic_steps_by_b[best_b]:.1f})")

#------------------------
# Final comparison: best linear vs best quadratic, head to head
#------------------------
best_linear = linear_results[best_a]
best_quadratic = quadratic_results[best_b]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(time_data, best_linear['err'], marker='o', color='tab:blue', label=f'linear, a={best_a:g} (RMS={best_linear["rms_error"]:.4f})')
ax1.plot(time_data, best_quadratic['err'], marker='s', color='tab:orange', label=f'quadratic, b={best_b:g} (RMS={best_quadratic["rms_error"]:.4f})')
ax1.set_xlabel('Time')
ax1.set_ylabel('combined error vs qutip')
ax1.set_title('Best-of-each-model error vs t')
ax1.legend(fontsize=8)

ax2.plot(time_data, best_linear['steps'], marker='o', color='tab:blue', label=f'linear, a={best_a:g}')
ax2.plot(time_data, best_quadratic['steps'], marker='s', color='tab:orange', label=f'quadratic, b={best_b:g}')
ax2.set_xlabel('Time')
ax2.set_ylabel('Trotter steps r(t)')
ax2.set_title('Resource cost (step count) vs t')
ax2.legend(fontsize=8)

winner = "linear" if best_linear['rms_error'] < best_quadratic['rms_error'] else "quadratic"
fig.suptitle(
    f"Best linear (a={best_a:g}) vs best quadratic (b={best_b:g})  --  winner: {winner}  "
    f"(noise ON, post-selection ON)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.9])
comparison_path = os.path.join(RESULTS_DIR, "linear_vs_quadratic_trotter_comparison.png")
fig.savefig(comparison_path, dpi=130)
plt.close(fig)

np.savez(
    os.path.join(RESULTS_DIR, "linear_vs_quadratic_trotter_comparison.npz"),
    time_data=time_data,
    A_VALUES=np.array(A_VALUES), B_VALUES=np.array(B_VALUES),
    linear_rms_by_a=np.array([linear_rms_by_a[a] for a in A_VALUES]),
    quadratic_rms_by_b=np.array([quadratic_rms_by_b[b] for b in B_VALUES]),
    best_a=best_a, best_b=best_b,
    best_linear_rms=best_linear['rms_error'], best_quadratic_rms=best_quadratic['rms_error'],
    best_linear_err=best_linear['err'], best_quadratic_err=best_quadratic['err'],
    best_linear_steps=best_linear['steps'], best_quadratic_steps=best_quadratic['steps'],
)

print(f"\n=== FINAL COMPARISON ===")
print(f"Best linear   (r=a*t,   a={best_a:g}): RMS error = {best_linear['rms_error']:.4f}, "
      f"mean steps = {np.mean(best_linear['steps']):.1f}")
print(f"Best quadratic (r=b*t^2, b={best_b:g}): RMS error = {best_quadratic['rms_error']:.4f}, "
      f"mean steps = {np.mean(best_quadratic['steps']):.1f}")
print(f"Winner (lower RMS error): {winner}")
print(f"Saved {comparison_path}")
