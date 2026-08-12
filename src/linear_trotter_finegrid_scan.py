"""Fine-grid follow-up to adaptive_trotter_model_scan.py: the coarse scan found the
linear model r=a*t beats r=b*t^2 outright, with a broad optimum around a=2 (RMS 0.157
at a=2, vs 0.182 at a=3 and 0.557 at a=1). This script narrows in on a in [1.5, 3.0] with
a finer step and more data points per run (more time points AND more shots per point) to
pin the constant down more precisely than the coarse 8-point scan could.

Noise ON, post-selection ON throughout -- same operational metric as the coarse scan.
Same model config as NuronSim.py's defaults / _SCHEDULE_CONFIG (D-19), so directly
comparable to the existing hardcoded TROTTER_SCHEDULE_C=4.0.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/linear_trotter_finegrid_scan.py
(GoogleQVM env -- qsimcirq/cirq_google, D-14)
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
    build_trotter_circuit,
    sample_shots_with_postselection,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
LINEAR_DIR = os.path.join(RESULTS_DIR, "linear trotter")
os.makedirs(LINEAR_DIR, exist_ok=True)

SMOKE_TEST = "--smoke-test" in sys.argv

#------------------------
# CONFIG -- same model as the coarse scan / NuronSim.py defaults
#------------------------
NumberOfBosonicModes = 1
NumberOfFockStates = 5
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]
spin_interaction_coefficient = 0.5
total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes

T_MIN, T_MAX = 0.5, 10.0
R_CAP = 60

if SMOKE_TEST:
    N_TIME_POINTS = 3
    NUM_SHOTS = 100
    A_VALUES = [2.0]
else:
    # Finer than the coarse scan (20 pts/800 shots): 40 time points, 1200 shots/point.
    N_TIME_POINTS = 40
    NUM_SHOTS = 1200
    A_VALUES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

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


def run_schedule_sweep(a):
    occ_arr = np.zeros(N_TIME_POINTS)
    mag_arr = np.zeros(N_TIME_POINTS)
    err_arr = np.zeros(N_TIME_POINTS)
    survival_arr = np.zeros(N_TIME_POINTS)
    steps_arr = np.zeros(N_TIME_POINTS, dtype=int)

    for i, t in enumerate(time_data):
        r = int(min(max(round(a * t), 1), R_CAP))
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


def plot_run(result, a):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(qutip_times, qutip_occ_curve, color='#d62728', linewidth=1.5, label='QuTiP occupation')
    ax2.plot(qutip_times, qutip_mag_curve, color='#1f77b4', linewidth=1.5, label='QuTiP magnetisation')
    ax1.scatter(time_data, result['occ'], color='#d62728', marker='o', edgecolors='black',
                linewidths=0.5, s=18, zorder=5, label='Cirq occupation (noisy, post-selected)')
    ax2.scatter(time_data, result['mag'], color='#1f77b4', marker='s', edgecolors='black',
                linewidths=0.5, s=18, zorder=5, label='Cirq magnetisation (noisy, post-selected)')

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
        f"Linear r=a*t (fine grid): a = {a:g}   |   RMS error = {result['rms_error']:.4f}   "
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
        f"({N_TIME_POINTS} t-points, {NUM_SHOTS} shots/point)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = os.path.join(LINEAR_DIR, f"linear_finegrid_a{a:g}.png")
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    return fname


#------------------------
# Fine-grid scan
#------------------------
print(f"\n=== Fine-grid scan: linear r = a*t over a = {A_VALUES} "
      f"({N_TIME_POINTS} t-points, {NUM_SHOTS} shots/point) ===")
results = {}
t_start = time.time()
for a in A_VALUES:
    result = run_schedule_sweep(a)
    results[a] = result
    fname = plot_run(result, a)
    np.savez(
        os.path.join(LINEAR_DIR, f"linear_finegrid_a{a:g}.npz"),
        time_data=time_data, occ=result['occ'], mag=result['mag'], err=result['err'],
        survival=result['survival'], steps=result['steps'], rms_error=result['rms_error'],
        a=a, qutip_time=qutip_times, qutip_occ=qutip_occ_curve, qutip_mag=qutip_mag_curve,
    )
    print(f"  a={a:g}: RMS error={result['rms_error']:.4f}, mean steps={np.mean(result['steps']):.1f}, "
          f"mean survival={np.mean(result['survival']):.1%}  ({time.time()-t_start:.0f}s elapsed) -> {fname}")

#------------------------
# Summary plot + parabolic refinement around the minimum
#------------------------
rms_errors = np.array([results[a]['rms_error'] for a in A_VALUES])
best_a = A_VALUES[int(np.argmin(rms_errors))]

# Parabolic (quadratic) fit through all points to interpolate a sub-grid-spacing optimum,
# since the true minimum need not land exactly on a tested grid point.
coeffs = np.polyfit(A_VALUES, rms_errors, 2)
a_fit_min = -coeffs[1] / (2 * coeffs[0]) if coeffs[0] > 0 else best_a

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(A_VALUES, rms_errors, marker='o', color='tab:green', label='measured RMS error')
a_dense = np.linspace(min(A_VALUES), max(A_VALUES), 200)
ax.plot(a_dense, np.polyval(coeffs, a_dense), color='tab:red', linestyle='--',
        label=f'parabolic fit (min at a={a_fit_min:.3f})')
ax.axvline(best_a, color='tab:gray', linestyle=':', label=f'best grid point a={best_a:g}')
ax.set_xlabel('a')
ax.set_ylabel('RMS combined error vs qutip')
ax.set_title(f'Fine-grid linear scan: error vs a\n({N_TIME_POINTS} t-points, {NUM_SHOTS} shots/point)', fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
summary_path = os.path.join(LINEAR_DIR, "summary_finegrid.png")
fig.savefig(summary_path, dpi=130)
plt.close(fig)

np.savez(
    os.path.join(LINEAR_DIR, "summary_finegrid.npz"),
    A_VALUES=np.array(A_VALUES), rms_errors=rms_errors, best_a=best_a, a_fit_min=a_fit_min,
    parabola_coeffs=coeffs,
)

print(f"\n=== FINE-GRID RESULT ===")
for a in A_VALUES:
    print(f"  a={a:g}: RMS={results[a]['rms_error']:.4f}, mean steps={np.mean(results[a]['steps']):.1f}, "
          f"mean survival={np.mean(results[a]['survival']):.1%}")
print(f"Best grid point: a = {best_a:g} (RMS = {results[best_a]['rms_error']:.4f})")
print(f"Parabolic-fit minimum: a ~= {a_fit_min:.3f}")
print(f"Saved {summary_path}")
