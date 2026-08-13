"""One-off: re-run results/VMBenchmark/N7_a2.5_adaptive_noisy_postselect (requested by
Drew, 2026-08-13) -- the existing file is a pre-D-23 leftover (fixed a=2.5, sweep capped
to t~8.4 by the old noise-budget check run against an unoptimized embedding).

This re-run uses the current find_low_error_qubit_embedding (percentile-scoring + T1
floor, D-23), NumberOfNoiseSamples=3000 (up from vm_benchmark.py's default 2000), and
sweeps the full requested Time=15 with the noise-budget cap disabled (informational
warning only, matching NuronSim.py's cap being turned off the same session) rather than
truncated -- so this intentionally reports points past where the schedule is expected to
be Trotter-converged; treat the tail of the sweep accordingly.

Self-contained rather than importing from vm_benchmark.py: that module runs all three of
its (multi-hour) parts as top-level code on import (no __main__ guard), so importing from
it here would trigger the whole benchmark suite. The small pieces needed
(qutip_ground_truth, simulate_sweep, plot_and_save, Willow device/calibration setup) are
copied directly from vm_benchmark.py instead.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/rerun_N7_a2.5.py
"""
import os
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
    build_trotter_circuit,
    map_and_compile_for_willow,
    sample_shots_with_postselection,
    annotate_trotter_step_segments,
    two_qubit_error_rate_for_chain,
    two_qubit_gates_per_trotter_step,
    estimate_max_affordable_trotter_steps,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "VMBenchmark")

color_boson = '#d62728'
color_spin = '#1f77b4'

N = 7
L = 1
D_list = [1.0]
J = 0.5  # inert at L=1 (H_spin's sum is empty), matching vm_benchmark.py Part 3's convention
A = 2.5
TIME = 15.0
N_TIME_POINTS = 50
N_SHOTS = 3000

print(f"N={N}, L={L}, a={A}, Time={TIME}, shots={N_SHOTS}, noise ON, post-selection ON")

print("Setting up Willow device, calibration, and noise model...")
processor_id = "willow_pink"
willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
willow_noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(
    cirq_google.engine.load_device_noise_properties(processor_id)
)
willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]


def qutip_ground_truth(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient,
                        SpinBosonInteractionCoefficent, Time):
    state_list = [tensor(basis(2, 0), fock(NumberOfFockStates, 0)) for _ in range(NumberOfBosonicModes)]
    psi0 = tensor(state_list)
    times = np.linspace(0, Time, int(Time * 10))
    H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list,
                                           spin_interaction_coefficient, SpinBosonInteractionCoefficent)
    result = sesolve(H, psi0, times)
    exp_n = np.array([expect(n_list[i], result.states) for i in range(NumberOfBosonicModes)])
    exp_sz = np.array([expect(sz_list[i], result.states) for i in range(NumberOfBosonicModes)])
    return times, exp_n, exp_sz


def simulate_sweep(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                    SpinBosonInteractionCoefficent, willow_qubit_chain,
                    Time, NumberOfTimeSteps, NumberOfNoiseSamples, UseNoiseModel, step_count_fn):
    noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model if UseNoiseModel else None)

    time_data = np.linspace(0.1, Time, NumberOfTimeSteps)
    occ_post = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
    mag_post = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
    occ_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
    mag_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
    survival = np.full(NumberOfTimeSteps, np.nan)
    steps_used = np.zeros(NumberOfTimeSteps, dtype=int)

    for i, t in enumerate(time_data):
        r = max(int(step_count_fn(t)), 1)
        steps_used[i] = r
        dt = t / r
        full_circuit, all_qubits = build_trotter_circuit(
            D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
            dt, r, SpinBosonInteractionCoefficent
        )
        compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
        if i == 0:
            willow_device.validate_circuit(compiled_circuit)

        shots = sample_shots_with_postselection(
            noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes,
            NumberOfNoiseSamples
        )
        occ_post[i], mag_post[i] = shots['occ_post'], shots['mag_post']
        occ_raw[i], mag_raw[i] = shots['occ_raw'], shots['mag_raw']
        survival[i] = shots['survival_rate']

        if (i + 1) % 10 == 0:
            print(f"  t={t:.2f} ({i+1}/{NumberOfTimeSteps}), r={r}")

    return dict(time_data=time_data, occ_post=occ_post, mag_post=mag_post, occ_raw=occ_raw, mag_raw=mag_raw,
                survival=survival, steps=steps_used)


def plot_and_save(qutip_times, qutip_occ, qutip_mag, sweep, title, output_path, cap_t=None):
    boson = sweep['occ_post']
    spin = sweep['mag_post']

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4))
    ax1.plot(qutip_times, qutip_occ[0], color=color_boson, linewidth=1.5)
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_boson)
    ax1.set_ylabel('Average Bosonic Occupation number', color=color_boson, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(qutip_times, qutip_mag[0], color=color_spin, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_spin)
    ax2.set_ylabel('Average Spin Magnetisation', color=color_spin, fontweight='bold')

    ax1.scatter(sweep['time_data'], boson[:, 0], color=color_boson, marker='o',
                edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)
    ax2.scatter(sweep['time_data'], spin[:, 0], color=color_spin, marker='s',
                edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)

    boson_max = max(np.max(qutip_occ[0]), 1e-9)
    ax1.set_ylim(-0.10 * boson_max, 1.10 * boson_max)
    ax2.set_ylim(-1.10, 1.10)

    if cap_t is not None and cap_t < sweep['time_data'].max():
        ax1.axvline(cap_t, color='gray', linestyle=':', linewidth=1, zorder=1)
    annotate_trotter_step_segments(ax1, sweep['time_data'], sweep['steps'])

    title_parts = title.split("  |  ", 1)
    fig.suptitle(title_parts[0], fontweight='bold', fontsize=11, y=0.98)
    if len(title_parts) > 1:
        fig.text(0.5, 0.90, title_parts[1], ha='center', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(output_path + ".png", dpi=130)
    np.savez(output_path + ".npz", qutip_time=qutip_times, qutip_occ=qutip_occ, qutip_mag=qutip_mag,
             time_data=sweep['time_data'], occ_post=sweep['occ_post'], mag_post=sweep['mag_post'],
             occ_raw=sweep['occ_raw'], mag_raw=sweep['mag_raw'], survival=sweep['survival'], steps=sweep['steps'])
    plt.close(fig)
    print(f"    survival: mean={np.nanmean(sweep['survival']):.1%}, "
          f"min={np.nanmin(sweep['survival']):.1%}, max={np.nanmax(sweep['survival']):.1%}")
    print(f"    Saved {output_path}.png")


G = find_optimal_SpinBosonInteractionCoefficent(N, D_list[0])
print(f"G = {G:.4g}")

qtimes, qocc, qmag = qutip_ground_truth(L, N, D_list, J, G, TIME)

chain = find_low_error_qubit_embedding(willow_device, willow_calibration, N, L)
print(f"Mapped {L * N + L} qubits onto Willow: " + " - ".join(str(q) for q in chain))

gates_per_step = two_qubit_gates_per_trotter_step(D_list, J, N, L, G, chain, willow_target_gateset)
mean_two_qubit_error = two_qubit_error_rate_for_chain(willow_calibration, chain)
r_max = estimate_max_affordable_trotter_steps(gates_per_step, mean_two_qubit_error, 0.5)
t_cap = r_max / A if A > 0 else TIME
if TIME > t_cap:
    print(f"WARNING: requested time {TIME} exceeds this config's noise-budget cap "
          f"(t_cap={t_cap:.2f}, r_max={r_max}) -- points beyond t_cap are not expected to be "
          f"Trotter-converged at any affordable step count. NOT truncating (Drew's request) -- "
          f"running the full sweep to t={TIME} anyway.")

t0 = time.time()
sweep = simulate_sweep(D_list, J, N, L, G, chain, TIME, N_TIME_POINTS, N_SHOTS,
                        UseNoiseModel=True, step_count_fn=lambda t, a=A: round(a * t))
plot_and_save(qtimes, qocc, qmag, sweep,
              title=f"N={N}, G={G:.3g}  |  Fixed schedule (a={A:g}), noise ON, post-selection ON, uncapped to t={TIME:g}",
              output_path=os.path.join(OUTPUT_DIR, "N7_a2.5_adaptive_noisy_postselect"), cap_t=t_cap)
print(f"({time.time()-t0:.0f}s elapsed)")
