"""Neuron-chain simulation: qutip ground truth vs. a first-order Trotter circuit,
mapped and compiled onto a real Willow calibration snapshot, run noisy (qsimcirq,
shot-sampled) with post-selection on the unary constraint.

Clean counterpart to src/NuronSim.py -- same run path, current best-known config
(D-21's schedule, D-23's qubit embedding, D-4/D-19's post-selection), with the R&D
scaffolding (commented-out debug prints, superseded toggles) removed. See
src_clean/neuron_circuit.py's module docstring and this project's DECISIONS.md for
the full derivation history behind each constant.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src_clean/NuronSim.py
(GoogleQVM env -- needs a working qsimcirq install, D-14.)
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, tensor, sesolve, expect, fock
import cirq
import cirq_google
import qsimcirq

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    compute_observables_from_z_expectations,
    find_low_error_qubit_embedding,
    map_and_compile_for_willow,
    plot_qubit_embedding_overlay,
    build_trotter_circuit,
    recommended_trotter_steps,
    check_trotter_schedule_config,
    trotter_schedule_cap_message,
    sample_shots_with_postselection,
    annotate_trotter_step_segments,
    describe_trotter_run,
    TROTTER_SCHEDULE_T_CAP,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results/clean"

#------------------------
# INPUT
#------------------------

NumberOfBosonicModes = 1
NumberOfFockStates = 5

# Per-site displacement coefficients D_j (D-3): geometric decay breaks the site symmetry
# that would otherwise make every site fire in lockstep.
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]

spin_interaction_coefficient = 0.5
Time = 10
NumberOfTrotterSteps = 7
NumberOfTimeSteps = 50

# Per-point adaptive step count (recommended_trotter_steps(t), D-21) instead of one fixed
# NumberOfTrotterSteps for the whole sweep -- a fixed count is simultaneously too many
# steps (too much noise) at small t and too few (too much Trotter error) at large t.
UseAdaptiveTrotterSteps = True

# Shots per time point for the noisy Cirq sim (qsimcirq trajectory sampling, D-14).
# Runtime scales ~linearly with this; shot noise scales ~1/sqrt(it). 2000 is the tested
# default; use fewer (e.g. 200-500) while iterating.
NumberOfNoiseSamples = 2000

# False isolates circuit/Trotter-error behaviour from noise (exact, no sampling) --
# e.g. to check whether a step count actually tracks qutip once noise isn't dominant.
UseNoiseModel = True

# False plots the raw (unfiltered) shot average instead of the post-selected one, to see
# what post-selection on the unary constraint (D-4) is buying. Only affects the noisy
# path -- both estimates are always computed and saved either way.
UsePostSelection = True

# Plot the shots post-selection discarded (unary-leakage detected) as greyed-out points
# alongside the kept result. No-op when UseNoiseModel=False (nothing is ever discarded).
ShowPostSelectionRemovedPoints = True

# Also produce the willow_pink calibration grid (T1 per qubit, two-qubit CZ error per
# bond) with this run's chosen qubit embedding highlighted, alongside the results plot --
# a sanity check that find_low_error_qubit_embedding landed on a good device patch.
ShowQubitEmbeddingOverlay = True

run_tag, run_stage_title = describe_trotter_run(UseNoiseModel, UsePostSelection, UseAdaptiveTrotterSteps, NumberOfTrotterSteps)

#------------------------
# QUTIP SIM (ground truth)
#------------------------

# Calibrate G at L=1 against the reference site (site 0), then reuse across the chain.
SpinBosonInteractionCoefficent = find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, D_list[0])
print(f"G calibrated at L=1 (D={D_list[0]:.4g}, N={NumberOfFockStates}): G = {SpinBosonInteractionCoefficent:.4g}")

state_list = [tensor(basis(2, 0), fock(NumberOfFockStates, 0)) for _ in range(NumberOfBosonicModes)]
psi0 = tensor(state_list)
times = np.linspace(0, Time, Time * 10)

H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent)

result = sesolve(H, psi0, times)
states = result.states

exp_n = np.array([expect(n_list[i], states) for i in range(NumberOfBosonicModes)])
exp_sz = np.array([expect(sz_list[i], states) for i in range(NumberOfBosonicModes)])

print(f"Site 0 min <Z> in L={NumberOfBosonicModes} chain: {np.min(exp_sz[0]):.4f} (target ~ -1; L=1 calibration reached < -0.99)")

time_data = times

qutip_fig, axes = plt.subplots(
    nrows=1, ncols=NumberOfBosonicModes, figsize=(6.5 * NumberOfBosonicModes, 4)
)
if NumberOfBosonicModes == 1:
    axes = [axes]

color_boson = '#d62728'  # Red
color_spin = '#1f77b4'   # Blue

qutip_ax1_list = []
qutip_ax2_list = []

for mode_idx in range(NumberOfBosonicModes):
    ax1 = axes[mode_idx]
    boson_data = exp_n[mode_idx]
    spin_data = exp_sz[mode_idx]

    ax1.set_xlabel('Time', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_boson)
    if mode_idx == 0:
        ax1.set_ylabel('Average Bosonic Occupation number', color=color_boson, fontweight='bold')
    ax1.plot(time_data, boson_data, color=color_boson, linestyle='-', linewidth=1.5)

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color_spin)
    if mode_idx == NumberOfBosonicModes - 1:
        ax2.set_ylabel('Average Spin Magnetisation', color=color_spin, fontweight='bold')
    ax2.plot(time_data, spin_data, color=color_spin, linestyle='-', linewidth=1.5)

    if NumberOfBosonicModes > 1:
        ax1.set_title(f"Mode {mode_idx}", fontweight='bold', pad=10)

    # Fixed axis limits so Cirq datapoints (added below) never rescale the qutip curve.
    boson_max = np.max(boson_data)
    ax1.set_ylim(-0.10 * boson_max, 1.10 * boson_max)
    ax2.set_ylim(-1.10, 1.10)

    time_min, time_max = time_data.min(), time_data.max()
    time_pad = 0.05 * (time_max - time_min)
    ax1.set_xlim(time_min - time_pad, time_max + time_pad)
    ax1.set_autoscale_on(False)
    ax2.set_autoscale_on(False)

    qutip_ax1_list.append(ax1)
    qutip_ax2_list.append(ax2)

qutip_fig.suptitle(
    f"N = {NumberOfFockStates}, G = {SpinBosonInteractionCoefficent:.3g}, J={spin_interaction_coefficient}",
    fontweight='bold', fontsize=12, y=0.98,
)
qutip_fig.text(0.5, 0.90, run_stage_title, ha='center', fontweight='bold', fontsize=10.5)
qutip_fig.tight_layout(rect=[0, 0, 1, 0.87])

#------------------------
# CIRQ SIM — mapped onto Willow, with Willow's noise model
#------------------------

total_qubits = (NumberOfBosonicModes * NumberOfFockStates) + NumberOfBosonicModes

# --- Willow device, calibration and noise model (see src/QVMSetup.ipynb) ---
processor_id = "willow_pink"
willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
willow_noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(
    cirq_google.engine.load_device_noise_properties(processor_id)
)
willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]

# Pick a connected, low-error, coherence-aware qubit embedding (D-23) once -- it doesn't
# depend on dt, only on the circuit's fixed qubit count (D-1).
willow_qubit_chain = find_low_error_qubit_embedding(willow_device, willow_calibration, NumberOfFockStates, NumberOfBosonicModes)
print(f"Mapped {total_qubits} logical qubits onto Willow ({processor_id}):")
print("  " + " - ".join(str(q) for q in willow_qubit_chain))

if ShowQubitEmbeddingOverlay:
    embedding_fig = plot_qubit_embedding_overlay(willow_calibration, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes)
    embedding_fig.savefig(f"{RESULTS_DIR}/NuronSim_{run_tag}_embedding.png", dpi=150)
    print(f"Saved {RESULTS_DIR}/NuronSim_{run_tag}_embedding.png")

# qsimcirq trajectory-samples the noise (Monte Carlo unraveling + measurement), matching
# real hardware execution -- needs the GoogleQVM conda env (D-14).
noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model if UseNoiseModel else None)
z_observables = [cirq.Z(q) for q in willow_qubit_chain]  # boson qubits per mode, then spin qubits
if not UseNoiseModel:
    print("UseNoiseModel = False — running the Willow-mapped circuit ideally (no noise).")

if UseAdaptiveTrotterSteps:
    check_trotter_schedule_config(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient)
    print("UseAdaptiveTrotterSteps = True — NumberOfTrotterSteps per point from recommended_trotter_steps(t).")
    # Informational only: past TROTTER_SCHEDULE_T_CAP, the required step count isn't
    # affordable under the noise budget (see trotter_schedule_cap_message), but the sweep
    # still runs to the full requested Time rather than being truncated.
    cap_message = trotter_schedule_cap_message(Time)
    if cap_message is not None:
        print(cap_message)

# *_post is the primary result (post-selected on the unary constraint, D-4); *_raw is the
# uncorrected marginal average; *_removed is the average over exactly the discarded shots.
bosonic_occupation_results = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
spin_magnetization_results = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
bosonic_occupation_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
spin_magnetization_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
bosonic_occupation_removed = np.full((NumberOfTimeSteps, NumberOfBosonicModes), np.nan)
spin_magnetization_removed = np.full((NumberOfTimeSteps, NumberOfBosonicModes), np.nan)
postselection_survival_rate = np.full(NumberOfTimeSteps, np.nan)
trotter_steps_used = np.zeros(NumberOfTimeSteps, dtype=int)

time_data = np.linspace(0.1, Time, NumberOfTimeSteps)

# =====================================================================
# EXECUTION LOOP: total time t varies; step count is either fixed or adaptive per point
# =====================================================================
for time_idx, t in enumerate(time_data):
    steps_t = recommended_trotter_steps(t) if UseAdaptiveTrotterSteps else NumberOfTrotterSteps
    trotter_steps_used[time_idx] = steps_t
    dt = t / steps_t

    full_circuit, all_qubits = build_trotter_circuit(
        D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
        dt, steps_t, SpinBosonInteractionCoefficent
    )
    compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
    if time_idx == 0:
        willow_device.validate_circuit(compiled_circuit)
        print("Compiled circuit validated against the Willow device (native gateset + connectivity).")

    if UseNoiseModel:
        shots = sample_shots_with_postselection(
            noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes,
            NumberOfNoiseSamples
        )
        bosonic_occupation_results[time_idx] = shots['occ_post'] if UsePostSelection else shots['occ_raw']
        spin_magnetization_results[time_idx] = shots['mag_post'] if UsePostSelection else shots['mag_raw']
        bosonic_occupation_raw[time_idx] = shots['occ_raw']
        spin_magnetization_raw[time_idx] = shots['mag_raw']
        bosonic_occupation_removed[time_idx] = shots['occ_removed']
        spin_magnetization_removed[time_idx] = shots['mag_removed']
        postselection_survival_rate[time_idx] = shots['survival_rate']
    else:
        z_expectations = [z.real for z in noisy_sim.simulate_expectation_values(compiled_circuit, observables=z_observables)]
        occ, mag = compute_observables_from_z_expectations(z_expectations, NumberOfFockStates, NumberOfBosonicModes)
        bosonic_occupation_results[time_idx] = occ
        spin_magnetization_results[time_idx] = mag
        bosonic_occupation_raw[time_idx] = occ
        spin_magnetization_raw[time_idx] = mag
        postselection_survival_rate[time_idx] = 1.0

# =====================================================================
# PLOTTING: overlay Cirq datapoints on the QuTiP figure/axes from above
# =====================================================================
_cirq_overlay_artists = []

for mode_idx in range(NumberOfBosonicModes):
    ax1 = qutip_ax1_list[mode_idx]
    ax2 = qutip_ax2_list[mode_idx]

    boson_data = bosonic_occupation_results[:, mode_idx]
    spin_data = spin_magnetization_results[:, mode_idx]

    if ShowPostSelectionRemovedPoints and UseNoiseModel and UsePostSelection:
        removed_boson_scatter = ax1.scatter(time_data, bosonic_occupation_removed[:, mode_idx], color='lightgray',
                                             marker='o', edgecolors='dimgray', linewidths=0.5, alpha=0.6, zorder=3)
        removed_spin_scatter = ax2.scatter(time_data, spin_magnetization_removed[:, mode_idx], color='lightgray',
                                            marker='s', edgecolors='dimgray', linewidths=0.5, alpha=0.6, zorder=3)
        _cirq_overlay_artists.extend([removed_boson_scatter, removed_spin_scatter])

    boson_scatter = ax1.scatter(time_data, boson_data, color=color_boson, marker='o', edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)
    spin_scatter = ax2.scatter(time_data, spin_data, color=color_spin, marker='s', edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5)
    _cirq_overlay_artists.extend([boson_scatter, spin_scatter])

    if UseAdaptiveTrotterSteps and TROTTER_SCHEDULE_T_CAP < time_data.max():
        cap_line = ax1.axvline(TROTTER_SCHEDULE_T_CAP, color='gray', linestyle=':', linewidth=1, zorder=1)
        _cirq_overlay_artists.append(cap_line)

    _cirq_overlay_artists.extend(annotate_trotter_step_segments(ax1, time_data, trotter_steps_used))

qutip_fig.tight_layout(rect=[0, 0, 1, 0.87])

if UseNoiseModel:
    applied_note = "applied — plotted result is post-selected" if UsePostSelection else "NOT applied — plotted result is the raw, unfiltered average"
    print(f"\nPost-selection survival rate (unary one-hot constraint, D-4; {applied_note}): "
          f"mean={np.nanmean(postselection_survival_rate):.3%}, "
          f"min={np.nanmin(postselection_survival_rate):.3%}, "
          f"max={np.nanmax(postselection_survival_rate):.3%} over the sweep.")

qutip_fig.savefig(f"{RESULTS_DIR}/NuronSim_{run_tag}.png", dpi=130)
np.savez(
    f"{RESULTS_DIR}/NuronSim_{run_tag}.npz",
    time_data=time_data, trotter_steps_used=trotter_steps_used,
    qutip_time=times, qutip_occ=exp_n, qutip_mag=exp_sz,
    bosonic_occupation_post=bosonic_occupation_results, spin_magnetization_post=spin_magnetization_results,
    bosonic_occupation_raw=bosonic_occupation_raw, spin_magnetization_raw=spin_magnetization_raw,
    bosonic_occupation_removed=bosonic_occupation_removed, spin_magnetization_removed=spin_magnetization_removed,
    postselection_survival_rate=postselection_survival_rate,
    D_list=np.array(D_list), NumberOfFockStates=NumberOfFockStates, NumberOfBosonicModes=NumberOfBosonicModes,
    spin_interaction_coefficient=spin_interaction_coefficient, SpinBosonInteractionCoefficent=SpinBosonInteractionCoefficent,
    UseNoiseModel=UseNoiseModel, UseAdaptiveTrotterSteps=UseAdaptiveTrotterSteps, UsePostSelection=UsePostSelection, Time=Time,
)
print(f"Saved {RESULTS_DIR}/NuronSim_{run_tag}.png and .npz")

plt.show()
