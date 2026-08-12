"""Second-order (Strang/symmetric) Trotterization variant of NuronSim.py.

Only difference from NuronSim.py: circuits are built with
build_second_order_trotter_circuit instead of build_trotter_circuit (see its docstring
in neuron_circuit.py for the merged-boundary-layer construction), and the adaptive step
schedule is second-order-specific (recommended_trotter_steps_2nd_order, D-22) rather than
reused from first order. See notes/second-order-trotter.md and
notes/first-vs-second-order-trotter-comparison.md.

Uses recommended_trotter_steps_2nd_order(t) -- its own schedule, fit directly against
noisy, post-selected observable error on THIS circuit (D-22,
src/second_order_trotter_linear_scan.py), superseding the earlier choice to reuse
first-order's schedule for an apples-to-apples matched-step comparison (D-20). D-22 found
the optimized coefficient (a=2.0) comes out close to first order's own (a=2.25) despite
second order's faster r^-2 convergence -- its ~1.6x higher per-step gate cost roughly
cancels the advantage -- and that even with its own fair schedule, second order still
loses to first order on RMS error (0.166 vs 0.157), though closely and with second order
actually ahead at early-to-mid t. See D-22 and
results/first_vs_second_order_optimized_trotter_comparison.png.
"""
import matplotlib.pyplot as plt
import math
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
    build_second_order_trotter_circuit,
    recommended_trotter_steps_2nd_order,
    check_trotter_schedule_config,
    trotter_schedule_cap_message_2nd_order,
    sample_shots_with_postselection,
    annotate_trotter_step_segments,
    TROTTER_SCHEDULE_T_CAP_2ND_ORDER,
)

#------------------------
# INPUT
#------------------------

NumberOfBosonicModes=1
NumberOfFockStates=5

# Per-site displacement coefficients D_j (D-3) — see Classical Simulation.py. With
# NumberOfBosonicModes=1 this is just [1.0], kept as a list so QutipHamiltonian and the
# Cirq circuit builder share the same per-site convention as the rest of the project.
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]

spin_interaction_coefficient=0.5
Time = 20
NumberOfTrotterSteps = 7
NumberOfTimeSteps = 100

# Set True to use recommended_trotter_steps_2nd_order(t) instead of a single fixed
# NumberOfTrotterSteps for every point in the sweep. Fitted directly for THIS (second-
# order) circuit against noisy, post-selected observable error (D-22).
UseAdaptiveTrotterSteps = True

# Shots per time point for the noisy Cirq sim's expectation-value estimate (qsimcirq
# trajectory sampling — see D-14). Runtime scales ~linearly with this; shot noise scales
# ~1/sqrt(it). 2000 matched the exact density-matrix result closely in testing. Use fewer
# (e.g. 200-500) when testing changes, especially at high NumberOfTrotterSteps.
NumberOfNoiseSamples = 2000

# Set False to run the Cirq sim through Willow's mapping/compilation with the noise model
# switched off (ideal, ~single-shot exact-ish result — ISOLATES circuit/Trotter-error
# behaviour from noise, e.g. to check whether a large NumberOfTrotterSteps actually tracks
# qutip once noise isn't the dominant effect, see D-15).
UseNoiseModel = True

# Set False to plot the raw (unfiltered) shot average instead of the post-selected one --
# lets you see what post-selection on the unary constraint (D-4) is actually buying you.
# Only affects the noisy path (UseNoiseModel=True); both estimates are always computed
# and saved regardless (occ/mag _post vs _raw in the .npz). Noiseless is unaffected --
# survival rate is exactly 1.0 there (D-18), so post-selected and raw are identical.
UsePostSelection = True

#------------------------
# QUTIP SIM
#------------------------

# Calibrate G at L=1 against the reference site (site 0), then reuse across the chain.
SpinBosonInteractionCoefficent = find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, D_list[0])
print(f"G calibrated at L=1 (D={D_list[0]:.4g}, N={NumberOfFockStates}): G = {SpinBosonInteractionCoefficent:.4g}")

state_list = [tensor(basis(2,0),fock(NumberOfFockStates,0)) for _ in range(NumberOfBosonicModes)]       # Tensor product of spin and boson vectors in ground state at each lattice point in a list
psi0 = tensor(state_list)                                           # Tensor product of all entries in the list
times = np.linspace(0, Time, Time*10)

H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent)

result = sesolve(H, psi0, times)
states = result.states

# Calculate expectation values
exp_n = np.array([expect(n_list[i], states) for i in range(NumberOfBosonicModes)])
exp_sz = np.array([expect(sz_list[i], states) for i in range(NumberOfBosonicModes)])

# Transfer check (open question (a) in PROJECT.md): does the L=1-calibrated G still
# flip site 0 once other sites/J are present?
print(f"Site 0 min <Z> in L={NumberOfBosonicModes} chain: {np.min(exp_sz[0]):.4f} (target ~ -1; L=1 calibration reached < -0.99)")

time_data = times

qutip_fig, axes = plt.subplots(
    nrows=1, ncols=NumberOfBosonicModes, figsize=(5 * NumberOfBosonicModes, 4)
)

if NumberOfBosonicModes == 1:
    axes = [axes]

color_boson = '#d62728'  # Red
color_spin = '#1f77b4'   # Blue

# Kept so the Cirq simulation cell below can plot its datapoints on these same axes
qutip_ax1_list = []
qutip_ax2_list = []

for mode_idx in range(NumberOfBosonicModes):
    ax1 = axes[mode_idx]

    boson_data = exp_n[mode_idx]
    spin_data = exp_sz[mode_idx]

    # Left Axis: Bosonic Occupation
    ax1.set_xlabel('Time', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_boson)
    if mode_idx == 0:
        ax1.set_ylabel(
            'Average Bosonic Occupation number', color=color_boson, fontweight='bold'
        )
    ax1.plot(time_data, boson_data, color=color_boson, linestyle='-', linewidth=1.5, label='QuTiP – Bosonic occupation')

    # Right Axis: Spin Magnetization
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor=color_spin)
    if mode_idx == NumberOfBosonicModes - 1:
        ax2.set_ylabel('Average Spin Magnetisation', color=color_spin, fontweight='bold')

    ax2.plot(time_data, spin_data, color=color_spin, linestyle='-', linewidth=1.5, label='QuTiP – Spin magnetisation')

    if NumberOfBosonicModes > 1:
        ax1.set_title(f"Mode {mode_idx}", fontweight='bold', pad=10)

    # Fix axis limits to the observed data range (with a little padding) so the
    # bosonic occupation curve fills the box, while still being fixed so adding
    # Cirq datapoints later (with a different NumberOfTrotterSteps) never rescales them
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

NTrot_label = "adaptive (recommended_trotter_steps_2nd_order)" if UseAdaptiveTrotterSteps else str(NumberOfTrotterSteps)
plt.suptitle(f"[2nd-order Trotter] N = {NumberOfFockStates}, G = {SpinBosonInteractionCoefficent:.3g}, J={spin_interaction_coefficient} , NTrot= {NTrot_label}", fontweight='bold', fontsize=12, y=0.98)

qutip_fig.tight_layout(rect=[0, 0, 1, 0.93])

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

# Pick a connected, low-error qubit chain matching our L x (N+1) linear topology
# once — it doesn't depend on dt, only on the circuit's fixed qubit count (D-1).
willow_qubit_chain = find_low_error_qubit_embedding(willow_device, willow_calibration, NumberOfFockStates, NumberOfBosonicModes)
print(f"Mapped {total_qubits} logical qubits onto Willow ({processor_id}):")
print("  " + " - ".join(str(q) for q in willow_qubit_chain))

# qsimcirq trajectory-samples the noise (Monte Carlo unraveling + measurement), the same
# way real hardware execution works — this needs the GoogleQVM conda env (Python 3.12),
# where qsimcirq has a working wheel; the project's default macOS .venv doesn't have one
# (D-14, supersedes D-12's DensityMatrixSimulator fallback).
noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model if UseNoiseModel else None)
z_observables = [cirq.Z(q) for q in willow_qubit_chain]  # boson qubits per mode, then spin qubits
if not UseNoiseModel:
    print("UseNoiseModel = False — running the Willow-mapped circuit ideally (no noise).")

if UseAdaptiveTrotterSteps:
    check_trotter_schedule_config(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient)
    print("UseAdaptiveTrotterSteps = True — NumberOfTrotterSteps per point from "
          "recommended_trotter_steps_2nd_order(t) (D-22, fitted directly for this second-order circuit).")
    # D-22: cap the sweep at THIS schedule's own noise-budget limit (own r_max, since
    # second-order steps cost more two-qubit gates each) rather than reporting points
    # that would need more Trotter depth than the noise budget affords.
    cap_message = trotter_schedule_cap_message_2nd_order(Time)
    if cap_message is not None:
        print(cap_message)
        print(f"Capping sweep at Time={TROTTER_SCHEDULE_T_CAP_2ND_ORDER:.2f} (was {Time}).")
        Time = TROTTER_SCHEDULE_T_CAP_2ND_ORDER

# Arrays to store measurements. *_post is the primary result (post-selected on the unary
# constraint, D-4); *_raw is the uncorrected marginal estimate over all shots, kept for
# comparison. survival_rate is the fraction of shots that passed post-selection.
bosonic_occupation_results = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
spin_magnetization_results = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
bosonic_occupation_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
spin_magnetization_raw = np.zeros((NumberOfTimeSteps, NumberOfBosonicModes))
postselection_survival_rate = np.full(NumberOfTimeSteps, np.nan)
trotter_steps_used = np.zeros(NumberOfTimeSteps, dtype=int)

# Time points to evaluate
time_data = np.linspace(0.1, Time, NumberOfTimeSteps)

# =====================================================================
# EXECUTION LOOP: total time t varies; step count is either fixed or adaptive per point
# =====================================================================
for time_idx, t in enumerate(time_data):
    steps_t = recommended_trotter_steps_2nd_order(t) if UseAdaptiveTrotterSteps else NumberOfTrotterSteps
    trotter_steps_used[time_idx] = steps_t
    dt = t / steps_t

    # 1-3. Vacuum state prep + steps_t repeats of the second-order (Strang) dt-step
    full_circuit, all_qubits = build_second_order_trotter_circuit(
        D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
        dt, steps_t, SpinBosonInteractionCoefficent
    )

    # 4. Map onto the chosen Willow qubit patch and compile to its native gateset
    compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
    if time_idx == 0:
        willow_device.validate_circuit(compiled_circuit)
        print("Compiled circuit validated against the Willow device (native gateset + connectivity).")

    # 5. Observables at time t. Noisy -> shot-sampled (hardware-realistic, D-14) with
    # post-selection on the unary constraint (D-4): shots where any boson column isn't
    # exactly one-hot are a detectable noise signature and are discarded before averaging.
    # Noiseless -> exact, no sampling overhead (fast, for isolating circuit/Trotter-error
    # behaviour from noise, D-15; post-selection is moot there since the unary subspace is
    # conserved exactly, D-18).
    if UseNoiseModel:
        shots = sample_shots_with_postselection(
            noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes,
            NumberOfNoiseSamples
        )
        bosonic_occupation_results[time_idx] = shots['occ_post'] if UsePostSelection else shots['occ_raw']
        spin_magnetization_results[time_idx] = shots['mag_post'] if UsePostSelection else shots['mag_raw']
        bosonic_occupation_raw[time_idx] = shots['occ_raw']
        spin_magnetization_raw[time_idx] = shots['mag_raw']
        postselection_survival_rate[time_idx] = shots['survival_rate']
    else:
        z_expectations = [z.real for z in noisy_sim.simulate_expectation_values(compiled_circuit, observables=z_observables)]
        occ, mag = compute_observables_from_z_expectations(
            z_expectations, NumberOfFockStates, NumberOfBosonicModes
        )
        bosonic_occupation_results[time_idx] = occ
        spin_magnetization_results[time_idx] = mag
        bosonic_occupation_raw[time_idx] = occ
        spin_magnetization_raw[time_idx] = mag
        postselection_survival_rate[time_idx] = 1.0

# =====================================================================
# PLOTTING: overlay Cirq datapoints on the QuTiP figure/axes from the cell above
# =====================================================================
if '_cirq_overlay_artists' in globals():
    for artist in _cirq_overlay_artists:
        artist.remove()
_cirq_overlay_artists = []

for mode_idx in range(NumberOfBosonicModes):
    ax1 = qutip_ax1_list[mode_idx]
    ax2 = qutip_ax2_list[mode_idx]

    boson_data = bosonic_occupation_results[:, mode_idx]
    spin_data = spin_magnetization_results[:, mode_idx]

    if UseNoiseModel:
        postselected_label = 'Cirq 2nd-order (Willow, noisy, post-selected)' if UsePostSelection else 'Cirq 2nd-order (Willow, noisy, NO post-selection)'
    else:
        postselected_label = 'Cirq 2nd-order (Willow, noiseless)'
    boson_scatter = ax1.scatter(time_data, boson_data, color=color_boson, marker='o', edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{postselected_label} – Bosonic occupation')
    spin_scatter = ax2.scatter(time_data, spin_data, color=color_spin, marker='s', edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{postselected_label} – Spin magnetisation')
    _cirq_overlay_artists.extend([boson_scatter, spin_scatter])

    if UseAdaptiveTrotterSteps and TROTTER_SCHEDULE_T_CAP_2ND_ORDER < time_data.max():
        cap_line = ax1.axvline(TROTTER_SCHEDULE_T_CAP_2ND_ORDER, color='gray', linestyle=':', linewidth=1, zorder=1,
                                label=f'Trotter schedule cap (t={TROTTER_SCHEDULE_T_CAP_2ND_ORDER:.2f})')
        _cirq_overlay_artists.append(cap_line)

    # Mark which NumberOfTrotterSteps value applies to each cluster of points (varies
    # with t under UseAdaptiveTrotterSteps) with vertical dividers + sparse "r=N" labels.
    _cirq_overlay_artists.extend(annotate_trotter_step_segments(ax1, time_data, trotter_steps_used))

qutip_fig.tight_layout(rect=[0, 0, 1, 0.93])

if UseNoiseModel:
    applied_note = "applied — plotted result is post-selected" if UsePostSelection else "NOT applied — plotted result is the raw, unfiltered average"
    print(f"\nPost-selection survival rate (unary one-hot constraint, D-4; {applied_note}): "
          f"mean={np.nanmean(postselection_survival_rate):.3%}, "
          f"min={np.nanmin(postselection_survival_rate):.3%}, "
          f"max={np.nanmax(postselection_survival_rate):.3%} over the sweep.")

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
if UseNoiseModel:
    suffix = "noisy" if UsePostSelection else "noisy_nopostselect"
else:
    suffix = "noiseless"
qutip_fig.savefig(f"{RESULTS_DIR}/NeuronSim2ndOrderTrotter_{suffix}.png", dpi=130)
np.savez(
    f"{RESULTS_DIR}/NeuronSim2ndOrderTrotter_{suffix}.npz",
    time_data=time_data, trotter_steps_used=trotter_steps_used,
    qutip_time=times, qutip_occ=exp_n, qutip_mag=exp_sz,
    bosonic_occupation_post=bosonic_occupation_results, spin_magnetization_post=spin_magnetization_results,
    bosonic_occupation_raw=bosonic_occupation_raw, spin_magnetization_raw=spin_magnetization_raw,
    postselection_survival_rate=postselection_survival_rate,
    D_list=np.array(D_list), NumberOfFockStates=NumberOfFockStates, NumberOfBosonicModes=NumberOfBosonicModes,
    spin_interaction_coefficient=spin_interaction_coefficient, SpinBosonInteractionCoefficent=SpinBosonInteractionCoefficent,
    UseNoiseModel=UseNoiseModel, UseAdaptiveTrotterSteps=UseAdaptiveTrotterSteps, UsePostSelection=UsePostSelection, Time=Time,
)
print(f"Saved {RESULTS_DIR}/NeuronSim2ndOrderTrotter_{suffix}.png and .npz")

plt.show()
