"""Analysis: optimal NumberOfTrotterSteps per time point for NuronSim.py's Cirq sim.

Total error against the qutip ground truth has two competing contributions:
  - Trotter error: shrinks as NumberOfTrotterSteps grows (finer discretization of a
    fixed total time t).
  - Gate/noise error: grows with NumberOfTrotterSteps, because NuronSim.py's sweep is
    "fixed step count per point, varying total time" (dt = t / steps) -- step count sets
    circuit depth directly, independent of t (D-15).

Since Trotter error at fixed steps grows with t (coarser dt for the same step count) but
noise error at fixed steps does not depend on t, the step count that minimizes total error
is expected to grow with t -- exactly the trade-off D-15 flagged as still open. This script
measures it directly (grid search over exact expectation values, not sampled -- avoids
shot noise complicating the search) rather than assuming a scaling law.

Finding worth flagging up front (see notes/optimal-trotter-steps.md): the raw per-t argmin
is jagged, not a clean monotonic curve. That's not noise in the statistical sense (these are
exact expectation values) -- it's because the system's own dynamics are oscillatory, so a
fixed low step count beats/aliases against the true frequency rather than producing error
that grows smoothly with t. A window-smoothed version of the error surface is used to derive
a practical schedule (see "Smoothing" section below).

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/optimal_trotter_steps.py
(needs the GoogleQVM conda env -- qsimcirq/cirq_google, see D-14 in DECISIONS.md)
Re-running loads the cached grid from results/optimal_trotter_steps_grid.npz if the
config below is unchanged, instead of repeating the ~5 minute grid search.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sesolve, expect, fock
import cirq
import cirq_google

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    compute_observables_from_z_expectations,
    find_low_error_qubit_chain,
    map_and_compile_for_willow,
    build_trotter_circuit,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
GRID_CACHE = os.path.join(RESULTS_DIR, "optimal_trotter_steps_grid.npz")

#------------------------
# CONFIG -- matches NuronSim.py's current defaults
#------------------------
NumberOfBosonicModes = 1
NumberOfFockStates = 5
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]
spin_interaction_coefficient = 0.5
Time = 20
total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes

# Grid to search. t_grid mirrors NuronSim.py's sweep range; steps_grid is denser at low
# step counts (where the interesting trade-off sits, D-15) and sparser at high step
# counts (already known to be noise-dominated -- NTROT=20-100 in D-15/D-12 testing).
# 7 is included explicitly so the current NuronSim.py default has a direct baseline.
t_grid = np.linspace(0.5, Time, 14)
steps_grid = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40]
BASELINE_STEPS = 7

#------------------------
# Ground truth (qutip)
#------------------------
print("Calibrating G and solving the qutip ground truth...")
SpinBosonInteractionCoefficent = find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, D_list[0])
print(f"G = {SpinBosonInteractionCoefficent:.4g}")

state_list = [tensor(basis(2, 0), fock(NumberOfFockStates, 0)) for _ in range(NumberOfBosonicModes)]
psi0 = tensor(state_list)
qutip_times = np.linspace(0, Time, Time * 20)
H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent)
qutip_result = sesolve(H, psi0, qutip_times)
qutip_occ_curve = expect(n_list[0], qutip_result.states)
qutip_mag_curve = expect(sz_list[0], qutip_result.states)


def qutip_ground_truth(t):
    occ = np.interp(t, qutip_times, qutip_occ_curve)
    mag = np.interp(t, qutip_times, qutip_mag_curve)
    return occ, mag


def combined_error(occ, mag, t):
    """Normalized combined error against qutip at time t: boson occupancy normalized
    by its range [0, N-1], spin magnetization by its range [-1, 1], combined in
    quadrature so neither observable dominates just because of its natural scale."""
    qutip_occ, qutip_mag = qutip_ground_truth(t)
    boson_err = (occ - qutip_occ) / (NumberOfFockStates - 1)
    spin_err = (mag - qutip_mag) / 2.0
    return float(np.sqrt(boson_err**2 + spin_err**2))


#------------------------
# Grid search (cached -- this is the expensive part, ~5 min)
#------------------------
cache_valid = False
if os.path.exists(GRID_CACHE):
    cached = np.load(GRID_CACHE)
    if (np.array_equal(cached["t_grid"], t_grid) and
            np.array_equal(cached["steps_grid"], np.array(steps_grid))):
        noisy_error = cached["noisy_error"]
        noiseless_error = cached["noiseless_error"]
        cache_valid = True
        print(f"Loaded cached grid from {GRID_CACHE}")

if not cache_valid:
    processor_id = "willow_pink"
    willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
    willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
    willow_noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(
        cirq_google.engine.load_device_noise_properties(processor_id)
    )
    willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]
    willow_qubit_chain = find_low_error_qubit_chain(willow_device, willow_calibration, total_qubits)
    z_observables = [cirq.Z(q) for q in willow_qubit_chain]

    # Exact (not sampled) simulators -- isolates the t/steps trade-off from shot noise.
    # complex128 (not the cirq default complex64): at the higher end of steps_grid the
    # accumulated rounding error in the default precision makes the density matrix fail
    # positive-semidefiniteness / trace-1 validation (hit directly during testing).
    noisy_sim = cirq.DensityMatrixSimulator(noise=willow_noise_model, dtype=np.complex128)
    noiseless_sim = cirq.DensityMatrixSimulator(noise=None, dtype=np.complex128)

    def exact_observables(sim, t, steps):
        dt = t / steps
        full_circuit, all_qubits = build_trotter_circuit(
            D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
            dt, steps, SpinBosonInteractionCoefficent
        )
        compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, willow_qubit_chain, willow_target_gateset)
        z_expectations = [z.real for z in sim.simulate_expectation_values(compiled_circuit, observables=z_observables)]
        occ, mag = compute_observables_from_z_expectations(z_expectations, NumberOfFockStates, NumberOfBosonicModes)
        return occ[0], mag[0]

    noisy_error = np.zeros((len(t_grid), len(steps_grid)))
    noiseless_error = np.zeros((len(t_grid), len(steps_grid)))

    t_start = time.time()
    for ti, t in enumerate(t_grid):
        for si, steps in enumerate(steps_grid):
            occ_n, mag_n = exact_observables(noisy_sim, t, steps)
            noisy_error[ti, si] = combined_error(occ_n, mag_n, t)

            occ_i, mag_i = exact_observables(noiseless_sim, t, steps)
            noiseless_error[ti, si] = combined_error(occ_i, mag_i, t)
        print(f"t={t:5.2f}  done ({ti+1}/{len(t_grid)}, {time.time()-t_start:.0f}s elapsed)")

    np.savez(GRID_CACHE, t_grid=t_grid, steps_grid=np.array(steps_grid),
             noisy_error=noisy_error, noiseless_error=noiseless_error)

#------------------------
# Raw per-t optimum (jagged -- see module docstring)
#------------------------
optimal_idx = np.argmin(noisy_error, axis=1)
optimal_steps = np.array([steps_grid[i] for i in optimal_idx])
optimal_error = noisy_error[np.arange(len(t_grid)), optimal_idx]

baseline_idx = steps_grid.index(BASELINE_STEPS)
baseline_error = noisy_error[:, baseline_idx]

print(f"\nt, optimal_steps, error_at_optimal, error_at_steps={BASELINE_STEPS} (baseline)")
for i, t in enumerate(t_grid):
    print(f"{t:6.2f}  {optimal_steps[i]:3d}  {optimal_error[i]:.4f}  {baseline_error[i]:.4f}")

#------------------------
# Smoothing: suppress the oscillatory-aliasing artifact
#------------------------
# The dynamics are themselves oscillatory (Rabi-like flopping between boson ceiling and
# spin flip), so error at a FIXED step count doesn't grow monotonically with t -- a coarse
# step count periodically re-syncs with (or drifts further from) the true trajectory's
# phase at that instant. Point-wise argmin over a 14-point t grid partly picks up that
# beating rather than the systematic "how much Trotter error accumulates" trend. Smoothing
# the error surface along t before taking argmin suppresses this without needing a finer
# (much more expensive) t grid.
def smooth_axis0(arr, window=3):
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode='edge')
    return np.array([padded[i:i+window].mean(axis=0) for i in range(arr.shape[0])])

smoothed_error = smooth_axis0(noisy_error, window=3)
smoothed_idx = np.argmin(smoothed_error, axis=1)
smoothed_optimal_steps = np.array([steps_grid[i] for i in smoothed_idx])
smoothed_optimal_error = smoothed_error[np.arange(len(t_grid)), smoothed_idx]

print(f"\nt, smoothed_optimal_steps, smoothed_error_at_optimal")
for i, t in enumerate(t_grid):
    print(f"{t:6.2f}  {smoothed_optimal_steps[i]:3d}  {smoothed_optimal_error[i]:.4f}")

print(f"\nrecommended_trotter_steps schedule (t_grid, steps) for neuron_circuit.py:")
print("t_grid =", list(np.round(t_grid, 2)))
print("steps  =", list(int(s) for s in smoothed_optimal_steps))

#------------------------
# Plots
#------------------------
fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))

im = axes[0].imshow(
    noisy_error.T, origin='lower', aspect='auto', cmap='viridis',
    extent=[t_grid[0], t_grid[-1], 0, len(steps_grid) - 1],
)
axes[0].set_yticks(range(len(steps_grid)))
axes[0].set_yticklabels(steps_grid)
axes[0].plot(t_grid, optimal_idx, color='red', marker='o', markersize=3, linewidth=1, label='raw argmin')
axes[0].set_xlabel('t')
axes[0].set_ylabel('NumberOfTrotterSteps')
axes[0].set_title('Total (noisy) error vs (t, steps)')
axes[0].legend(fontsize=8)
fig.colorbar(im, ax=axes[0], label='combined error')

axes[1].plot(t_grid, optimal_steps, marker='o', color='tab:red', alpha=0.5, label='raw optimal steps(t)')
axes[1].plot(t_grid, smoothed_optimal_steps, marker='s', color='tab:purple', label='smoothed optimal steps(t)')
axes[1].set_xlabel('t')
axes[1].set_ylabel('NumberOfTrotterSteps')
axes[1].set_title('Optimal step count vs t')
axes[1].legend(fontsize=8)

axes[2].plot(t_grid, optimal_error, marker='o', color='tab:green', alpha=0.5, label='raw-optimal steps(t)')
axes[2].plot(t_grid, smoothed_optimal_error, marker='D', color='tab:purple', label='smoothed-optimal steps(t)')
axes[2].plot(t_grid, baseline_error, marker='s', color='tab:gray', label=f'fixed steps={BASELINE_STEPS}')
axes[2].set_xlabel('t')
axes[2].set_ylabel('combined error vs qutip')
axes[2].set_title('Achieved error: adaptive vs fixed')
axes[2].legend(fontsize=8)

# Illustrates *why* the raw argmin is jagged: trotter-only error at a fixed LOW step
# count oscillates with t instead of growing monotonically, because dt = t/steps is not
# small compared to the system's own oscillation period once steps is small and t grows.
idx_low = steps_grid.index(1)
idx_high = steps_grid.index(40)
axes[3].plot(t_grid, noiseless_error[:, idx_low], marker='o', color='tab:orange', label='trotter-only error, steps=1')
axes[3].plot(t_grid, noiseless_error[:, idx_high], marker='o', color='tab:blue', label='trotter-only error, steps=40')
axes[3].set_xlabel('t')
axes[3].set_ylabel('trotter-only error vs qutip (noise off)')
axes[3].set_title('Why raw argmin is jagged: dynamics are oscillatory')
axes[3].legend(fontsize=8)

fig.suptitle(f"N={NumberOfFockStates}, D={D_list[0]:.3g}, J={spin_interaction_coefficient}, G={SpinBosonInteractionCoefficent:.3g}, willow_pink", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(RESULTS_DIR, "optimal_trotter_steps.png"), dpi=130)
print(f"\nSaved {RESULTS_DIR}/optimal_trotter_steps.png")
