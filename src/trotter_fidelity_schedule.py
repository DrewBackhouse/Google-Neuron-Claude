"""D-18: derive the adaptive Trotter-step schedule from noiseless state fidelity
instead of the retired two-observable error metric (D-17's needles, notes/optimal-
trotter-steps.md Part 2). Replaces src/optimal_trotter_steps.py's role; that script and
its cached grid (results/optimal_trotter_steps_grid.npz) are left in place for the
before/after comparison this script prints at the end, not reused as data.

Method (D-18):
  1. Trotter requirement r_T(t): smallest r such that 1 - F(t, r') < tol for ALL r' >= r
     in the tested grid ("sustained convergence" -- immune to an isolated lucky point,
     the same quantifier D-17 used, now applied to a metric that shouldn't need it).
  2. Noise budget r_max: from the compiled circuit's two-qubit gate count per step and
     willow_pink's calibrated two-qubit error, independent of t (D-15's mechanism).
  3. Schedule: fit the monotone form r*(t) = round(c * t) to r_T(t); cap the sweep at the
     largest t with r_T(t) <= r_max.

F(t, r) = |<psi_qutip(t) | psi_trotter(t, r)>|^2, computed on the *logical* (uncompiled)
circuit -- see neuron_circuit.trotter_final_statevector's docstring for why that's
equivalent to using the Willow-compiled one here and avoids a qubit-ordering pitfall.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/trotter_fidelity_schedule.py
(GoogleQVM env for qutip + cirq_google, D-14). Re-running loads the cached grid unless
the config below changes.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sesolve, fock
import cirq
import cirq_google

from neuron_circuit import (
    QutipHamiltonian,
    find_optimal_SpinBosonInteractionCoefficent,
    find_low_error_qubit_chain,
    map_and_compile_for_willow,
    build_trotter_circuit,
    trotter_final_statevector,
    noiseless_state_fidelity,
    two_qubit_error_rate_for_chain,
    two_qubit_gates_per_trotter_step,
    estimate_max_affordable_trotter_steps,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
GRID_CACHE = os.path.join(RESULTS_DIR, "trotter_fidelity_schedule_grid.npz")

#------------------------
# CONFIG -- matches NuronSim.py's current defaults / _SCHEDULE_CONFIG
#------------------------
NumberOfBosonicModes = 1
NumberOfFockStates = 5
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]
spin_interaction_coefficient = 0.5
Time = 20
total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes

# Same grid as src/optimal_trotter_steps.py so r_T(t) is directly comparable to D-17's
# cap table and the observable-metric argmins at the 8 "basin" points.
t_grid = np.linspace(0.5, Time, 14)
steps_grid = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50, 60]

FIDELITY_TOL = 0.05          # require 1 - F < tol, i.e. F > 95%, sustained for all larger r
NOISE_BUDGET_MIN_FIDELITY = 0.5   # r_max: largest r keeping a naive two-qubit-error product above this
LEAKAGE_TOLERANCE = 1e-6     # noiseless unary-subspace leakage must be ~0 (D-18 correctness check)

#------------------------
# Ground truth (qutip) -- exact statevectors at the grid points
#------------------------
print("Calibrating G and solving the qutip ground truth...")
SpinBosonInteractionCoefficent = find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, D_list[0])
print(f"G = {SpinBosonInteractionCoefficent:.4g}")

psi0 = tensor(basis(2, 0), fock(NumberOfFockStates, 0))
H, n_list, sz_list = QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent)
# sesolve treats tlist[0] as the initial time (psi0 is assumed to already be the state
# there) -- t_grid starts at 0.5, not 0, so it must be prepended with t=0 or sesolve
# silently starts the integration from t=0.5 instead of propagating psi0 from t=0.
solve_times = np.concatenate(([0.0], np.sort(t_grid)))
qutip_result = sesolve(H, psi0, solve_times)
qutip_states_by_t = {t: state for t, state in zip(np.sort(t_grid), qutip_result.states[1:])}

#------------------------
# Fidelity grid (cached)
#------------------------
cache_valid = False
if os.path.exists(GRID_CACHE):
    cached = np.load(GRID_CACHE)
    if (np.array_equal(cached["t_grid"], t_grid) and
            np.array_equal(cached["steps_grid"], np.array(steps_grid))):
        fidelity_grid = cached["fidelity_grid"]
        leaked_grid = cached["leaked_grid"]
        cache_valid = True
        print(f"Loaded cached grid from {GRID_CACHE}")

if not cache_valid:
    fidelity_grid = np.zeros((len(t_grid), len(steps_grid)))
    leaked_grid = np.zeros((len(t_grid), len(steps_grid)))

    t_start = time.time()
    for ti, t in enumerate(t_grid):
        qutip_state = qutip_states_by_t[t]
        for si, steps in enumerate(steps_grid):
            dt = t / steps
            statevector = trotter_final_statevector(
                D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                dt, steps, SpinBosonInteractionCoefficent
            )
            fidelity, leaked = noiseless_state_fidelity(qutip_state, statevector, NumberOfFockStates, NumberOfBosonicModes)
            fidelity_grid[ti, si] = fidelity
            leaked_grid[ti, si] = leaked
        print(f"t={t:5.2f}  done ({ti+1}/{len(t_grid)}, {time.time()-t_start:.1f}s elapsed)")

    np.savez(GRID_CACHE, t_grid=t_grid, steps_grid=np.array(steps_grid),
             fidelity_grid=fidelity_grid, leaked_grid=leaked_grid)

#------------------------
# D-18 correctness check: noiseless leaked probability must be ~0 everywhere
#------------------------
max_leak = float(np.max(leaked_grid))
print(f"\nMax unary-subspace leakage over the whole grid (noiseless): {max_leak:.3e} "
      f"(tolerance {LEAKAGE_TOLERANCE:.0e})")
assert max_leak < LEAKAGE_TOLERANCE, (
    f"Noiseless Trotter state leaked {max_leak:.3e} probability out of the unary subspace "
    "-- should be exactly 0 (every Hamiltonian term conserves it). This means a "
    "compilation or qubit-ordering bug, not physics (D-18)."
)
print("PASS -- unary subspace is exactly conserved by the noiseless circuit at every (t, r) tested.")

#------------------------
# Stage 1: r_T(t) -- smallest r with sustained convergence (1-F < tol for all larger r)
#------------------------
error_grid = 1.0 - fidelity_grid


def sustained_convergence_index(errors_at_t, tol):
    """Smallest index i such that errors_at_t[i:] are all < tol. None if no such i."""
    below = errors_at_t < tol
    for i in range(len(below)):
        if np.all(below[i:]):
            return i
    return None


r_T = []
for ti in range(len(t_grid)):
    idx = sustained_convergence_index(error_grid[ti], FIDELITY_TOL)
    r_T.append(steps_grid[idx] if idx is not None else np.nan)
r_T = np.array(r_T)

print(f"\nt, r_T(t) [smallest r with 1-F<{FIDELITY_TOL} sustained], r_T/t")
for i, t in enumerate(t_grid):
    ratio = r_T[i] / t if not np.isnan(r_T[i]) else np.nan
    print(f"{t:6.2f}  {r_T[i]!s:>5}  {ratio:.3f}" if not np.isnan(r_T[i]) else f"{t:6.2f}    --   (not converged by r={steps_grid[-1]})")

valid = ~np.isnan(r_T)
ratios = r_T[valid] / t_grid[valid]
c_median_all = float(np.median(ratios))
print(f"\nFitted c = r_T/t over ALL converged points: median={c_median_all:.3f}, mean={np.mean(ratios):.3f}, "
      f"std={np.std(ratios):.3f}, range=[{ratios.min():.3f}, {ratios.max():.3f}] over {valid.sum()}/{len(t_grid)} points")

# The low-t points are quantization-dominated: steps_grid is coarse (spacing ~2 near
# r_T~10), so a small denominator t turns a +/-1-grid-point rounding error in r_T into a
# large swing in the ratio (e.g. t=0.5 is left-censored at the grid's minimum r=1; t=2.0's
# jump from grid points 8->10 alone moves the ratio by ~25%). The tail (larger t, larger
# r_T) is far less sensitive to this and is also the region that actually matters for
# setting the cap (that's where the schedule gets pushed against r_max). Fit c from the
# tail half of the converged points and use *that* as the adopted, slightly more
# conservative schedule constant, reporting the whole-range median above for comparison.
tail_mask = valid & (t_grid >= np.median(t_grid[valid]))
c_fit = float(np.median(r_T[tail_mask] / t_grid[tail_mask]))
print(f"Fitted c = r_T/t over the tail half (t >= {t_grid[valid][len(t_grid[valid])//2]:.2f}, "
      f"less quantization-sensitive): median={c_fit:.3f} over {tail_mask.sum()} points -- adopted as the schedule constant.")

#------------------------
# Stage 2: r_max from the noise budget (independent of t)
#------------------------
print("\nComputing r_max from Willow calibration + compiled gate count...")
processor_id = "willow_pink"
willow_device = cirq_google.engine.create_device_from_processor_id(processor_id)
willow_calibration = cirq_google.engine.load_median_device_calibration(processor_id)
willow_target_gateset = willow_device.metadata.compilation_target_gatesets[0]
willow_qubit_chain = find_low_error_qubit_chain(willow_device, willow_calibration, total_qubits)

gates_per_step = two_qubit_gates_per_trotter_step(
    D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
    SpinBosonInteractionCoefficent, willow_qubit_chain, willow_target_gateset
)
mean_two_qubit_error = two_qubit_error_rate_for_chain(willow_calibration, willow_qubit_chain)
r_max = estimate_max_affordable_trotter_steps(gates_per_step, mean_two_qubit_error, NOISE_BUDGET_MIN_FIDELITY)

print(f"Two-qubit gates per Trotter step (compiled): {gates_per_step}")
print(f"Mean two-qubit XEB Pauli error over the chosen chain: {mean_two_qubit_error:.4%}")
print(f"r_max (largest r keeping (1-eps)^(gates*r) >= {NOISE_BUDGET_MIN_FIDELITY}): {r_max}")

#------------------------
# Stage 3: schedule + cap
#------------------------
t_cap = r_max / c_fit if c_fit > 0 else float(t_grid[-1])
print(f"\nSchedule: r*(t) = round({c_fit:.3f} * t)")
print(f"Cap: sweep should not run past t ~= {t_cap:.2f} (where r*(t) first exceeds r_max={r_max})")

#------------------------
# Comparison against D-17's observable-metric cap table (sanity check on the "basin" points)
#------------------------
D17_BASIN_T = [0.5, 3.5, 5.0, 6.5, 11.0, 12.5, 14.0, 15.5]
D17_BASIN_R = [1, 6, 6, 10, 22, 35, 22, 40]
print("\nComparison against D-17's observable-metric cap table at the basin points "
      "(t, D-17 r-cap, this script's r_T):")
for t_b, r_b in zip(D17_BASIN_T, D17_BASIN_R):
    ti = np.argmin(np.abs(t_grid - t_b))
    print(f"  t={t_b:5.2f}  D-17 r-cap={r_b:3d}  fidelity r_T={r_T[ti]!s:>5}  "
          f"(fidelity grid t={t_grid[ti]:.2f})")

#------------------------
# Plots
#------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

im = axes[0].imshow(
    error_grid.T, origin='lower', aspect='auto', cmap='viridis',
    extent=[t_grid[0], t_grid[-1], 0, len(steps_grid) - 1], vmin=0, vmax=1,
)
axes[0].set_yticks(range(len(steps_grid)))
axes[0].set_yticklabels(steps_grid)
axes[0].set_xlabel('t')
axes[0].set_ylabel('NumberOfTrotterSteps')
axes[0].set_title('1 - F(t, r)  (noiseless)')
fig.colorbar(im, ax=axes[0], label='1 - fidelity')

axes[1].plot(t_grid[valid], r_T[valid], marker='o', color='tab:purple', label='r_T(t) (sustained conv.)')
fit_t = np.linspace(0, Time, 100)
axes[1].plot(fit_t, c_fit * fit_t, color='tab:red', linestyle='--', label=f'fit: {c_fit:.2f}*t')
axes[1].axhline(r_max, color='tab:gray', linestyle=':', label=f'r_max={r_max}')
axes[1].axvline(t_cap, color='tab:gray', linestyle=':')
axes[1].set_xlabel('t')
axes[1].set_ylabel('NumberOfTrotterSteps')
axes[1].set_title('Trotter requirement vs noise budget')
axes[1].legend(fontsize=8)

axes[2].plot(t_grid, r_T / t_grid, marker='o', color='tab:blue')
axes[2].axhline(c_fit, color='tab:red', linestyle='--', label=f'median c={c_fit:.2f}')
axes[2].set_xlabel('t')
axes[2].set_ylabel('r_T(t) / t')
axes[2].set_title('Monotonicity check: r_T/t should be ~constant')
axes[2].legend(fontsize=8)

fig.suptitle(f"N={NumberOfFockStates}, D={D_list[0]:.3g}, J={spin_interaction_coefficient}, "
             f"G={SpinBosonInteractionCoefficent:.3g}, willow_pink -- fidelity tol={FIDELITY_TOL}", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(RESULTS_DIR, "trotter_fidelity_schedule.png"), dpi=130)
print(f"\nSaved {RESULTS_DIR}/trotter_fidelity_schedule.png")

print(f"\n=== Values to hardcode into neuron_circuit.py ===")
print(f"c = {c_fit:.4f}")
print(f"r_max = {r_max}")
print(f"t_cap = {t_cap:.4f}")
