import matplotlib.pyplot as plt
import math
import numpy as np
from qutip import about, basis, tensor, destroy, mcsolve, sesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent
from scipy.optimize import minimize_scalar
import sympy
import cirq

from G_analytic import G_area_theorem

#------------------------
# INPUT
#------------------------

NumberOfBosonicModes = 1
NumberOfFockStates = 6

# Per-site displacement coefficients D_j (D-3): uniform D gives lockstep firing and no
# propagation, since H_spin has nothing to transport between identical sites. Decreasing
# D_j means site 0 reaches the Fock ceiling first. Exact profile is an open question
# (PROJECT.md) — this is a first guess to be tuned once step 1 results come in.
D_list = [1.0 * (0.85 ** j) for j in range(NumberOfBosonicModes)]

# J: chain coupling (was 0, i.e. isolated sites). First guess, scaled off D_list[0];
# tune alongside the D_j profile once propagation is visible.
#spin_interaction_coefficient = 0.2 * D_list[0]
spin_interaction_coefficient = 0

Time = 20

#------------------------
# FUNCTIONS
#------------------------

def QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent):

    sx_list, sy_list, sz_list = [], [], []                       # Define pauli X and Y operator lists
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]       # Define list of 2D and ND identity operators for each lattice site
        op_list[i] = tensor(sigmax(), qeye(NumberOfFockStates))                      # Replace ith entry with pauli X and I_N
        sx_list.append(tensor(op_list))                             # Add to Pauli X operator list
        op_list[i] = tensor(sigmay(), qeye(NumberOfFockStates))                      # Repeat for Pauli Y
        sy_list.append(tensor(op_list))
        op_list[i] = tensor(sigmaz(), qeye(NumberOfFockStates))                      # Repeat for Pauli Z
        sz_list.append(tensor(op_list))

    a_list = []                                     # Repeat for ladder and number operators
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates))
        a_list.append(tensor(op_list))

    n_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates).dag() * destroy(NumberOfFockStates))
        n_list.append(tensor(op_list))

    n_max_list = []                                 # Treat |n*><n*| as an operator and repeat
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), fock(NumberOfFockStates, NumberOfFockStates-1) * fock(NumberOfFockStates, NumberOfFockStates-1).dag())
        n_max_list.append(tensor(op_list))

    H = sum(D_list[i] * (a_list[i] + a_list[i].dag()) + SpinBosonInteractionCoefficent * n_max_list[i] * sx_list[i] for i in range(NumberOfBosonicModes))
    H += sum(spin_interaction_coefficient * (sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1])/2 for i in range(NumberOfBosonicModes-1))

    return H, n_list, sz_list

def find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, displacement_coefficient,
                                                  correction_factor=1.2, search_span=0.6, coarse_points=60,
                                                  fallback_bounds=(0.05 * np.pi, 4 * np.pi), fallback_points=400):
    """
    Calibrates G on an isolated single site (L=1, J irrelevant) — firing is a
    single-site property, so this should transfer to the full chain (open question
    in PROJECT.md; checked below by comparing against the L=NumberOfBosonicModes run).

    Seeded analytically: the truncated (a+a†) spectrum gives a closed-form (first-order,
    no-back-action) area-theorem estimate G_area (G_analytic.py). Back-action makes the
    true G larger by an empirically stable ~1.17-1.25x (notes/G-analytic-estimate.md,
    swept N=3-20, D=0.3-4) — 1.2x is used as the seed, with a local scan + refine around
    it in place of the previous blind 400-point scan over (0.05π, 4π). We want the
    *smallest* G (least back-action), and the area theorem targets the first full-flip
    window, so the seed should land near it directly.

    Falls back to the original full-range scan if the seeded window turns up nothing
    near a full flip, so correctness doesn't depend on the ratio holding everywhere.
    """
    psi0 = tensor(basis(2, 0), fock(NumberOfFockStates, 0))
    t_hit = np.sqrt(NumberOfFockStates - 1) / displacement_coefficient
    times = np.linspace(0, 2.5 * t_hit, 300)

    def objective(SpinBosonInteractionCoefficent_val):
        H, _, sz_list = QutipHamiltonian(1, NumberOfFockStates, [displacement_coefficient], 0.0, SpinBosonInteractionCoefficent_val)
        res = sesolve(H, psi0, times)
        exp_sz = expect(sz_list[0], res.states)
        return np.min(exp_sz)

    def refine_near_flip(G_grid):
        obj_grid = np.array([objective(G) for G in G_grid])
        near_flip = np.where(obj_grid < -0.99)[0]
        if len(near_flip) == 0:
            return None
        idx0 = near_flip[0]
        lo = G_grid[max(idx0 - 1, 0)]
        hi = G_grid[min(idx0 + 1, len(G_grid) - 1)]
        return minimize_scalar(objective, bounds=(lo, hi), method='bounded').x

    G_seed = correction_factor * G_area_theorem(displacement_coefficient, NumberOfFockStates)
    seeded_grid = np.linspace(G_seed * (1 - search_span), G_seed * (1 + search_span), coarse_points)
    result = refine_near_flip(seeded_grid)
    if result is not None:
        return result

    fallback_grid = np.linspace(fallback_bounds[0], fallback_bounds[1], fallback_points)
    result = refine_near_flip(fallback_grid)
    if result is not None:
        return result
    return fallback_grid[np.argmin([objective(G) for G in fallback_grid])]


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

# Transfer check (open question (a)): does the L=1-calibrated G still flip site 0 in
# the full chain, now with J and neighbouring sites present?
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

    ax1.set_title(f"Site {mode_idx}, D = {D_list[mode_idx]:.3g}", fontweight='bold', pad=10)

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

plt.suptitle(f"N = {NumberOfFockStates}, D_j = {[round(d, 3) for d in D_list]} \n J = {spin_interaction_coefficient:.3g}, G = {SpinBosonInteractionCoefficent:.3g}", fontweight='bold', fontsize=10, y=0.98)

qutip_fig.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
