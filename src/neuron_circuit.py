"""Shared Hamiltonian / circuit / Willow-mapping building blocks for the neuron model.

Extracted from NuronSim.py (2026-08-10) so this logic can be reused by analysis scripts
(e.g. optimal_trotter_steps.py) without re-running NuronSim.py's own top-level sim + plot.
NuronSim.py imports from here; it no longer defines these itself.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, sesolve, expect, qeye, sigmax, sigmay, sigmaz, fock
from scipy.optimize import minimize_scalar
import cirq
import cirq_google

from G_analytic import G_area_theorem


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

def _boson_bond_exponent(displacement_coefficient, time_evolution, j):
    """ISwapPowGate exponent implementing exp(-i (D sqrt(j+1)/2)(X_jX_{j+1}+Y_jY_{j+1}) t)
    on the (j, j+1) bond -- the single shared formula behind both BosonicDisplacementGate
    (even bonds then odd bonds, one first-order sweep of H_boson) and
    _boson_even_odd_layers (the same two bond groups exposed separately, needed so a
    higher-order outer splitting can symmetrize them independently -- see
    build_second_order_trotter_circuit)."""
    return -2 * displacement_coefficient * time_evolution * (j + 1) ** 0.5 / np.pi


# Bosonic Displacement Gate
class BosonicDisplacementGate(cirq.Gate):
    def __init__(self, displacement_coefficient, NumberOfFockStates, time_evolution):
        self.displacement_coefficient = displacement_coefficient
        self.NumberOfFockStates = NumberOfFockStates
        self.time_evolution = time_evolution

    def _num_qubits_(self) -> int:
        return self.NumberOfFockStates

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["[" + "Bosonic Displacement Gate" + "]"] + ["[" + "│".center(len("Bosonic Displacement Gate")) + "]"] * (self._num_qubits_() - 1)
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))

    def _decompose_(self, qubits):
        for j in range(0, self.NumberOfFockStates - 1, 2):
            yield cirq.ISwapPowGate(exponent=_boson_bond_exponent(self.displacement_coefficient, self.time_evolution, j))(qubits[j], qubits[j+1])
        for j in range(1, self.NumberOfFockStates - 1, 2):
            yield cirq.ISwapPowGate(exponent=_boson_bond_exponent(self.displacement_coefficient, self.time_evolution, j))(qubits[j], qubits[j+1])

# Spin Interactions Gate
class SpinInteractionsGate(cirq.Gate):
    def __init__(self, spin_interaction_coefficient, NumberOfBosonicModes, time_evolution):
        self.spin_interaction_coefficient = spin_interaction_coefficient
        self.NumberOfBosonicModes = NumberOfBosonicModes
        self.time_evolution = time_evolution

    def _num_qubits_(self) -> int:
        return self.NumberOfBosonicModes

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["[" + "Spin Interactions Gate" + "]"] + ["[" + "│".center(len("Spin Interactions Gate")) + "]"] * (self._num_qubits_() - 1)
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))

    def _decompose_(self, qubits):
        for j in range(0, self.NumberOfBosonicModes - 1, 2):
                    yield cirq.ISwapPowGate(exponent=-2 * self.spin_interaction_coefficient * self.time_evolution / (np.pi))(qubits[j], qubits[j+1])
        for j in range(1, self.NumberOfBosonicModes - 1, 2):
            yield cirq.ISwapPowGate(exponent=-2 * self.spin_interaction_coefficient * self.time_evolution / (np.pi))(qubits[j], qubits[j+1])

# --- Individual Hamiltonian-term layers, factored out of NetworkOfNeuronsTrotterStep so
# they can be recombined with different time arguments (needed for second-order/Strang
# splitting — see build_second_order_trotter_circuit) without duplicating the gate logic.

def _boson_layer(D_list, NumberOfFockStates, NumberOfBosonicModes, time_evolution, start_qubit_index=0):
    """H_boson layer: one BosonicDisplacementGate per site (D_j is per-site, D-3).
    Returns (circuit, bosonic_control_qubits, next_free_qubit_index)."""
    circuit = cirq.Circuit()
    bosonic_control_qubits = []
    current_qubit_index = start_qubit_index
    for i in range(NumberOfBosonicModes):
        bosonic_qubits = cirq.LineQubit.range(current_qubit_index, current_qubit_index + NumberOfFockStates)
        bosonic_control_qubits.append(bosonic_qubits[-1])
        bosonic_gate = BosonicDisplacementGate(D_list[i], NumberOfFockStates, time_evolution)
        circuit.append(bosonic_gate(*bosonic_qubits))
        current_qubit_index += NumberOfFockStates
    return circuit, bosonic_control_qubits, current_qubit_index


def _boson_even_odd_layers(D_list, NumberOfFockStates, NumberOfBosonicModes, time_evolution, start_qubit_index=0):
    """H_boson's even-bond and odd-bond groups, exposed as two SEPARATE circuits rather
    than bundled into one BosonicDisplacementGate sweep. Needed by
    build_second_order_trotter_circuit: even and odd bonds don't commute with each other,
    so BosonicDisplacementGate's even-then-odd decomposition is itself only a first-order
    approximation of exp(-i H_boson * time_evolution) -- fine as the atomic "boson layer"
    of a first-order outer step, but if reused unchanged as one atomic term inside a
    second-order outer splitting it would cap the *whole* circuit's convergence at first
    order regardless of how carefully the outer A/B/C ordering is symmetrized (this was
    caught by consistency_checks.py's convergence-order test, not assumed). Splitting even
    and odd into their own atomic terms lets the outer composition symmetrize them too.
    Returns (even_circuit, odd_circuit, bosonic_control_qubits, next_free_qubit_index)."""
    even_circuit = cirq.Circuit()
    odd_circuit = cirq.Circuit()
    bosonic_control_qubits = []
    current_qubit_index = start_qubit_index
    for i in range(NumberOfBosonicModes):
        bosonic_qubits = cirq.LineQubit.range(current_qubit_index, current_qubit_index + NumberOfFockStates)
        bosonic_control_qubits.append(bosonic_qubits[-1])
        D = D_list[i]
        for j in range(0, NumberOfFockStates - 1, 2):
            even_circuit.append(cirq.ISwapPowGate(exponent=_boson_bond_exponent(D, time_evolution, j))(bosonic_qubits[j], bosonic_qubits[j + 1]))
        for j in range(1, NumberOfFockStates - 1, 2):
            odd_circuit.append(cirq.ISwapPowGate(exponent=_boson_bond_exponent(D, time_evolution, j))(bosonic_qubits[j], bosonic_qubits[j + 1]))
        current_qubit_index += NumberOfFockStates
    return even_circuit, odd_circuit, bosonic_control_qubits, current_qubit_index


def _spin_layer(spin_interaction_coefficient, NumberOfBosonicModes, time_evolution, spin_qubits):
    """H_spin layer. Identically empty when NumberOfBosonicModes == 1 (no neighbour to
    hop to) -- SpinInteractionsGate._decompose_ yields nothing in that case."""
    spin_gate = SpinInteractionsGate(spin_interaction_coefficient, NumberOfBosonicModes, time_evolution)
    return cirq.Circuit(spin_gate(*spin_qubits))


def _cnot_layer(SpinBosonInteractionCoefficent, time_evolution, bosonic_control_qubits, spin_qubits):
    """H_CNOT layer: controlled-X rotation between each site's top boson qubit and its spin qubit."""
    circuit = cirq.Circuit()
    for i in range(len(bosonic_control_qubits)):
        circuit.append(cirq.rx(2 * time_evolution * SpinBosonInteractionCoefficent).controlled()(bosonic_control_qubits[i], spin_qubits[i]))
    return circuit


# Network of Neurons Trotter Step Circuit construction (first-order: A(dt) B(dt) C(dt))
def NetworkOfNeuronsTrotterStep(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes, time_evolution, SpinBosonInteractionCoefficent):

    boson_circuit, bosonic_control_qubits, current_qubit_index = _boson_layer(
        D_list, NumberOfFockStates, NumberOfBosonicModes, time_evolution
    )
    spin_qubits = cirq.LineQubit.range(current_qubit_index, current_qubit_index + NumberOfBosonicModes)
    spin_circuit = _spin_layer(spin_interaction_coefficient, NumberOfBosonicModes, time_evolution, spin_qubits)
    cnot_circuit = _cnot_layer(SpinBosonInteractionCoefficent, time_evolution, bosonic_control_qubits, spin_qubits)

    TrotterStepCircuit = cirq.Circuit()
    TrotterStepCircuit.append(boson_circuit.all_operations())
    TrotterStepCircuit.append(spin_circuit.all_operations())
    TrotterStepCircuit.append(cnot_circuit.all_operations())

    return TrotterStepCircuit

# Function to compute observables from per-qubit <Z> expectation values
def compute_observables_from_z_expectations(z_expectations, num_fock_states: int, num_bosonic_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Computes occupation numbers and magnetization from single-qubit <Z_j>
    expectation values, ordered boson qubits (per mode) then spin qubits — matching
    `willow_qubit_chain`. Both observables are sums of single-qubit marginals
    (P(qubit=1) = (1 - <Z>) / 2), so they never need the full joint distribution —
    which also makes them cheap to estimate by sampling a noisy circuit.
    """
    probs_1 = [(1.0 - z) / 2.0 for z in z_expectations]

    # 1. Bosonic occupations per mode
    boson_occ = np.zeros(num_bosonic_modes)
    for mode_idx in range(num_bosonic_modes):
        start_q = mode_idx * num_fock_states
        boson_occ[mode_idx] = sum(j * probs_1[start_q + j] for j in range(num_fock_states))

    # 2. Spin Magnetization <Z> per spin qubit (just the sampled value, relabelled)
    spin_qubit_start = num_bosonic_modes * num_fock_states
    spin_mag = np.array([
        1.0 - 2.0 * probs_1[spin_qubit_start + mode_idx] for mode_idx in range(num_bosonic_modes)
    ])

    return boson_occ, spin_mag


# Schedule constant fitted by src/adaptive_trotter_model_scan.py + src/linear_trotter_
# finegrid_scan.py (2026-08-12, D-21) directly against the noisy, post-selected observable
# error vs qutip (D-16/D-17's combined-error metric, RMS'd over a t-sweep) -- superseding
# D-18/D-19's c=4.0, which was fit against noiseless state fidelity F(t,r) instead.
#
# Why the change: D-18 moved to fidelity because the old observable-error metric could be
# fooled by accidental crossings ("needles", D-17). That fix is sound as a way to *detect
# Trotter-only convergence* without contamination, but the resulting schedule was then read
# off a metric that is blind to noise -- it will always recommend more steps than is
# actually optimal once every extra step also costs post-selection survival. D-21 confirmed
# this directly: scanning r=a*t and r=b*t^2 with noise ON and post-selection ON, linear beat
# quadratic outright (best RMS 0.157 @ a=2 vs 0.292 @ b=0.6), and a finer scan around the
# linear optimum found a clean unimodal minimum at a=2.25 (parabolic fit: a~=2.26) -- about
# half of the noiseless-fidelity-derived c=4.0, and close to D-17's own noisy-metric fit
# (c=1.76+-0.27, which D-18 had discarded along with the needle contamination it also had).
#
# r_max is unchanged by this -- it's a naive (1 - eps_2q)^(gates_per_step * r) >= 0.5 budget
# against willow_pink's calibrated two-qubit error on the compiled circuit, a property of
# the circuit/hardware, not of how c was fit (re-verified numerically post-refit: still 33).
# T_CAP = r_max / c is where this schedule first exceeds that budget if extrapolated --
# note the D-21 scan only tested t in [0.5, 10], so T_CAP=14.67 is an algebraic consequence
# of r_max and c, not itself a validated claim that a=2.25 stays optimal all the way out to
# t=14.67; re-check if that range starts mattering.
#
# Specific to the config below (G calibrates to ~2.03) against one willow_pink median
# calibration snapshot -- re-run src/adaptive_trotter_model_scan.py if any of these change,
# this is not a general law. _SCHEDULE_CONFIG lets callers check they still match it.
_SCHEDULE_CONFIG = dict(NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=(1.0,), spin_interaction_coefficient=0.5)
TROTTER_SCHEDULE_C = 2.25
TROTTER_SCHEDULE_R_MAX = 33
TROTTER_SCHEDULE_T_CAP = TROTTER_SCHEDULE_R_MAX / TROTTER_SCHEDULE_C


def check_trotter_schedule_config(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient):
    """Warns (does not raise) if the current model config doesn't match the one
    recommended_trotter_steps was derived for -- the schedule is a fitted result for one
    specific Hamiltonian + noise environment, not a general law (see its docstring)."""
    current = dict(NumberOfBosonicModes=NumberOfBosonicModes, NumberOfFockStates=NumberOfFockStates,
                    D_list=tuple(D_list), spin_interaction_coefficient=spin_interaction_coefficient)
    if current != _SCHEDULE_CONFIG:
        print(f"WARNING: recommended_trotter_steps was derived for {_SCHEDULE_CONFIG}, "
              f"but the current config is {current}. The schedule likely no longer reflects "
              f"the true error-minimizing step counts -- re-run src/trotter_fidelity_schedule.py.")


def recommended_trotter_steps(t: float) -> int:
    """NumberOfTrotterSteps for time t: the monotone schedule r*(t) = round(TROTTER_SCHEDULE_C * t),
    fit directly against noisy, post-selected observable error (D-21,
    src/adaptive_trotter_model_scan.py + src/linear_trotter_finegrid_scan.py).
    Monotone by construction, so unlike D-16's interpolated per-point argmin it cannot
    produce D-17's accidental low-step dips. Does not itself enforce the noise-budget cap
    (TROTTER_SCHEDULE_T_CAP) -- callers sweeping a t range should check that separately
    (e.g. via trotter_schedule_cap_message) and cap the sweep rather than call this
    one point at a time past it, since printing a warning on every one of ~100 sweep
    points would be noise, not signal.
    """
    return max(round(TROTTER_SCHEDULE_C * t), 1)


def trotter_schedule_cap_message(requested_time: float) -> str | None:
    """Returns a one-line warning if requested_time exceeds TROTTER_SCHEDULE_T_CAP (the
    noise budget cap, D-18 stage 2/3), else None. Meant to be checked once per sweep
    (not once per point) -- see recommended_trotter_steps' docstring."""
    if requested_time > TROTTER_SCHEDULE_T_CAP:
        return (f"WARNING: requested sweep time {requested_time:.2f} exceeds the Trotter "
                f"schedule's noise-budget cap (t_cap={TROTTER_SCHEDULE_T_CAP:.2f}, "
                f"r_max={TROTTER_SCHEDULE_R_MAX}) -- beyond this point the required step "
                f"count is not affordable under the noise budget, so results are not "
                f"Trotter-converged at any reachable step count (D-17/D-18/D-21).")
    return None


# ---------------------------------------------------------------------------
# Second-order (Strang) schedule (D-22) -- own constant, not a reuse of the first-order
# one. src/second_order_trotter_linear_scan.py repeated D-21's coarse-then-fine method
# directly on build_second_order_trotter_circuit: coarse grid over a broad a range, fine
# grid around the winner. Result: a clean unimodal minimum at a=2.0 (parabolic fit:
# a~=2.15) -- essentially the SAME coefficient as first order's a=2.25, despite second
# order's ~r^-2 (vs ~r^-1) operator-norm convergence. The naive expectation was that
# faster convergence should let second order use noticeably fewer steps; that expectation
# turned out to be roughly cancelled by its higher per-step gate cost (16 two-qubit gates
# for one unmerged Strang step here vs 10 for one first-order step -- a ~1.6x ratio,
# consistent with D-20's ~1.51x amortized-large-r estimate).
#
# Head-to-head at each order's own optimum (results/first_vs_second_order_optimized_
# trotter_comparison.png): first order still wins on RMS error (0.1565 vs 0.1657, ~6%
# relative), so D-20's original finding (second order loses once noise is real) survives
# even with a fair, order-specific schedule instead of the shared first-order one it was
# testing against before -- not a schedule-mismatch artefact after all. The margin is
# small and time-dependent, not uniform: per-point error (comparison plot) shows second
# order tracking BETTER at early-to-mid t (t<7, where first order has a pronounced local
# bump around t~2) and WORSE past t~7.5 -- the RMS verdict is close and driven by the
# late-time region, not a blowout either way.
#
# r_max recomputed analogously (not reused from first order): two_qubit_gates_per_
# trotter_step run against a single (r=1, unmerged) Strang step -- deliberately the
# conservative, un-amortized per-step count rather than the slightly lower large-r
# average, matching estimate_max_affordable_trotter_steps' own "deliberately coarse"
# spirit. Gives r_max=20 (vs 33 for first order, tracking the ~1.6x gate-count ratio) and
# T_CAP=r_max/c=10.0 -- which happens to sit right at the edge of D-22's tested range
# (t<=10), so unlike first order's T_CAP this one is NOT an extrapolation past what was
# actually scanned.
TROTTER_SCHEDULE_C_2ND_ORDER = 2.0
TROTTER_SCHEDULE_R_MAX_2ND_ORDER = 20
TROTTER_SCHEDULE_T_CAP_2ND_ORDER = TROTTER_SCHEDULE_R_MAX_2ND_ORDER / TROTTER_SCHEDULE_C_2ND_ORDER


def recommended_trotter_steps_2nd_order(t: float) -> int:
    """NumberOfTrotterSteps for time t, for the SECOND-order (Strang) circuit: the
    monotone schedule r*(t) = round(TROTTER_SCHEDULE_C_2ND_ORDER * t), fit directly
    against noisy, post-selected observable error on build_second_order_trotter_circuit
    (D-22, src/second_order_trotter_linear_scan.py). Own coefficient, own noise-budget
    cap -- do not reuse recommended_trotter_steps (the first-order schedule) for the
    second-order circuit; see the module-level comment above TROTTER_SCHEDULE_C_2ND_ORDER."""
    return max(round(TROTTER_SCHEDULE_C_2ND_ORDER * t), 1)


def trotter_schedule_cap_message_2nd_order(requested_time: float) -> str | None:
    """Second-order analogue of trotter_schedule_cap_message -- checked once per sweep,
    not once per point."""
    if requested_time > TROTTER_SCHEDULE_T_CAP_2ND_ORDER:
        return (f"WARNING: requested sweep time {requested_time:.2f} exceeds the SECOND-ORDER "
                f"Trotter schedule's noise-budget cap (t_cap={TROTTER_SCHEDULE_T_CAP_2ND_ORDER:.2f}, "
                f"r_max={TROTTER_SCHEDULE_R_MAX_2ND_ORDER}) -- beyond this point the required step "
                f"count is not affordable under the noise budget, so results are not "
                f"Trotter-converged at any reachable step count (D-22).")
    return None


def required_adjacency_edges(NumberOfFockStates: int, NumberOfBosonicModes: int) -> set:
    """Logical-qubit pairs (cirq.LineQubit, matching build_trotter_circuit's ordering --
    each site's N boson qubits, then one spin qubit per site) that need a two-qubit gate
    somewhere in one Trotter step. Derived by actually building and decomposing
    NetworkOfNeuronsTrotterStep with dummy (nonzero) coefficients and reading off its
    two-qubit operations, rather than hand-derived from the Hamiltonian's structure --
    only the circuit's qubit-usage STRUCTURE matters here, not the coefficient values, and
    deriving it directly from the real circuit builder means this can never drift out of
    sync if that construction changes.

    At NumberOfBosonicModes==1 this is a simple chain (every consecutive LineQubit pair).
    At NumberOfBosonicModes>1 it is a comb: L separate site-chains of N boson qubits each,
    joined only through a spine of spin-spin links (H_spin) at one end -- see
    find_low_error_qubit_embedding for why that does NOT embed as a straight 1D chain once
    NumberOfBosonicModes>=3."""
    dummy_D_list = [1.0] * NumberOfBosonicModes
    step_circuit = NetworkOfNeuronsTrotterStep(
        dummy_D_list, 1.0, NumberOfFockStates, NumberOfBosonicModes, 0.1, 1.0
    )
    decomposed = cirq.Circuit(cirq.decompose(step_circuit))
    edges = set()
    for op in decomposed.all_operations():
        if len(op.qubits) == 2:
            edges.add(tuple(sorted(op.qubits)))
    return edges


def find_low_error_qubit_embedding(device: "cirq_google.GridDevice", calibration: "cirq_google.Calibration",
                                    NumberOfFockStates: int, NumberOfBosonicModes: int) -> list:
    """Finds a low-error placement of the logical circuit (build_trotter_circuit's
    LineQubit.range(total_qubits), in order) onto real device qubits such that every
    required two-qubit gate lands on an actual device edge -- SWAP-free by construction
    (D-1). Returns a list ordered to match LineQubit.range(total_qubits), so it drops
    straight into map_and_compile_for_willow exactly like the old chain search did.

    Generalizes the previous chain-only search (correct only for NumberOfBosonicModes<=1,
    where the logical topology genuinely is a straight path) to the comb-shaped topology
    every NumberOfBosonicModes>1 config actually needs: L site-chains of N boson qubits
    each, joined only through a spine of spin-spin links (H_spin) at one end. That comb has
    NO Hamiltonian path once NumberOfBosonicModes>=3: each site's far boson qubit (the end
    away from its spin qubit) has degree 1 in the LOGICAL adjacency graph, i.e. is forced
    to be a path endpoint -- and a path has only two endpoints, so >=3 such qubits already
    makes a straight-chain embedding impossible regardless of the search algorithm. Forcing
    the old chain search's qubit order onto a straight hardware chain for
    NumberOfBosonicModes==2 (where a Hamiltonian path *does* exist, via a boson-spin-spin-
    boson "snake") was the bug this replaces: `LineQubit.range` orders all of site 0's
    qubits before all of site 1's, so the CNOT layer's site-1 boson-to-spin gate lands on
    logical qubits that are far apart in that ordering and therefore not adjacent on a
    hardware chain built to match consecutive LineQubit indices -- e.g. the
    'ValueError: Qubit pair is not valid on device' seen running NumberOfBosonicModes=2.

    Backtracking search: places logical qubits in an order that keeps each newly placed
    qubit adjacent (in the required-edges graph) to at least one already-placed qubit, and
    at each step restricts candidates to device qubits that are device-neighbours of every
    already-placed logical neighbour -- greedily preferring the lowest calibrated two-qubit
    XEB Pauli error, the same heuristic the old chain search used. A single such search
    only finds *a* valid embedding, not the lowest-error one -- it commits to the first
    complete assignment found and never revisits an earlier, locally-fine-looking choice
    just because a later edge (e.g. the one carrying H_CNOT) turned out worse than it could
    have been. So the search is repeated once per candidate starting device qubit (there
    are only ~100, and each attempt is fast), and the embedding with the lowest total score
    among all successful attempts is kept -- this matters in practice: an earlier version
    that returned the first success placed a config's H_CNOT edge on a ~1.7x higher-error
    link than this version does, degrading spin magnetization specifically (post-selection
    has no way to catch or correct that -- it only checks the boson registers' one-hot
    constraint, D-4) while leaving boson occupation and survival rate looking fine, i.e. a
    regression that would NOT show up in the usual health checks.

    The score itself combines TWO calibration signals, each converted to a percentile rank
    among all device edges/qubits (0=best, 1=worst) so they're on comparable, unit-free
    scales before summing: two-qubit CZ XEB Pauli error per required edge, and T1-driven
    idle decoherence (1/T1) per qubit used. Two-qubit error alone is not enough -- checked
    directly (2026-08-12): a version scoring only on summed two-qubit edge error found an
    embedding with LOWER total two-qubit error than a known-good reference chain, yet
    reproducibly (3 independent noisy runs) gave ~50% worse RMS error against qutip, because
    it happened to route through one qubit with T1=39us against a chip-wide typical ~70us --
    a coherence problem the two-qubit-only score couldn't see. Percentile ranks (rather than
    a hand-tuned relative weight between a dimensionless error probability and an
    inverse-microseconds quantity) sidestep needing to calibrate that weight from gate
    durations this function doesn't have access to.

    Even that percentile score is a SOFT trade-off, though, and turned out not to be
    enough on its own either -- checked directly (2026-08-12): the same T1=39us qubit
    still got selected for two different (larger) configs even with the percentile score
    active, because its neighbourhood's two-qubit edges were good enough that the summed
    score still favoured including it over an alternative that avoided it -- inflating
    noisy-run RMS error against qutip by ~2x for those configs. So candidacy now has a
    hard floor too: any device qubit below the 10th percentile of chip-wide T1 is excluded
    from the search outright, before scoring -- it doesn't matter how good its edges are.

    Raises if no embedding is found at all -- possible in principle even on a 2D device at
    large NumberOfBosonicModes/NumberOfFockStates, and now slightly more likely given the
    T1 floor shrinks the usable qubit pool by construction; not exercised here beyond the
    sizes in consistency_checks.py.
    """
    edges = required_adjacency_edges(NumberOfFockStates, NumberOfBosonicModes)
    total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes
    logical_qubits = cirq.LineQubit.range(total_qubits)

    neighbor_map = {q: set() for q in logical_qubits}
    for a, b in edges:
        neighbor_map[a].add(b)
        neighbor_map[b].add(a)

    # Placement order: BFS from the qubit with the most logical neighbours, so every
    # qubit placed after the first already has a placed neighbour to anchor the device-
    # adjacency search against (keeps candidate sets small instead of scanning the whole
    # device for an unconstrained node).
    start_logical = max(logical_qubits, key=lambda q: len(neighbor_map[q]))
    order = [start_logical]
    seen = {start_logical}
    frontier = [start_logical]
    while frontier:
        next_frontier = []
        for q in frontier:
            for n in sorted(neighbor_map[q], key=lambda x: x.x):
                if n not in seen:
                    seen.add(n)
                    order.append(n)
                    next_frontier.append(n)
        frontier = next_frontier
    for q in logical_qubits:  # only reachable if some logical qubit has no edges at all
        if q not in seen:
            order.append(q)

    graph = device.metadata.nx_graph
    two_qubit_error = calibration['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle']
    t1_micros = calibration['single_qubit_idle_t1_micros']

    def edge_error(a, b):
        val = two_qubit_error.get((a, b)) or two_qubit_error.get((b, a))
        return val[0] if val is not None else 1.0  # uncalibrated edge: treat as worst-case

    def inv_t1(q):
        val = t1_micros.get((q,))
        return 1.0 / val[0] if val is not None else 1.0  # uncalibrated qubit: treat as worst-case

    def mean_neighbour_error(node):
        neighbours = list(graph.neighbors(node))
        return np.mean([edge_error(node, n) for n in neighbours]) if neighbours else 1.0

    # Hard T1 floor: exclude the chip-wide worst decile from candidacy entirely, rather
    # than only soft-penalizing it via the percentile score below. The percentile score
    # alone let one T1=39us qubit (chip median ~70us) into two chains anyway -- checked
    # directly (2026-08-12): its neighbourhood's two-qubit edges were good enough that the
    # SUM of percentile ranks still favoured including it over a chain that avoided it,
    # inflating noisy-run RMS error against qutip by ~2x for those configs despite the
    # percentile score being "aware" of it. A hard floor on the input candidate pool fixes
    # this the way the soft penalty couldn't: this qubit (and its rank-mates) simply
    # cannot be chosen, regardless of how good its edges are.
    T1_FLOOR_PERCENTILE = 10
    t1_floor = np.percentile([t1_micros[(q,)][0] for q in graph.nodes() if (q,) in t1_micros], T1_FLOOR_PERCENTILE)
    usable_qubits = {q for q in graph.nodes() if t1_micros.get((q,), [0.0])[0] >= t1_floor}

    device_qubits_by_quality = sorted(usable_qubits, key=mean_neighbour_error)

    # Percentile-rank tables for the final cross-attempt scoring (see docstring) -- built
    # once, over every device edge/qubit, so any candidate's value can be looked up by
    # position in a sorted array rather than repeatedly recomputing a rank from scratch.
    all_edge_errors = np.sort([edge_error(a, b) for a, b in graph.edges()])
    all_inv_t1 = np.sort([inv_t1(q) for q in graph.nodes()])

    def edge_error_percentile(a, b):
        return np.searchsorted(all_edge_errors, edge_error(a, b)) / max(len(all_edge_errors) - 1, 1)

    def t1_penalty_percentile(q):
        return np.searchsorted(all_inv_t1, inv_t1(q)) / max(len(all_inv_t1) - 1, 1)

    def attempt(forced_first_choice):
        placement = {}
        used = set()

        def candidates_for(logical_q, i):
            if i == 0:
                return [forced_first_choice]
            placed_neighbours = [placement[n] for n in neighbor_map[logical_q] if n in placement]
            if not placed_neighbours:
                return device_qubits_by_quality
            common = set(graph.neighbors(placed_neighbours[0])) & usable_qubits
            for pn in placed_neighbours[1:]:
                common &= set(graph.neighbors(pn))
            return sorted(common, key=lambda dq: np.mean([edge_error(dq, pn) for pn in placed_neighbours]))

        def backtrack(i):
            if i == len(order):
                return True
            logical_q = order[i]
            for dq in candidates_for(logical_q, i):
                if dq in used:
                    continue
                placement[logical_q] = dq
                used.add(dq)
                if backtrack(i + 1):
                    return True
                del placement[logical_q]
                used.discard(dq)
            return False

        return dict(placement) if backtrack(0) else None

    best_placement, best_score = None, None
    for start_dq in device_qubits_by_quality:
        result = attempt(start_dq)
        if result is None:
            continue
        score = (sum(edge_error_percentile(result[a], result[b]) for a, b in edges)
                 + sum(t1_penalty_percentile(dq) for dq in result.values()))
        if best_score is None or score < best_score:
            best_placement, best_score = result, score

    if best_placement is None:
        raise RuntimeError(
            f"No SWAP-free embedding found for {total_qubits} logical qubits "
            f"(NumberOfBosonicModes={NumberOfBosonicModes}, NumberOfFockStates={NumberOfFockStates}) "
            f"on this device."
        )
    return [best_placement[q] for q in logical_qubits]


def map_and_compile_for_willow(circuit: cirq.Circuit, source_qubits: list, qubit_chain: list, target_gateset) -> cirq.Circuit:
    """Relabels `circuit` from `source_qubits` (in order) onto the hardware `qubit_chain`,
    then compiles into Willow's native gateset (CZ + single-qubit gates)."""
    qubit_map = dict(zip(source_qubits, qubit_chain))
    mapped_circuit = circuit.transform_qubits(qubit_map)
    return cirq.optimize_for_target_gateset(mapped_circuit, gateset=target_gateset)


def plot_qubit_embedding_overlay(willow_calibration, willow_qubit_chain: list,
                                  NumberOfFockStates: int, NumberOfBosonicModes: int):
    """Willow calibration grid (T1 per qubit, two-qubit CZ error per bond -- same layout as
    src/willow_calibration_grid.py) with the specific qubits and edges THIS run's embedding
    (find_low_error_qubit_embedding) actually uses highlighted in red on top -- lets the
    chosen device patch be checked against the chip-wide calibration at a glance, e.g.
    whether it's hugging a high-T1/low-error region or was forced through a mediocre one.

    used_edges is derived from required_adjacency_edges (the same source-of-truth used by
    find_low_error_qubit_embedding itself) mapped through willow_qubit_chain, rather than
    just connecting consecutive chain entries -- correct at NumberOfBosonicModes<=1 (a
    simple chain) and also at NumberOfBosonicModes>1, where the logical topology is a comb
    and consecutive LineQubit indices are not all logically adjacent.

    Returns the created Figure; caller saves/shows/closes it (matching NuronSim.py's own
    figure-handling convention rather than this module doing file I/O)."""
    used_qubits = list(dict.fromkeys(willow_qubit_chain))  # de-duplicate, preserve order
    logical_edges = required_adjacency_edges(NumberOfFockStates, NumberOfBosonicModes)
    used_edges = [(willow_qubit_chain[a.x], willow_qubit_chain[b.x]) for a, b in logical_edges]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    willow_calibration.heatmap("single_qubit_idle_t1_micros").plot(ax1)
    ax1.scatter([q.col for q in used_qubits], [q.row for q in used_qubits],
                s=280, facecolors='none', edgecolors='red', linewidths=2.5, zorder=5)

    two_qubit_error_pct = {
        willow_calibration.key_to_qubits(key): willow_calibration.value_to_float(value) * 100
        for key, value in willow_calibration["two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"].items()
    }
    cirq.TwoQubitInteractionHeatmap(
        two_qubit_error_pct,
        title="Two Qubit Parallel Cz Gate Xeb Pauli Error Per Cycle",
        annotation_format=".2f",
        annotation_text_kwargs={"fontsize": 7},
        colorbar_options={"label": "%"},
    ).plot(ax2)
    for qa, qb in used_edges:
        ax2.plot([qa.col, qb.col], [qa.row, qb.row], color='red', linewidth=3.5, zorder=5, solid_capstyle='round')
    ax2.scatter([q.col for q in used_qubits], [q.row for q in used_qubits],
                s=45, color='red', zorder=6)

    fig.suptitle(
        f"willow_pink calibration ({willow_calibration.timestamp_str()}) -- current embedding highlighted",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def build_trotter_circuit(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                           dt, NumberOfTrotterSteps, SpinBosonInteractionCoefficent):
    """Builds the full logical circuit (vacuum state prep + NumberOfTrotterSteps repeats
    of a single dt-step), as cirq.LineQubits — ready to be mapped onto hardware qubits
    via map_and_compile_for_willow. Pulled out of NuronSim.py's execution loop so
    analysis scripts (e.g. optimal_trotter_steps.py) build circuits identically."""
    total_qubits = (NumberOfBosonicModes * NumberOfFockStates) + NumberOfBosonicModes
    all_qubits = cirq.LineQubit.range(total_qubits)

    full_circuit = cirq.Circuit()
    for i in range(NumberOfBosonicModes):
        full_circuit.append(cirq.X(all_qubits[i * NumberOfFockStates]))

    trotter_step_circuit = NetworkOfNeuronsTrotterStep(
        D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes, dt, SpinBosonInteractionCoefficent
    )
    decomposed_step = cirq.Circuit(cirq.decompose(trotter_step_circuit))
    full_circuit.append(decomposed_step for _ in range(NumberOfTrotterSteps))

    return full_circuit, all_qubits


def build_second_order_trotter_circuit(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                                        dt, NumberOfTrotterSteps, SpinBosonInteractionCoefficent):
    """Second-order (Strang/symmetric) Trotterization.

    H_boson's own even/odd bond groups do not commute with each other (see
    _boson_even_odd_layers), so they must be treated as their own atomic terms in the
    symmetric composition, not bundled via the first-order-only BosonicDisplacementGate --
    doing that instead bottlenecks the whole circuit to first-order convergence regardless
    of how the outer terms are ordered (caught by consistency_checks.py's log-log
    convergence-order fit: bundled gave the same ~r^-2 scaling as the first-order circuit,
    not the ~r^-4 a true second-order method gives). With four atomic terms in order
    Even, Odd, Spin (B), CNOT (C), the symmetric (palindromic) single dt-step is:

        Even(dt/2) Odd(dt/2) B(dt/2) C(dt) B(dt/2) Odd(dt/2) Even(dt/2)

    Repeating r times and merging neighbouring half-strength layers at adjacent step
    boundaries (e^{Even dt/2} e^{Even dt/2} = e^{Even dt}) -- only the OUTERMOST term
    (Even) is adjacent across a step boundary, so only it merges; Odd and B stay at half
    strength on both sides of every step, and C stays once per step at full strength:

        Even(dt/2) [Odd(dt/2) B(dt/2) C(dt) B(dt/2) Odd(dt/2) Even(dt)]^(r-1)
                    Odd(dt/2) B(dt/2) C(dt) B(dt/2) Odd(dt/2) Even(dt/2)

    layer counts: Even: r+1, Odd: 2r, B: 2r, C: r (vs 2r/2r/2r/r for naive unmerged
    concatenation of r independent symmetric steps -- merging only removes the redundant
    Even applications). Odd sits between two Bs (or, at L=1, two C-adjacent slots) on
    *both* sides of every step, so it never lands at a step boundary and cannot merge --
    it stays at 2r applications regardless. Net boson-gate overhead vs first order (which
    applies Even+Odd once each per step) is therefore driven by Odd's doubling, ~1.5x
    (confirmed empirically in consistency_checks.py), not 1x and not the ~2x a fully
    unmerged construction would cost. At NumberOfBosonicModes == 1, B is identically empty
    (_spin_layer), so the accounting above simplifies to just Even/Odd/C.
    """
    total_qubits = (NumberOfBosonicModes * NumberOfFockStates) + NumberOfBosonicModes
    all_qubits = cirq.LineQubit.range(total_qubits)

    full_circuit = cirq.Circuit()
    for i in range(NumberOfBosonicModes):
        full_circuit.append(cirq.X(all_qubits[i * NumberOfFockStates]))

    r = NumberOfTrotterSteps
    if r <= 0:
        return full_circuit, all_qubits

    even_half, odd_half, bosonic_control_qubits, current_qubit_index = _boson_even_odd_layers(
        D_list, NumberOfFockStates, NumberOfBosonicModes, dt / 2
    )
    even_full, _, _, _ = _boson_even_odd_layers(D_list, NumberOfFockStates, NumberOfBosonicModes, dt)
    spin_qubits = cirq.LineQubit.range(current_qubit_index, current_qubit_index + NumberOfBosonicModes)
    spin_half = _spin_layer(spin_interaction_coefficient, NumberOfBosonicModes, dt / 2, spin_qubits)
    cnot_full = _cnot_layer(SpinBosonInteractionCoefficent, dt, bosonic_control_qubits, spin_qubits)

    def append_layer(layer_circuit):
        full_circuit.append(cirq.decompose(layer_circuit))

    append_layer(even_half)
    for step in range(r):
        append_layer(odd_half)
        append_layer(spin_half)
        append_layer(cnot_full)
        append_layer(spin_half)
        append_layer(odd_half)
        append_layer(even_full if step < r - 1 else even_half)

    return full_circuit, all_qubits


# ---------------------------------------------------------------------------
# Unary-subspace projection and noiseless state fidelity (D-18)
# ---------------------------------------------------------------------------

def _unary_basis_index(fock_indices, spin_bits, NumberOfFockStates, NumberOfBosonicModes):
    """Bit index into a 2^(L*(N+1))-dim qubit statevector for the unary-encoded physical
    basis state with site i's boson mode in Fock state fock_indices[i] and spin qubit i
    in spin_bits[i]. Qubit layout matches build_trotter_circuit / NetworkOfNeuronsTrotterStep:
    boson qubits grouped by site (site0's N qubits, ..., siteL-1's N qubits), then one
    spin qubit per site, and cirq's default big-endian statevector indexing (qubit 0 most
    significant)."""
    N = NumberOfFockStates
    L = NumberOfBosonicModes
    bits = [0] * (L * N + L)
    for site in range(L):
        bits[site * N + fock_indices[site]] = 1
    for site in range(L):
        bits[L * N + site] = spin_bits[site]
    index = 0
    for b in bits:
        index = (index << 1) | b
    return index


def _physical_basis_index(fock_indices, spin_bits, NumberOfFockStates):
    """Index into the (2N)^L-dim physical Hilbert space in qutip's tensor ordering:
    QutipHamiltonian tensors per-site blocks tensor(spin(2), boson(N)) together, site 0
    most significant, spin more significant than boson within a site -- matching
    tensor(basis(2,s), fock(N,n)) and psi0's construction."""
    N = NumberOfFockStates
    index = 0
    for s, n in zip(spin_bits, fock_indices):
        index = index * 2 + s
        index = index * N + n
    return index


def unary_subspace_projection(statevector, NumberOfFockStates, NumberOfBosonicModes):
    """Projects a 2^(L*(N+1))-dim qubit statevector onto the physical unary subspace
    (every boson column exactly one-hot), returning the amplitudes reindexed into
    qutip's (2N)^L physical basis ordering plus the leaked probability (norm^2 of the
    discarded, unphysical component). Every Hamiltonian term conserves the unary
    constraint exactly (each boson-column excitation number is conserved by H_boson's
    XX+YY form, and H_CNOT/H_spin act on qubits outside the column they don't touch), so
    a noiseless Trotterized state's leaked probability should be 0 to numerical precision
    at every step count -- this doubles as a qubit-ordering/compilation smoke test (D-18)."""
    N = NumberOfFockStates
    L = NumberOfBosonicModes
    dim_phys = (2 * N) ** L
    projected = np.zeros(dim_phys, dtype=complex)
    subspace_probability = 0.0
    for spin_bits in itertools.product((0, 1), repeat=L):
        for fock_indices in itertools.product(range(N), repeat=L):
            uidx = _unary_basis_index(fock_indices, spin_bits, N, L)
            pidx = _physical_basis_index(fock_indices, spin_bits, N)
            amp = statevector[uidx]
            projected[pidx] = amp
            subspace_probability += abs(amp) ** 2
    leaked_probability = 1.0 - subspace_probability
    return projected, leaked_probability


def noiseless_state_fidelity(qutip_state, trotter_statevector, NumberOfFockStates, NumberOfBosonicModes):
    """F(t, r) = |<psi_qutip(t) | psi_trotter(t, r)>|^2 (D-18) -- projects the Trotter
    circuit's qubit statevector onto the physical unary subspace and takes the overlap
    with qutip's full statevector there. |.|^2 makes global phase irrelevant. Returns
    (fidelity, leaked_probability); leaked_probability should be ~0 for a noiseless run
    (see unary_subspace_projection)."""
    projected, leaked_probability = unary_subspace_projection(trotter_statevector, NumberOfFockStates, NumberOfBosonicModes)
    qutip_amplitudes = np.asarray(qutip_state.full()).flatten()
    fidelity = float(np.abs(np.vdot(qutip_amplitudes, projected)) ** 2)
    return fidelity, leaked_probability


def trotter_final_statevector(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                               dt, NumberOfTrotterSteps, SpinBosonInteractionCoefficent, circuit_builder=build_trotter_circuit):
    """Noiseless final statevector of the *logical* (uncompiled) Trotter circuit, in
    build_trotter_circuit's LineQubit order. Computed pre-Willow-mapping rather than on
    the compiled circuit: compilation to Willow's native gateset is an exact unitary
    resynthesis, not an approximation, so the statevector (and hence any fidelity
    computed from it) is unaffected -- and this sidesteps having to track
    willow_qubit_chain's qubit order through the simulator (the ordering pitfall D-18
    flags), since LineQubit order is controlled directly here."""
    full_circuit, all_qubits = circuit_builder(
        D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
        dt, NumberOfTrotterSteps, SpinBosonInteractionCoefficent
    )
    result = cirq.Simulator().simulate(full_circuit, qubit_order=all_qubits)
    return result.final_state_vector


# ---------------------------------------------------------------------------
# Noise-budget estimate for r_max (D-18 stage 2) -- a property of the compiled circuit
# and calibration snapshot, independent of t (D-15's mechanism).
# ---------------------------------------------------------------------------

def two_qubit_error_rate_for_chain(willow_calibration, willow_qubit_chain):
    """Mean two-qubit XEB Pauli error over the edges of the chosen qubit chain -- same
    calibration field find_low_error_qubit_embedding optimizes against."""
    two_qubit_error = willow_calibration['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle']
    errs = []
    for a, b in zip(willow_qubit_chain[:-1], willow_qubit_chain[1:]):
        val = two_qubit_error.get((a, b)) or two_qubit_error.get((b, a))
        if val is not None:
            errs.append(val[0])
    return float(np.mean(errs)) if errs else 1.0


def two_qubit_gates_per_trotter_step(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                                      SpinBosonInteractionCoefficent, willow_qubit_chain, target_gateset,
                                      step_builder=NetworkOfNeuronsTrotterStep, dt=0.1):
    """Two-qubit gate count of one compiled dt-step. Independent of dt's actual value
    (only the rotation angles depend on it, not the gate count), so any nonzero dt gives
    the right answer -- used as a fixed per-step cost for the r_max noise-budget estimate."""
    step_circuit = step_builder(D_list, spin_interaction_coefficient, NumberOfFockStates,
                                 NumberOfBosonicModes, dt, SpinBosonInteractionCoefficent)
    decomposed = cirq.Circuit(cirq.decompose(step_circuit))
    total_qubits = (NumberOfBosonicModes * NumberOfFockStates) + NumberOfBosonicModes
    all_qubits = cirq.LineQubit.range(total_qubits)
    compiled = map_and_compile_for_willow(decomposed, all_qubits, willow_qubit_chain, target_gateset)
    return sum(1 for op in compiled.all_operations() if len(op.qubits) == 2)


def estimate_max_affordable_trotter_steps(gates_per_step, mean_two_qubit_error, min_fidelity_budget=0.5, max_r_search=1000):
    """Largest step count r such that a simple product-of-gate-fidelities estimate,
    (1 - mean_two_qubit_error)^(gates_per_step * r), stays at or above min_fidelity_budget.
    Deliberately coarse (ignores single-qubit/readout error, T1, and any noise
    cancellation, D-2's "budget against cumulative two-qubit gate count" argument) --
    it only needs to set where the sweep gets capped (D-18 stage 2), not predict an exact
    fidelity."""
    if gates_per_step <= 0 or mean_two_qubit_error <= 0:
        return max_r_search
    for r in range(1, max_r_search + 1):
        if (1.0 - mean_two_qubit_error) ** (gates_per_step * r) < min_fidelity_budget:
            return max(r - 1, 1)
    return max_r_search


# ---------------------------------------------------------------------------
# Post-selection on the unary constraint (D-4): discard shots where any boson column
# isn't exactly one-hot -- the detectable signature of bit-flip/amplitude-damping-type
# noise leaving the physical subspace.
# ---------------------------------------------------------------------------

def sample_shots_with_postselection(sim, compiled_circuit, qubit_chain, NumberOfFockStates, NumberOfBosonicModes, num_samples):
    """Runs compiled_circuit num_samples times measuring every qubit (qubit_chain order:
    boson qubits per mode, then spin qubits -- matching compute_observables_from_z_expectations),
    then computes bosonic occupation and spin magnetization two ways from the same shots:

    - *_raw: marginal estimator over all shots (sum_j j * P(qubit_j=1) per mode), same
      quantity NuronSim.py's sample_expectation_values path was already estimating.
    - *_post: identical estimator but restricted to shots where every mode's boson
      register is exactly one-hot (the physical/unary subspace) -- shots that leaked out
      of it (detectable, D-4) are discarded rather than silently averaged in.
    - *_removed: the complementary estimator, over exactly the shots *_post discards --
      lets a caller plot what post-selection is actually throwing away, not just what it
      keeps. NaN (via the same estimate() helper, not a special case) when nothing was
      removed at a given point (survival_rate == 1.0).

    Returns a dict with occ_raw, mag_raw, occ_post, mag_post, occ_removed, mag_removed
    (each length-L arrays) and survival_rate (fraction of shots kept by post-selection)."""
    N = NumberOfFockStates
    L = NumberOfBosonicModes

    measure_circuit = compiled_circuit + cirq.measure(*qubit_chain, key='m')
    result = sim.run(measure_circuit, repetitions=num_samples)
    bits = np.asarray(result.measurements['m'], dtype=int)  # (num_samples, total_qubits)

    boson_bits = bits[:, :L * N].reshape(num_samples, L, N)
    spin_bits = bits[:, L * N:L * N + L]

    onehot_per_mode = boson_bits.sum(axis=2) == 1  # (num_samples, L)
    valid_mask = onehot_per_mode.all(axis=1)
    survival_rate = float(valid_mask.mean())

    fock_index = np.arange(N)

    def estimate(mask):
        n_sel = int(mask.sum())
        if n_sel == 0:
            return np.full(L, np.nan), np.full(L, np.nan)
        occ = (boson_bits[mask].mean(axis=0) * fock_index[None, :]).sum(axis=1)
        mag = 1.0 - 2.0 * spin_bits[mask].mean(axis=0)
        return occ, mag

    occ_raw, mag_raw = estimate(np.ones(num_samples, dtype=bool))
    occ_post, mag_post = estimate(valid_mask)
    occ_removed, mag_removed = estimate(~valid_mask)

    return dict(occ_raw=occ_raw, mag_raw=mag_raw, occ_post=occ_post, mag_post=mag_post,
                occ_removed=occ_removed, mag_removed=mag_removed, survival_rate=survival_rate)


# ---------------------------------------------------------------------------
# Plotting helper: mark which NumberOfTrotterSteps value each cluster of sweep points
# used (UseAdaptiveTrotterSteps means this varies with t -- easy to lose track of by eye).
# ---------------------------------------------------------------------------

def annotate_trotter_step_segments(ax, time_data, trotter_steps_used, max_labels=14):
    """Draws a thin vertical divider at every point where the adaptive step count
    (trotter_steps_used) changes, and labels a readable subset of the resulting
    constant-step segments with "r=N" in the headroom above the data (the y-axis padding
    NuronSim.py's plotting already reserves, ~10% of the data range -- see ax1.set_ylim).

    Every change point gets a divider line; not every one gets a text label. The adaptive
    schedule (round(TROTTER_SCHEDULE_C * t)) changes roughly every 3 sweep points at
    NuronSim.py's default 100-point/8.25-time sweep, so labeling every single segment
    would overlap illegibly -- labels are thinned to at most `max_labels`, evenly spaced
    in time, while the divider lines (cheap, unobtrusive) still mark every transition.

    Returns the list of created artists (for cleanup on re-run, e.g. via
    _cirq_overlay_artists.extend(...) alongside the existing scatter-artist cleanup).
    """
    trotter_steps_used = np.asarray(trotter_steps_used)
    change_idx = np.where(np.diff(trotter_steps_used) != 0)[0] + 1
    segment_starts = np.concatenate(([0], change_idx))
    segment_ends = np.concatenate((change_idx, [len(time_data)]))  # exclusive

    artists = []
    trans = ax.get_xaxis_transform()  # x: data coords, y: axes-fraction coords
    for idx in change_idx:
        artists.append(ax.axvline(time_data[idx], color='gray', linestyle=':', linewidth=0.5, alpha=0.4, zorder=1))

    time_span = time_data[-1] - time_data[0]
    min_gap = time_span / max_labels if max_labels > 0 else 0.0
    last_labeled_t = -np.inf
    n_segments = len(segment_starts)
    for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        mid_t = (time_data[start] + time_data[min(end, len(time_data) - 1)]) / 2
        is_edge_segment = i == 0 or i == n_segments - 1
        if mid_t - last_labeled_t < min_gap and not is_edge_segment:
            continue
        steps_val = trotter_steps_used[start]
        artists.append(ax.text(mid_t, 0.97, f'r={steps_val}', transform=trans, ha='center', va='top',
                                fontsize=6, color='dimgray', alpha=0.85, rotation=90, zorder=1))
        last_labeled_t = mid_t

    return artists


def describe_trotter_run(UseNoiseModel, UsePostSelection, UseAdaptiveTrotterSteps, NumberOfTrotterSteps):
    """Describes a NuronSim.py / NeuronSim2ndOrderTrotter.py run configuration -- which
    error sources are active (noise, a low fixed Trotter step count) and which mitigations
    are active (the adaptive step schedule, D-18/D-19; post-selection, D-4/D-19) -- as a
    short filename tag (for non-clobbering output names when sweeping configurations) and
    a human-readable line (for the plot title). Post-selection is only meaningful when
    noise is on, so it's included in neither when UseNoiseModel is False.
    """
    tag_parts = ["noisy" if UseNoiseModel else "noiseless"]
    tag_parts.append("adaptive" if UseAdaptiveTrotterSteps else f"r{NumberOfTrotterSteps}")

    steps_title = "steps: adaptive" if UseAdaptiveTrotterSteps else f"steps: r={NumberOfTrotterSteps}"
    title = f"Noise {'ON' if UseNoiseModel else 'OFF'}  |  Trotter {steps_title}"

    if UseNoiseModel:
        tag_parts.append("postselect" if UsePostSelection else "nopostselect")
        title += f"  |  Post-select {'ON' if UsePostSelection else 'OFF'}"

    return "_".join(tag_parts), title
