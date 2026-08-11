"""Shared Hamiltonian / circuit / Willow-mapping building blocks for the neuron model.

Extracted from NuronSim.py (2026-08-10) so this logic can be reused by analysis scripts
(e.g. optimal_trotter_steps.py) without re-running NuronSim.py's own top-level sim + plot.
NuronSim.py imports from here; it no longer defines these itself.
"""
import itertools
import numpy as np
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


# Schedule derived by src/trotter_fidelity_schedule.py (2026-08-11) from noiseless state
# fidelity F(t,r) = |<psi_qutip(t)|psi_trotter(t,r)>|^2 (D-18) -- replaces D-16/D-17's
# two-observable-error schedule, which D-17 itself showed was contaminated by accidental
# crossings ("needles": 6 of 14 argmins were isolated lucky matches in two real numbers,
# not real convergence -- notes/optimal-trotter-steps.md). Monotone by construction
# (r*(t) = round(c*t)), so it cannot reproduce that failure mode.
#
# c is fitted from r_T(t) = smallest r with 1-F(t,r') < 0.05 sustained for all larger r',
# using the tail half of the tested t-range (t >= 8) -- the low-t points are quantization-
# noisy (steps_grid's coarse spacing near small r turns rounding into large swings in the
# r_T/t ratio; see the script's printed whole-range-vs-tail comparison). r_max is a naive
# (1 - eps_2q)^(gates_per_step * r) >= 0.5 budget against willow_pink's calibrated
# two-qubit error on the compiled circuit (D-18 stage 2, independent of t, D-15's
# mechanism); T_CAP = r_max / c is where the Trotter requirement first exceeds that
# budget -- D-17's "cap the sweep" recommendation, now derived from fidelity instead of
# asserted from an ad hoc "hard window".
#
# Specific to the config below (G calibrates to ~2.03) against one willow_pink median
# calibration snapshot -- re-run src/trotter_fidelity_schedule.py if any of these change,
# this is not a general law. _SCHEDULE_CONFIG lets callers check they still match it.
_SCHEDULE_CONFIG = dict(NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=(1.0,), spin_interaction_coefficient=0.5)
TROTTER_SCHEDULE_C = 4.0
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
    """NumberOfTrotterSteps for time t: the monotone fidelity-derived schedule
    r*(t) = round(TROTTER_SCHEDULE_C * t) (D-18, src/trotter_fidelity_schedule.py).
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
                f"Trotter-converged at any reachable step count (D-17/D-18).")
    return None


def find_low_error_qubit_chain(device: "cirq_google.GridDevice", calibration: "cirq_google.Calibration", chain_length: int) -> list:
    """Depth-first search for a connected chain of `chain_length` GridQubits on
    `device`, matching the circuit's own linear nearest-neighbour topology (every
    mapped Pauli string is nearest-neighbour on the L x (N+1) grid, PROJECT.md).
    Requiring the chain to follow device edges makes the embedding SWAP-free by
    construction (D-1).

    Greedily prefers the lowest-error two-qubit edges from the median calibration,
    since the unused qubits on a 105-qubit chip are worth more as fidelity headroom
    than as extra L/N (D-1 corollary) — this is a heuristic search for a good chain,
    not a proof of the globally lowest-error one.
    """
    graph = device.metadata.nx_graph
    two_qubit_error = calibration['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle']

    def edge_error(a, b):
        val = two_qubit_error.get((a, b)) or two_qubit_error.get((b, a))
        return val[0] if val is not None else 1.0  # uncalibrated edge: treat as worst-case

    def mean_neighbour_error(node):
        neighbours = list(graph.neighbors(node))
        return np.mean([edge_error(node, n) for n in neighbours]) if neighbours else 1.0

    for start in sorted(graph.nodes(), key=mean_neighbour_error):
        chain = [start]
        visited = {start}

        def extend():
            if len(chain) == chain_length:
                return True
            current = chain[-1]
            candidates = sorted(
                (n for n in graph.neighbors(current) if n not in visited),
                key=lambda n: edge_error(current, n),
            )
            for n in candidates:
                chain.append(n)
                visited.add(n)
                if extend():
                    return True
                chain.pop()
                visited.discard(n)
            return False

        if extend():
            return chain

    raise RuntimeError(f"No connected {chain_length}-qubit chain found on this device.")


def map_and_compile_for_willow(circuit: cirq.Circuit, source_qubits: list, qubit_chain: list, target_gateset) -> cirq.Circuit:
    """Relabels `circuit` from `source_qubits` (in order) onto the hardware `qubit_chain`,
    then compiles into Willow's native gateset (CZ + single-qubit gates)."""
    qubit_map = dict(zip(source_qubits, qubit_chain))
    mapped_circuit = circuit.transform_qubits(qubit_map)
    return cirq.optimize_for_target_gateset(mapped_circuit, gateset=target_gateset)


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
    calibration field find_low_error_qubit_chain optimizes against."""
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

    Returns a dict with occ_raw, mag_raw, occ_post, mag_post (each length-L arrays) and
    survival_rate (fraction of shots kept by post-selection)."""
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

    return dict(occ_raw=occ_raw, mag_raw=mag_raw, occ_post=occ_post, mag_post=mag_post,
                survival_rate=survival_rate)


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
