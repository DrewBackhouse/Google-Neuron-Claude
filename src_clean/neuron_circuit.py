"""Hamiltonian / circuit / Willow-mapping building blocks for the neuron model
(first-order Trotter only).

This is the clean, "production" counterpart to src/neuron_circuit.py: it keeps every
function NuronSim.py's run path actually calls, using the current best-known config
(D-21's adaptive-step schedule, D-23's coherence-aware qubit embedding, D-4/D-19's
post-selection), and drops everything that only existed to *derive* those numbers
(the fidelity/observable-error schedule scans, the second-order/Strang circuit, the
noise-budget recomputation helpers) or that was superseded along the way. The full
derivation history for every constant below lives in DECISIONS.md and src/ in the
project root -- re-run the analysis there, not here, if the model config or the
Willow calibration snapshot changes.
"""
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, destroy, sesolve, expect, qeye, sigmax, sigmay, sigmaz, fock
from scipy.optimize import minimize_scalar
import cirq
import cirq_google

from G_analytic import G_area_theorem


# ---------------------------------------------------------------------------
# Physics: exact (qutip) Hamiltonian and G calibration
# ---------------------------------------------------------------------------

def QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient, SpinBosonInteractionCoefficent):
    sx_list, sy_list, sz_list = [], [], []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2), qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(sigmax(), qeye(NumberOfFockStates))
        sx_list.append(tensor(op_list))
        op_list[i] = tensor(sigmay(), qeye(NumberOfFockStates))
        sy_list.append(tensor(op_list))
        op_list[i] = tensor(sigmaz(), qeye(NumberOfFockStates))
        sz_list.append(tensor(op_list))

    a_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2), qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates))
        a_list.append(tensor(op_list))

    n_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2), qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates).dag() * destroy(NumberOfFockStates))
        n_list.append(tensor(op_list))

    n_max_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2), qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), fock(NumberOfFockStates, NumberOfFockStates - 1) * fock(NumberOfFockStates, NumberOfFockStates - 1).dag())
        n_max_list.append(tensor(op_list))

    H = sum(D_list[i] * (a_list[i] + a_list[i].dag()) + SpinBosonInteractionCoefficent * n_max_list[i] * sx_list[i] for i in range(NumberOfBosonicModes))
    H += sum(spin_interaction_coefficient * (sx_list[i] * sx_list[i + 1] + sy_list[i] * sy_list[i + 1]) / 2 for i in range(NumberOfBosonicModes - 1))

    return H, n_list, sz_list


def find_optimal_SpinBosonInteractionCoefficent(NumberOfFockStates, displacement_coefficient,
                                                  correction_factor=1.2, search_span=0.6, coarse_points=60,
                                                  fallback_bounds=(0.05 * np.pi, 4 * np.pi), fallback_points=400):
    """Calibrates G on an isolated single site (L=1) to give a full spin flip when the
    boson hits the Fock ceiling. Seeded from G_analytic.G_area_theorem's closed-form
    (first-order, no-back-action) estimate, scaled by an empirically stable ~1.2x
    back-action correction (notes/G-analytic-estimate.md), then locally refined -- falls
    back to the original full-range scan if the seeded window finds nothing near a full
    flip, so correctness never depends on the ratio holding everywhere."""
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


# ---------------------------------------------------------------------------
# Circuit construction (first-order Trotter: A(dt) B(dt) C(dt) per step)
# ---------------------------------------------------------------------------

def _boson_bond_exponent(displacement_coefficient, time_evolution, j):
    """ISwapPowGate exponent implementing exp(-i (D sqrt(j+1)/2)(X_jX_{j+1}+Y_jY_{j+1}) t)
    on the (j, j+1) bond."""
    return -2 * displacement_coefficient * time_evolution * (j + 1) ** 0.5 / np.pi


class BosonicDisplacementGate(cirq.Gate):
    """H_boson on one site's N Fock-state qubits: even bonds then odd bonds, one
    first-order sweep."""
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
            yield cirq.ISwapPowGate(exponent=_boson_bond_exponent(self.displacement_coefficient, self.time_evolution, j))(qubits[j], qubits[j + 1])
        for j in range(1, self.NumberOfFockStates - 1, 2):
            yield cirq.ISwapPowGate(exponent=_boson_bond_exponent(self.displacement_coefficient, self.time_evolution, j))(qubits[j], qubits[j + 1])


class SpinInteractionsGate(cirq.Gate):
    """H_spin (XY hopping) across the L spin qubits; empty at L=1 (no neighbour)."""
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
            yield cirq.ISwapPowGate(exponent=-2 * self.spin_interaction_coefficient * self.time_evolution / np.pi)(qubits[j], qubits[j + 1])
        for j in range(1, self.NumberOfBosonicModes - 1, 2):
            yield cirq.ISwapPowGate(exponent=-2 * self.spin_interaction_coefficient * self.time_evolution / np.pi)(qubits[j], qubits[j + 1])


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


def _spin_layer(spin_interaction_coefficient, NumberOfBosonicModes, time_evolution, spin_qubits):
    """H_spin layer."""
    spin_gate = SpinInteractionsGate(spin_interaction_coefficient, NumberOfBosonicModes, time_evolution)
    return cirq.Circuit(spin_gate(*spin_qubits))


def _cnot_layer(SpinBosonInteractionCoefficent, time_evolution, bosonic_control_qubits, spin_qubits):
    """H_CNOT layer: controlled-X rotation between each site's top boson qubit and its spin qubit."""
    circuit = cirq.Circuit()
    for i in range(len(bosonic_control_qubits)):
        circuit.append(cirq.rx(2 * time_evolution * SpinBosonInteractionCoefficent).controlled()(bosonic_control_qubits[i], spin_qubits[i]))
    return circuit


def NetworkOfNeuronsTrotterStep(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes, time_evolution, SpinBosonInteractionCoefficent):
    """One first-order Trotter step: A(dt) B(dt) C(dt) = H_boson, H_spin, H_CNOT."""
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


def build_trotter_circuit(D_list, spin_interaction_coefficient, NumberOfFockStates, NumberOfBosonicModes,
                           dt, NumberOfTrotterSteps, SpinBosonInteractionCoefficent):
    """Full logical circuit (vacuum state prep + NumberOfTrotterSteps repeats of a single
    dt-step), as cirq.LineQubits -- ready to map onto hardware qubits via
    map_and_compile_for_willow."""
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


def compute_observables_from_z_expectations(z_expectations, num_fock_states: int, num_bosonic_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Bosonic occupation and spin magnetization from single-qubit <Z_j> expectation
    values, ordered boson qubits (per mode) then spin qubits -- matching
    willow_qubit_chain. Both are sums of single-qubit marginals, so they never need the
    full joint distribution."""
    probs_1 = [(1.0 - z) / 2.0 for z in z_expectations]

    boson_occ = np.zeros(num_bosonic_modes)
    for mode_idx in range(num_bosonic_modes):
        start_q = mode_idx * num_fock_states
        boson_occ[mode_idx] = sum(j * probs_1[start_q + j] for j in range(num_fock_states))

    spin_qubit_start = num_bosonic_modes * num_fock_states
    spin_mag = np.array([
        1.0 - 2.0 * probs_1[spin_qubit_start + mode_idx] for mode_idx in range(num_bosonic_modes)
    ])
    return boson_occ, spin_mag


# ---------------------------------------------------------------------------
# Adaptive Trotter-step schedule (D-21): r*(t) = round(C * t), capped where the required
# depth exceeds the noise budget. Fit directly against noisy, post-selected observable
# error vs qutip (src/adaptive_trotter_model_scan.py + src/linear_trotter_finegrid_scan.py
# in the project's main src/), which is why it's linear rather than the t^2 a fixed-
# Trotter-tolerance bound alone would predict -- it also prices in that every extra step
# is a real noisy gate, not just Trotter error (D-17's derivation: minimizing
# eps_T + eps_N = A t^2/r + B r gives r* = t*sqrt(A/B), linear in t).
#
# TROTTER_SCHEDULE_R_MAX is a (1-eps_2q)^(gates_per_step * r) >= 0.5 noise budget against
# willow_pink's calibrated two-qubit error on the compiled circuit -- a property of the
# circuit/hardware, not of how C was fit.
#
# Specific to the config below (G calibrates to ~2.03) against one willow_pink median
# calibration snapshot -- not a general law. _SCHEDULE_CONFIG lets callers check they
# still match it.
# ---------------------------------------------------------------------------
_SCHEDULE_CONFIG = dict(NumberOfBosonicModes=1, NumberOfFockStates=5, D_list=(1.0,), spin_interaction_coefficient=0.5)
TROTTER_SCHEDULE_C = 2.25
TROTTER_SCHEDULE_R_MAX = 33
TROTTER_SCHEDULE_T_CAP = TROTTER_SCHEDULE_R_MAX / TROTTER_SCHEDULE_C


def check_trotter_schedule_config(NumberOfBosonicModes, NumberOfFockStates, D_list, spin_interaction_coefficient):
    """Warns (does not raise) if the current model config doesn't match the one
    recommended_trotter_steps was derived for."""
    current = dict(NumberOfBosonicModes=NumberOfBosonicModes, NumberOfFockStates=NumberOfFockStates,
                    D_list=tuple(D_list), spin_interaction_coefficient=spin_interaction_coefficient)
    if current != _SCHEDULE_CONFIG:
        print(f"WARNING: recommended_trotter_steps was derived for {_SCHEDULE_CONFIG}, "
              f"but the current config is {current}. The schedule likely no longer reflects "
              f"the true error-minimizing step counts -- re-run src/adaptive_trotter_model_scan.py.")


def recommended_trotter_steps(t: float) -> int:
    """NumberOfTrotterSteps for time t: the monotone schedule r*(t) = round(TROTTER_SCHEDULE_C * t).
    Does not itself enforce the noise-budget cap (TROTTER_SCHEDULE_T_CAP) -- callers
    sweeping a t range should check that separately (trotter_schedule_cap_message)."""
    return max(round(TROTTER_SCHEDULE_C * t), 1)


def trotter_schedule_cap_message(requested_time: float) -> str | None:
    """Returns a one-line warning if requested_time exceeds TROTTER_SCHEDULE_T_CAP (the
    noise-budget cap), else None. Informational only -- callers decide whether to act on
    it (NuronSim.py in this folder prints it but does not truncate the sweep)."""
    if requested_time > TROTTER_SCHEDULE_T_CAP:
        return (f"WARNING: requested sweep time {requested_time:.2f} exceeds the Trotter "
                f"schedule's noise-budget cap (t_cap={TROTTER_SCHEDULE_T_CAP:.2f}, "
                f"r_max={TROTTER_SCHEDULE_R_MAX}) -- beyond this point the required step "
                f"count is not affordable under the noise budget, so results are not "
                f"Trotter-converged at any reachable step count.")
    return None


# ---------------------------------------------------------------------------
# Willow qubit embedding (D-23): coherence-aware, SWAP-free placement search.
# ---------------------------------------------------------------------------

def required_adjacency_edges(NumberOfFockStates: int, NumberOfBosonicModes: int) -> set:
    """Logical-qubit pairs (cirq.LineQubit, matching build_trotter_circuit's ordering)
    that need a two-qubit gate somewhere in one Trotter step. Derived by actually
    building and decomposing NetworkOfNeuronsTrotterStep with dummy coefficients and
    reading off its two-qubit operations, so this can never drift out of sync with the
    real circuit builder.

    At NumberOfBosonicModes==1 this is a simple chain. At NumberOfBosonicModes>1 it's a
    comb: L separate site-chains of N boson qubits, joined only through a spine of
    spin-spin links at one end -- see find_low_error_qubit_embedding."""
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
    """Finds a low-error placement of the logical circuit onto real device qubits such
    that every required two-qubit gate lands on an actual device edge -- SWAP-free by
    construction (D-1). Returns a list ordered to match LineQubit.range(total_qubits),
    ready for map_and_compile_for_willow.

    Backtracking search over the comb-shaped logical topology (a straight-chain search
    is only correct at NumberOfBosonicModes<=1; the comb has no Hamiltonian path once
    NumberOfBosonicModes>=3). Repeated once per candidate starting device qubit (~100,
    each fast) and scored across all successful attempts, rather than keeping the first
    valid placement found -- a single-attempt search can land the H_CNOT edge on a
    much-higher-error link purely by bad luck in the search order.

    The score combines two calibration signals, each converted to a percentile rank
    (0=best, 1=worst) so they're on comparable, unit-free scales before summing:
    two-qubit CZ XEB Pauli error per required edge, and T1-driven idle decoherence
    (1/T1) per qubit used. Two-qubit error alone is not enough -- a low-total-edge-error
    embedding can still route through a single low-T1 qubit and measurably underperform,
    a coherence problem gate-error scoring alone can't see. Even the percentile score is
    a soft trade-off (a bad-enough-but-not-worst qubit can still get outweighed by good
    neighbouring edges), so there's also a hard floor: any device qubit below the 10th
    percentile of chip-wide T1 is excluded from candidacy entirely, before scoring.

    Raises if no embedding is found at all."""
    edges = required_adjacency_edges(NumberOfFockStates, NumberOfBosonicModes)
    total_qubits = NumberOfBosonicModes * NumberOfFockStates + NumberOfBosonicModes
    logical_qubits = cirq.LineQubit.range(total_qubits)

    neighbor_map = {q: set() for q in logical_qubits}
    for a, b in edges:
        neighbor_map[a].add(b)
        neighbor_map[b].add(a)

    # Placement order: BFS from the qubit with the most logical neighbours, so every
    # qubit placed after the first already has a placed neighbour to anchor the
    # device-adjacency search against.
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

    T1_FLOOR_PERCENTILE = 10
    t1_floor = np.percentile([t1_micros[(q,)][0] for q in graph.nodes() if (q,) in t1_micros], T1_FLOOR_PERCENTILE)
    usable_qubits = {q for q in graph.nodes() if t1_micros.get((q,), [0.0])[0] >= t1_floor}

    device_qubits_by_quality = sorted(usable_qubits, key=mean_neighbour_error)

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
    """Willow calibration grid (T1 per qubit, two-qubit CZ error per bond) with the
    specific qubits and edges THIS run's embedding (find_low_error_qubit_embedding)
    actually uses highlighted in red on top -- lets the chosen device patch be checked
    against the chip-wide calibration at a glance.

    used_edges is derived from required_adjacency_edges mapped through
    willow_qubit_chain, not just consecutive chain entries -- correct at
    NumberOfBosonicModes>1 too, where the logical topology is a comb.

    Returns the created Figure; caller saves/shows/closes it."""
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


# ---------------------------------------------------------------------------
# Post-selection on the unary constraint (D-4): discard shots where any boson column
# isn't exactly one-hot -- the detectable signature of bit-flip/amplitude-damping-type
# noise leaving the physical subspace.
# ---------------------------------------------------------------------------

def sample_shots_with_postselection(sim, compiled_circuit, qubit_chain, NumberOfFockStates, NumberOfBosonicModes, num_samples):
    """Runs compiled_circuit num_samples times measuring every qubit (qubit_chain order:
    boson qubits per mode, then spin qubits), then computes bosonic occupation and spin
    magnetization three ways from the same shots:

    - *_raw: marginal estimator over all shots.
    - *_post: restricted to shots where every mode's boson register is exactly one-hot
      (the physical/unary subspace) -- leaked shots are discarded rather than silently
      averaged in.
    - *_removed: the complementary estimator, over exactly the shots *_post discards --
      lets a caller plot what post-selection is actually throwing away. NaN when nothing
      was removed at a given point.

    Returns a dict with occ_raw, mag_raw, occ_post, mag_post, occ_removed, mag_removed
    (each length-L arrays) and survival_rate (fraction of shots kept)."""
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
# Plotting helpers
# ---------------------------------------------------------------------------

def annotate_trotter_step_segments(ax, time_data, trotter_steps_used, max_labels=14):
    """Draws a thin vertical divider at every point where the adaptive step count
    changes, and labels a readable subset of the resulting constant-step segments with
    "r=N" in the headroom above the data. Every change point gets a divider; labels are
    thinned to at most max_labels, evenly spaced in time, to avoid overlap.

    Returns the list of created artists (for cleanup on re-run)."""
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
    """Describes a run configuration as a short filename tag (non-clobbering output names
    across configs) and a human-readable line (for the plot title)."""
    tag_parts = ["noisy" if UseNoiseModel else "noiseless"]
    tag_parts.append("adaptive" if UseAdaptiveTrotterSteps else f"r{NumberOfTrotterSteps}")

    steps_title = "steps: adaptive" if UseAdaptiveTrotterSteps else f"steps: r={NumberOfTrotterSteps}"
    title = f"Noise {'ON' if UseNoiseModel else 'OFF'}  |  Trotter {steps_title}"

    if UseNoiseModel:
        tag_parts.append("postselect" if UsePostSelection else "nopostselect")
        title += f"  |  Post-select {'ON' if UsePostSelection else 'OFF'}"

    return "_".join(tag_parts), title
