"""VM benchmark figure suite -> results/VMBenchmark/ (requested by Drew, 2026-08-12).

Three parts, run in order:

1. Four fixed configs at L=1, N=5, D=1.0, J=0.5 (NuronSim.py's own defaults) -- a
   step-count/noise/post-selection ablation:
     a. 100 fixed Trotter steps, no noise, no post-selection
     b. 7 fixed Trotter steps, noise, no post-selection
     c. Optimised adaptive Trotter steps (recommended_trotter_steps, D-21/D-19 c=2.25),
        noise, no post-selection
     d. Optimised adaptive Trotter steps, noise, post-selection
   (c) and (d) share one noisy sample run -- sample_shots_with_postselection always
   computes both the raw and post-selected estimator from the same shots (D-19), so there
   is no need to sample twice just to look at both.

2. Optimised adaptive schedule (the SAME global c=2.25 schedule, reused as-is -- this is
   what "Optimised Adaptive" already means throughout the codebase) + noise +
   post-selection, N=5, J=0.1, sweeping NumberOfBosonicModes (L) = 1, 2, 3. Uses
   find_low_error_qubit_embedding (this session's fix) since L=2,3 need a genuine 2D
   embedding, not a straight chain. check_trotter_schedule_config's mismatch warning is
   EXPECTED at L=2,3 (the schedule is fit for L=1) and is not treated as an error, exactly
   matching what NuronSim.py itself would print if you changed NumberOfBosonicModes and
   reran it.

3. A fresh, per-N schedule fit (L=1, D=1.0) at N=3..7, since the D-21 schedule constant
   was fit for N=5 specifically and the Hamiltonian (hence G, hence the whole dynamics)
   genuinely changes with N. For each N: a reduced-resolution coarse-then-refine linear
   (r=a*t) scan (same method as D-21/src/adaptive_trotter_model_scan.py, cheaper grid --
   this is 5 separate schedule fits, not one, so full D-21 resolution at every N would
   cost several hours) against noisy, post-selected error vs qutip, then a full-resolution
   production run+plot at that N's fitted coefficient.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/vm_benchmark.py
(GoogleQVM env -- qsimcirq/cirq_google, D-14)
Pass --smoke-test for a fast, tiny-grid sanity run of every code path before the full run.
Pass --part1 / --part2 / --part3 (any combination) to run only those parts -- e.g. after a
change to find_low_error_qubit_embedding, to re-check one part without re-running all
three (Part 2's L=3 case alone takes ~3 hours). With none of these flags, all three run.
"""
import os
import sys
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
    compute_observables_from_z_expectations,
    find_low_error_qubit_embedding,
    map_and_compile_for_willow,
    build_trotter_circuit,
    recommended_trotter_steps,
    check_trotter_schedule_config,
    trotter_schedule_cap_message,
    sample_shots_with_postselection,
    annotate_trotter_step_segments,
    two_qubit_error_rate_for_chain,
    two_qubit_gates_per_trotter_step,
    estimate_max_affordable_trotter_steps,
    TROTTER_SCHEDULE_T_CAP,
)

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "VMBenchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOKE_TEST = "--smoke-test" in sys.argv
_part_flags = {"--part1", "--part2", "--part3"} & set(sys.argv)
RUN_PART1 = not _part_flags or "--part1" in _part_flags
RUN_PART2 = not _part_flags or "--part2" in _part_flags
RUN_PART3 = not _part_flags or "--part3" in _part_flags

if SMOKE_TEST:
    TIME_DEFAULT, N_TIME_POINTS, N_SHOTS = 3.0, 3, 100
    SCHEDULE_COARSE_A = [1.0, 2.0]
    SCHEDULE_COARSE_T_POINTS, SCHEDULE_COARSE_SHOTS = 3, 50
    SCHEDULE_REFINE_T_POINTS, SCHEDULE_REFINE_SHOTS = 3, 50
    N_SWEEP = [3, 5]
else:
    TIME_DEFAULT, N_TIME_POINTS, N_SHOTS = 10.0, 50, 2000
    SCHEDULE_COARSE_A = [0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8]
    SCHEDULE_COARSE_T_POINTS, SCHEDULE_COARSE_SHOTS = 15, 500
    SCHEDULE_REFINE_T_POINTS, SCHEDULE_REFINE_SHOTS = 25, 800
    N_SWEEP = [3, 4, 5, 6, 7]
    # --n=6,7 restricts Part 3 to specific N values -- e.g. to re-run only the configs
    # affected by a fix without re-doing already-good ones (N=3,4,5 were already solid
    # after the T1-aware embedding fix; only N=6,7 needed the hard T1-floor follow-up).
    _n_arg = next((a for a in sys.argv if a.startswith("--n=")), None)
    if _n_arg is not None:
        N_SWEEP = [int(x) for x in _n_arg[len("--n="):].split(",")]

color_boson = '#d62728'
color_spin = '#1f77b4'

# ---------------------------------------------------------------------------
# Shared Willow device/calibration/noise-model objects -- static data, independent of
# the model config, so built once regardless of how many (N, L) combinations follow.
# ---------------------------------------------------------------------------
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
    """Runs the noisy/noiseless Cirq sweep. step_count_fn(t) -> r works for both fixed
    (lambda t: FIXED_R) and adaptive schedules. Always computes both raw and post-selected
    estimates on the noisy path (sample_shots_with_postselection, D-19) so callers can look
    at either without re-sampling."""
    noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model if UseNoiseModel else None)
    z_observables = [cirq.Z(q) for q in willow_qubit_chain]

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

        if UseNoiseModel:
            shots = sample_shots_with_postselection(
                noisy_sim, compiled_circuit, willow_qubit_chain, NumberOfFockStates, NumberOfBosonicModes,
                NumberOfNoiseSamples
            )
            occ_post[i], mag_post[i] = shots['occ_post'], shots['mag_post']
            occ_raw[i], mag_raw[i] = shots['occ_raw'], shots['mag_raw']
            survival[i] = shots['survival_rate']
        else:
            z_expectations = [z.real for z in noisy_sim.simulate_expectation_values(compiled_circuit, observables=z_observables)]
            occ, mag = compute_observables_from_z_expectations(z_expectations, NumberOfFockStates, NumberOfBosonicModes)
            occ_post[i] = occ_raw[i] = occ
            mag_post[i] = mag_raw[i] = mag
            survival[i] = 1.0

    return dict(time_data=time_data, occ_post=occ_post, mag_post=mag_post, occ_raw=occ_raw, mag_raw=mag_raw,
                survival=survival, steps=steps_used)


def plot_and_save(qutip_times, qutip_occ, qutip_mag, sweep, NumberOfBosonicModes, use_post_select,
                   UseNoiseModel, title, output_path, cap_t=None):
    use_post = use_post_select and UseNoiseModel
    boson = sweep['occ_post'] if use_post else sweep['occ_raw']
    spin = sweep['mag_post'] if use_post else sweep['mag_raw']

    fig, axes = plt.subplots(nrows=1, ncols=NumberOfBosonicModes, figsize=(6.5 * NumberOfBosonicModes, 4))
    if NumberOfBosonicModes == 1:
        axes = [axes]

    for mode_idx in range(NumberOfBosonicModes):
        ax1 = axes[mode_idx]
        ax1.plot(qutip_times, qutip_occ[mode_idx], color=color_boson, linewidth=1.5, label='QuTiP occupation')
        ax1.set_xlabel('Time', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_boson)
        if mode_idx == 0:
            ax1.set_ylabel('Average Bosonic Occupation number', color=color_boson, fontweight='bold')

        ax2 = ax1.twinx()
        ax2.plot(qutip_times, qutip_mag[mode_idx], color=color_spin, linewidth=1.5, label='QuTiP magnetisation')
        ax2.tick_params(axis='y', labelcolor=color_spin)
        if mode_idx == NumberOfBosonicModes - 1:
            ax2.set_ylabel('Average Spin Magnetisation', color=color_spin, fontweight='bold')

        label = 'Cirq (Willow, noisy, post-selected)' if use_post else \
                ('Cirq (Willow, noisy, NOT post-selected)' if UseNoiseModel else 'Cirq (Willow, noiseless)')
        ax1.scatter(sweep['time_data'], boson[:, mode_idx], color=color_boson, marker='o',
                    edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{label} – occupation')
        ax2.scatter(sweep['time_data'], spin[:, mode_idx], color=color_spin, marker='s',
                    edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{label} – magnetisation')

        boson_max = max(np.max(qutip_occ[mode_idx]), 1e-9)
        ax1.set_ylim(-0.10 * boson_max, 1.10 * boson_max)
        ax2.set_ylim(-1.10, 1.10)
        if NumberOfBosonicModes > 1:
            ax1.set_title(f"Mode {mode_idx}", fontweight='bold', pad=10)

        if cap_t is not None and cap_t < sweep['time_data'].max():
            ax1.axvline(cap_t, color='gray', linestyle=':', linewidth=1, zorder=1, label=f'schedule cap (t={cap_t:.2f})')
        annotate_trotter_step_segments(ax1, sweep['time_data'], sweep['steps'])

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')

    # Split "config | run description" onto two lines (matching NuronSim.py's own
    # suptitle + fig.text convention) -- a single line long enough to include both halves
    # overflows the figure width at NumberOfBosonicModes==1's narrower size.
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
    if UseNoiseModel:
        print(f"    survival: mean={np.nanmean(sweep['survival']):.1%}, "
              f"min={np.nanmin(sweep['survival']):.1%}, max={np.nanmax(sweep['survival']):.1%}")
    print(f"    Saved {output_path}.png")


# ===========================================================================
# PART 1: L=1, N=5, D=1.0, J=0.5 -- step-count/noise/post-selection ablation
# ===========================================================================
if RUN_PART1:
    print("\n" + "=" * 70)
    print("PART 1: L=1, N=5 ablation (100 steps noiseless / 7 steps noisy / adaptive x2)")
    print("=" * 70)

    P1_L, P1_N, P1_J = 1, 5, 0.5
    P1_D_list = [1.0 * (0.85 ** j) for j in range(P1_L)]

    print("Calibrating G and solving the qutip ground truth...")
    P1_G = find_optimal_SpinBosonInteractionCoefficent(P1_N, P1_D_list[0])
    print(f"G = {P1_G:.4g}")
    p1_qtimes, p1_qocc, p1_qmag = qutip_ground_truth(P1_L, P1_N, P1_D_list, P1_J, P1_G, TIME_DEFAULT)

    p1_chain = find_low_error_qubit_embedding(willow_device, willow_calibration, P1_N, P1_L)
    print(f"Mapped {P1_L * P1_N + P1_L} qubits onto Willow: " + " - ".join(str(q) for q in p1_chain))

    t0 = time.time()
    print("\n(a) 100 fixed Trotter steps, no noise, no post-selection")
    sweep_a = simulate_sweep(P1_D_list, P1_J, P1_N, P1_L, P1_G, p1_chain, TIME_DEFAULT, N_TIME_POINTS, N_SHOTS,
                              UseNoiseModel=False, step_count_fn=lambda t: 100)
    plot_and_save(p1_qtimes, p1_qocc, p1_qmag, sweep_a, P1_L, use_post_select=False, UseNoiseModel=False,
                  title=f"N={P1_N}, G={P1_G:.3g}, J={P1_J}  |  100 fixed Trotter steps, no noise, no post-selection",
                  output_path=os.path.join(OUTPUT_DIR, "01_100steps_noiseless"))
    print(f"    ({time.time()-t0:.0f}s elapsed)")

    t0 = time.time()
    print("\n(b) 7 fixed Trotter steps, noise ON, no post-selection")
    sweep_b = simulate_sweep(P1_D_list, P1_J, P1_N, P1_L, P1_G, p1_chain, TIME_DEFAULT, N_TIME_POINTS, N_SHOTS,
                              UseNoiseModel=True, step_count_fn=lambda t: 7)
    plot_and_save(p1_qtimes, p1_qocc, p1_qmag, sweep_b, P1_L, use_post_select=False, UseNoiseModel=True,
                  title=f"N={P1_N}, G={P1_G:.3g}, J={P1_J}  |  7 fixed Trotter steps, noise ON, no post-selection",
                  output_path=os.path.join(OUTPUT_DIR, "02_7steps_noisy"))
    print(f"    ({time.time()-t0:.0f}s elapsed)")

    t0 = time.time()
    print("\n(c)+(d) Optimised adaptive Trotter steps, noise ON -- one sample run, two plots (raw vs post-selected)")
    check_trotter_schedule_config(P1_L, P1_N, P1_D_list, P1_J)
    cap_msg = trotter_schedule_cap_message(TIME_DEFAULT)
    p1_time_cd = TIME_DEFAULT
    if cap_msg is not None:
        print(cap_msg)
        p1_time_cd = TROTTER_SCHEDULE_T_CAP
    sweep_cd = simulate_sweep(P1_D_list, P1_J, P1_N, P1_L, P1_G, p1_chain, p1_time_cd, N_TIME_POINTS, N_SHOTS,
                               UseNoiseModel=True, step_count_fn=recommended_trotter_steps)
    plot_and_save(p1_qtimes, p1_qocc, p1_qmag, sweep_cd, P1_L, use_post_select=False, UseNoiseModel=True,
                  title=f"N={P1_N}, G={P1_G:.3g}, J={P1_J}  |  Optimised adaptive Trotter steps, noise ON, no post-selection",
                  output_path=os.path.join(OUTPUT_DIR, "03_adaptive_noisy_nopostselect"), cap_t=TROTTER_SCHEDULE_T_CAP)
    plot_and_save(p1_qtimes, p1_qocc, p1_qmag, sweep_cd, P1_L, use_post_select=True, UseNoiseModel=True,
                  title=f"N={P1_N}, G={P1_G:.3g}, J={P1_J}  |  Optimised adaptive Trotter steps, noise ON, post-selection ON",
                  output_path=os.path.join(OUTPUT_DIR, "04_adaptive_noisy_postselect"), cap_t=TROTTER_SCHEDULE_T_CAP)
    print(f"    ({time.time()-t0:.0f}s elapsed)")

    print("\nPART 1 COMPLETE.")

# ===========================================================================
# PART 2: N=5, J=0.1, optimised adaptive + noise + post-selection, L = 1, 2, 3
# ===========================================================================
if RUN_PART2:
    print("\n" + "=" * 70)
    print("PART 2: L sweep (1,2,3), N=5, J=0.1, optimised adaptive + noise + post-selection")
    print("=" * 70)

    P2_N, P2_J = 5, 0.1

    # G depends only on (N, D_list[0]=1.0), not on L or J (calibrated at the reference site,
    # reused across the chain -- same convention as NuronSim.py) -- compute once.
    P2_G = find_optimal_SpinBosonInteractionCoefficent(P2_N, 1.0)
    print(f"G (shared across L=1,2,3) = {P2_G:.4g}")

    for L in [1, 2, 3]:
        print(f"\n--- L={L} ---")
        D_list = [1.0 * (0.85 ** j) for j in range(L)]
        qtimes, qocc, qmag = qutip_ground_truth(L, P2_N, D_list, P2_J, P2_G, TIME_DEFAULT)

        chain = find_low_error_qubit_embedding(willow_device, willow_calibration, P2_N, L)
        print(f"Mapped {L * P2_N + L} qubits onto Willow: " + " - ".join(str(q) for q in chain))

        check_trotter_schedule_config(L, P2_N, D_list, P2_J)  # expected mismatch warning at L=2,3
        cap_msg = trotter_schedule_cap_message(TIME_DEFAULT)
        run_time = TIME_DEFAULT
        if cap_msg is not None:
            print(cap_msg)
            run_time = TROTTER_SCHEDULE_T_CAP

        t0 = time.time()
        sweep = simulate_sweep(D_list, P2_J, P2_N, L, P2_G, chain, run_time, N_TIME_POINTS, N_SHOTS,
                                UseNoiseModel=True, step_count_fn=recommended_trotter_steps)
        plot_and_save(qtimes, qocc, qmag, sweep, L, use_post_select=True, UseNoiseModel=True,
                      title=f"L={L}, N={P2_N}, G={P2_G:.3g}, J={P2_J}  |  Optimised adaptive Trotter steps, noise ON, post-selection ON",
                      output_path=os.path.join(OUTPUT_DIR, f"0{4+L}_L{L}_N{P2_N}_J{P2_J}_adaptive_noisy_postselect"),
                      cap_t=TROTTER_SCHEDULE_T_CAP)
        print(f"    ({time.time()-t0:.0f}s elapsed)")

    print("\nPART 2 COMPLETE.")

# ===========================================================================
# PART 3: N sweep (3..7), L=1 -- re-fit the adaptive linear schedule per N, then plot
# ===========================================================================
if RUN_PART3:
    print("\n" + "=" * 70)
    print(f"PART 3: N sweep {N_SWEEP}, L=1 -- re-fitting the adaptive schedule per N")
    print("=" * 70)

    P3_L = 1
    P3_D_list = [1.0]
    P3_J = 0.5  # inert at L=1 (H_spin's sum is empty, _spin_layer decomposes to nothing) -- any value works


    def combined_error(occ, mag, t, qutip_times, qutip_occ_curve, qutip_mag_curve, NumberOfFockStates):
        qocc = np.interp(t, qutip_times, qutip_occ_curve)
        qmag = np.interp(t, qutip_times, qutip_mag_curve)
        boson_err = (occ - qocc) / (NumberOfFockStates - 1)
        spin_err = (mag - qmag) / 2.0
        return float(np.sqrt(boson_err ** 2 + spin_err ** 2))


    def rms_error_for_a(a, N, G, chain, qutip_times, qutip_occ_curve, qutip_mag_curve, n_time_points, n_shots):
        time_data = np.linspace(0.5, TIME_DEFAULT, n_time_points)
        errs = np.zeros(n_time_points)
        noisy_sim = qsimcirq.QSimSimulator(noise=willow_noise_model)
        for i, t in enumerate(time_data):
            r = max(round(a * t), 1)
            dt = t / r
            full_circuit, all_qubits = build_trotter_circuit(P3_D_list, P3_J, N, P3_L, dt, r, G)
            compiled_circuit = map_and_compile_for_willow(full_circuit, all_qubits, chain, willow_target_gateset)
            shots = sample_shots_with_postselection(noisy_sim, compiled_circuit, chain, N, P3_L, n_shots)
            errs[i] = combined_error(shots['occ_post'][0], shots['mag_post'][0], t,
                                      qutip_times, qutip_occ_curve, qutip_mag_curve, N)
        return float(np.sqrt(np.mean(errs ** 2)))


    for N in N_SWEEP:
        print(f"\n--- N={N} ---")
        G = find_optimal_SpinBosonInteractionCoefficent(N, P3_D_list[0])
        print(f"G = {G:.4g}")
        qtimes, qocc, qmag = qutip_ground_truth(P3_L, N, P3_D_list, P3_J, G, TIME_DEFAULT)
        qocc_curve, qmag_curve = qocc[0], qmag[0]

        chain = find_low_error_qubit_embedding(willow_device, willow_calibration, N, P3_L)
        print(f"Mapped {P3_L * N + P3_L} qubits onto Willow: " + " - ".join(str(q) for q in chain))

        # --- Coarse scan ---
        t0 = time.time()
        print(f"Coarse scan over a = {SCHEDULE_COARSE_A} ({SCHEDULE_COARSE_T_POINTS} t-points, {SCHEDULE_COARSE_SHOTS} shots/point)...")
        coarse_rms = []
        for a in SCHEDULE_COARSE_A:
            rms = rms_error_for_a(a, N, G, chain, qtimes, qocc_curve, qmag_curve, SCHEDULE_COARSE_T_POINTS, SCHEDULE_COARSE_SHOTS)
            coarse_rms.append(rms)
            print(f"  a={a:g}: RMS={rms:.4f}  ({time.time()-t0:.0f}s elapsed)")
        coarse_best_a = SCHEDULE_COARSE_A[int(np.argmin(coarse_rms))]
        print(f"Coarse winner: a={coarse_best_a:g}")

        # --- Local refine around the coarse winner ---
        refine_a = sorted({round(coarse_best_a + d, 4) for d in (-0.5, 0, 0.5) if coarse_best_a + d > 0})
        print(f"Refine scan over a = {refine_a} ({SCHEDULE_REFINE_T_POINTS} t-points, {SCHEDULE_REFINE_SHOTS} shots/point)...")
        refine_rms = []
        for a in refine_a:
            rms = rms_error_for_a(a, N, G, chain, qtimes, qocc_curve, qmag_curve, SCHEDULE_REFINE_T_POINTS, SCHEDULE_REFINE_SHOTS)
            refine_rms.append(rms)
            print(f"  a={a:g}: RMS={rms:.4f}  ({time.time()-t0:.0f}s elapsed)")
        best_a = refine_a[int(np.argmin(refine_rms))]
        print(f"N={N} fitted schedule: a={best_a:g} (RMS={min(refine_rms):.4f})")

        # --- Schedule-fit diagnostic plot ---
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(SCHEDULE_COARSE_A, coarse_rms, marker='o', color='tab:gray', alpha=0.6, label='coarse scan')
        ax.plot(refine_a, refine_rms, marker='o', color='tab:green', label='refine scan')
        ax.axvline(best_a, color='tab:red', linestyle='--', label=f'fitted a={best_a:g}')
        ax.set_xlabel('a')
        ax.set_ylabel('RMS combined error vs qutip')
        ax.set_title(f'N={N}: adaptive schedule re-fit (r=a*t)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"schedule_fit_N{N}.png"), dpi=130)
        plt.close(fig)

        # --- Noise-budget cap for this N's fitted schedule ---
        gates_per_step = two_qubit_gates_per_trotter_step(P3_D_list, P3_J, N, P3_L, G, chain, willow_target_gateset)
        mean_two_qubit_error = two_qubit_error_rate_for_chain(willow_calibration, chain)
        r_max = estimate_max_affordable_trotter_steps(gates_per_step, mean_two_qubit_error, 0.5)
        t_cap = r_max / best_a if best_a > 0 else TIME_DEFAULT
        run_time = TIME_DEFAULT
        if run_time > t_cap:
            print(f"WARNING: requested time {run_time} exceeds this N's noise-budget cap "
                  f"(t_cap={t_cap:.2f}, r_max={r_max}). Capping sweep at t_cap.")
            run_time = t_cap

        # --- Production run + plot at the fitted schedule ---
        t0 = time.time()
        sweep = simulate_sweep(P3_D_list, P3_J, N, P3_L, G, chain, run_time, N_TIME_POINTS, N_SHOTS,
                                UseNoiseModel=True, step_count_fn=lambda t, a=best_a: round(a * t))
        plot_and_save(qtimes, qocc, qmag, sweep, P3_L, use_post_select=True, UseNoiseModel=True,
                      title=f"N={N}, G={G:.3g}  |  Optimised adaptive Trotter steps (re-fit: a={best_a:g}), noise ON, post-selection ON",
                      output_path=os.path.join(OUTPUT_DIR, f"N{N}_adaptive_noisy_postselect"), cap_t=t_cap)
        print(f"    ({time.time()-t0:.0f}s elapsed)")

    print("\nPART 3 COMPLETE.")
print(f"\nAll figures saved to {OUTPUT_DIR}")
