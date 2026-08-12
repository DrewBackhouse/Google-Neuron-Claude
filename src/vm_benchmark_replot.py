"""One-off: regenerate VMBenchmark PNGs from their already-saved .npz data, using the
fixed two-line title layout (vm_benchmark.py's plot_and_save previously crammed the whole
title onto one suptitle line, which overflowed the figure width at NumberOfBosonicModes=1's
narrower size -- e.g. the N7 plot's title was clipped on both edges). No re-simulation
needed -- every run's raw sweep + qutip curve is already in its .npz.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from neuron_circuit import annotate_trotter_step_segments

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "VMBenchmark")

color_boson = '#d62728'
color_spin = '#1f77b4'

# (npz filename without extension, title line 1, title line 2, use_post_select, UseNoiseModel)
FIGURES = [
    ("01_100steps_noiseless", "N=5, G=2.03, J=0.5", "100 fixed Trotter steps, no noise, no post-selection", False, False),
    ("02_7steps_noisy", "N=5, G=2.03, J=0.5", "7 fixed Trotter steps, noise ON, no post-selection", False, True),
    ("03_adaptive_noisy_nopostselect", "N=5, G=2.03, J=0.5", "Optimised adaptive Trotter steps, noise ON, no post-selection", False, True),
    ("04_adaptive_noisy_postselect", "N=5, G=2.03, J=0.5", "Optimised adaptive Trotter steps, noise ON, post-selection ON", True, True),
    ("05_L1_N5_J0.1_adaptive_noisy_postselect", "L=1, N=5, G=2.03, J=0.1", "Optimised adaptive Trotter steps, noise ON, post-selection ON", True, True),
    ("06_L2_N5_J0.1_adaptive_noisy_postselect", "L=2, N=5, G=2.03, J=0.1", "Optimised adaptive Trotter steps, noise ON, post-selection ON", True, True),
    ("07_L3_N5_J0.1_adaptive_noisy_postselect", "L=3, N=5, G=2.03, J=0.1", "Optimised adaptive Trotter steps, noise ON, post-selection ON", True, True),
    ("N3_adaptive_noisy_postselect", "N=3, G=1.557", "Optimised adaptive Trotter steps (re-fit: a=2.5), noise ON, post-selection ON", True, True),
    ("N4_adaptive_noisy_postselect", "N=4, G=1.801", "Optimised adaptive Trotter steps (re-fit: a=2.5), noise ON, post-selection ON", True, True),
    ("N5_adaptive_noisy_postselect", "N=5, G=2.029", "Optimised adaptive Trotter steps (re-fit: a=1.5), noise ON, post-selection ON", True, True),
    ("N6_adaptive_noisy_postselect", "N=6, G=2.28", "Optimised adaptive Trotter steps (re-fit: a=2.5), noise ON, post-selection ON", True, True),
    ("N7_adaptive_noisy_postselect", "N=7, G=2.472", "Optimised adaptive Trotter steps (re-fit: a=2.5), noise ON, post-selection ON", True, True),
]


def replot(name, title1, title2, use_post_select, UseNoiseModel):
    data = np.load(os.path.join(OUTPUT_DIR, f"{name}.npz"))
    qutip_times, qutip_occ, qutip_mag = data['qutip_time'], data['qutip_occ'], data['qutip_mag']
    time_data, steps = data['time_data'], data['steps']
    use_post = use_post_select and UseNoiseModel
    boson = data['occ_post'] if use_post else data['occ_raw']
    spin = data['mag_post'] if use_post else data['mag_raw']
    NumberOfBosonicModes = qutip_occ.shape[0]

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
        ax1.scatter(time_data, boson[:, mode_idx], color=color_boson, marker='o',
                    edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{label} – occupation')
        ax2.scatter(time_data, spin[:, mode_idx], color=color_spin, marker='s',
                    edgecolors='black', linewidths=0.5, alpha=0.8, zorder=5, label=f'{label} – magnetisation')

        boson_max = max(np.max(qutip_occ[mode_idx]), 1e-9)
        ax1.set_ylim(-0.10 * boson_max, 1.10 * boson_max)
        ax2.set_ylim(-1.10, 1.10)
        if NumberOfBosonicModes > 1:
            ax1.set_title(f"Mode {mode_idx}", fontweight='bold', pad=10)

        annotate_trotter_step_segments(ax1, time_data, steps)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')

    fig.suptitle(title1, fontweight='bold', fontsize=11, y=0.98)
    fig.text(0.5, 0.90, title2, ha='center', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=130)
    plt.close(fig)
    print(f"Regenerated {name}.png")


for name, title1, title2, use_post_select, UseNoiseModel in FIGURES:
    replot(name, title1, title2, use_post_select, UseNoiseModel)

print(f"\nDone -- {len(FIGURES)} figures regenerated in {OUTPUT_DIR}")
