"""Plots the willow_pink qubit grid: T1 per qubit and two-qubit CZ gate error per bond,
from the same static calibration snapshot find_low_error_qubit_embedding uses (2026-08-13,
requested by Drew) -- see cirq_google.Calibration.heatmap, which auto-picks a per-qubit or
per-edge heatmap based on the metric's key arity.

The two-qubit error panel is built by hand rather than via calibration.heatmap(): the raw
metric is a fraction (~0.0005-0.012), which as annotation text either shows all-but-identical
leading "0.00.." digits or gets truncated in the small bond hexagons. Scaling by 100 and
labelling the colorbar "%" (rather than repeating a percent sign on all 182 bonds) keeps each
label to 4 characters and legible at this qubit count.

Run: /opt/miniconda3/envs/GoogleQVM/bin/python3 src/willow_calibration_grid.py
"""
import matplotlib.pyplot as plt
import cirq
import cirq_google

RESULTS_DIR = "/Users/drewbackhouse/Documents/Claude/Google Neuron/results"

calibration = cirq_google.engine.load_median_device_calibration("willow_pink")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
calibration.heatmap("single_qubit_idle_t1_micros").plot(ax1)

two_qubit_error_pct = {
    calibration.key_to_qubits(key): calibration.value_to_float(value) * 100
    for key, value in calibration["two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle"].items()
}
cirq.TwoQubitInteractionHeatmap(
    two_qubit_error_pct,
    title="Two Qubit Parallel Cz Gate Xeb Pauli Error Per Cycle",
    annotation_format=".2f",
    annotation_text_kwargs={"fontsize": 7},
    colorbar_options={"label": "%"},
).plot(ax2)

fig.suptitle(f"willow_pink calibration snapshot ({calibration.timestamp_str()})", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

output_path = f"{RESULTS_DIR}/willow_calibration_grid.png"
fig.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
