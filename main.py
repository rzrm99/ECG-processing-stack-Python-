# from ecg_hrv_pipeline_physionet import ecg_hrv_pipeline_physionet
#
# # Example run
# ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='light', powerline_freq=60)
#
# # or
# # ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='dark', powerline_freq=60)
#
#
#




#####################  mitdb/100
# from dataclasses import asdict
# from ecg_hrv_pipeline_physionet import (
#     ecg_hrv_pipeline_physionet,
#     evaluate_against_annotations,
#     hrv_from_reference_annotations,
# )
#
# # Run pipeline
# metrics, locs_R, fs = ecg_hrv_pipeline_physionet(
#     'mitdb/100', dur_s=180, theme='light', powerline_freq=60
# )
#
# # 1) R-peak accuracy vs ground truth (±50 ms)
# eval_res = evaluate_against_annotations('mitdb/100', locs_R, fs, tol_ms=50)
# print("\n--- R-peak Detection Accuracy ---")
# for k, v in eval_res.items():
#     print(f"{k}: {v}")
#
# # 2) HRV vs reference annotations
# ref_metrics = hrv_from_reference_annotations('mitdb/100', fs, dur_s=180)
# print("\n--- HRV (ours vs reference) ---")
# for key in ["meanNN_ms","sdnn_ms","rmssd_ms","pnn50","lfnu","hfnu","SD1_ms","SD2_ms"]:
#     ours = metrics[key]
#     refv = asdict(ref_metrics)[key]
#     diff = ours - refv
#     rel = (diff / refv * 100.0) if refv not in (0, float('nan')) else float('nan')
#     print(f"{key}: ours={ours:.2f}, ref={refv:.2f}, Δ={diff:.2f} ({rel:.2f} %)")



####################  batch "mitdb/100", "mitdb/101", "mitdb/103", "mitdb/105" ###########
###################################################################################

from dataclasses import asdict
from ecg_hrv_pipeline_physionet import (
    ecg_hrv_pipeline_physionet,
    evaluate_against_annotations,
    hrv_from_reference_annotations,
)

import numpy as np
import csv

records = ["mitdb/100", "mitdb/101", "mitdb/103", "mitdb/105"]
tol_ms = 50  # matching tolerance for evaluation

rows = []
print(f"{'Record':<10} {'Se':>6} {'PPV':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'ΔmeanNN(ms)':>12} {'ΔSDNN(ms)':>10}")
print("-" * 70)

for rec in records:
    # run pipeline (returns metrics dict, detected R locations, and fs)
    metrics, locs_R, fs = ecg_hrv_pipeline_physionet(
        rec, dur_s=180, theme='light', powerline_freq=60
    )

    # evaluate QRS detection vs annotations
    ev = evaluate_against_annotations(rec, locs_R, fs, tol_ms=tol_ms)

    # HRV vs. reference
    refm = asdict(hrv_from_reference_annotations(rec, fs, dur_s=180))
    d_mean = metrics["meanNN_ms"] - refm["meanNN_ms"]
    d_sdnn = metrics["sdnn_ms"] - refm["sdnn_ms"]

    print(f"{rec:<10} {ev['Se']:.3f} {ev['PPV']:.3f} {ev['F1']:.3f} "
          f"{ev['TP']:>5} {ev['FP']:>5} {ev['FN']:>5} {d_mean:>12.2f} {d_sdnn:>10.2f}")

    rows.append((rec, ev["Se"], ev["PPV"], ev["F1"], ev["TP"], ev["FP"], ev["FN"], d_mean, d_sdnn))

# Summary
Se_vals = np.array([r[1] for r in rows], dtype=float)
PPV_vals = np.array([r[2] for r in rows], dtype=float)
F1_vals = np.array([r[3] for r in rows], dtype=float)

print("\n--- Summary (±{} ms) ---".format(tol_ms))
print(f"Mean Se:   {Se_vals.mean():.3f}   Median Se:   {np.median(Se_vals):.3f}")
print(f"Mean PPV:  {PPV_vals.mean():.3f}   Median PPV:  {np.median(PPV_vals):.3f}")
print(f"Mean F1:   {F1_vals.mean():.3f}   Median F1:   {np.median(F1_vals):.3f}")

# Save CSV
with open("qrs_eval_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["record", "Se", "PPV", "F1", "TP", "FP", "FN", "DeltaMeanNN_ms", "DeltaSDNN_ms"])
    w.writerows(rows)
print("Saved qrs_eval_summary.csv")


