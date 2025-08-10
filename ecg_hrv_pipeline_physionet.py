
#####################  mitdb/100 #############

# """
# ECG + HRV pipeline on real PhysioNet data (Python)
# --------------------------------------------------
# Dependencies:
#   pip install wfdb numpy scipy matplotlib
#
# Usage examples:
#   from ecg_hrv_pipeline_physionet import ecg_hrv_pipeline_physionet
#   ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='light', powerline_freq=60)
#   ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='dark',  powerline_freq=60)
#
# Notes:
# - The function will download records automatically via WFDB if needed.
# - If passing 'mitdb/100' fails on your setup, try: record='100', pn_dir='mitdb'.
# """
# from __future__ import annotations
#
# import math
# from dataclasses import dataclass, asdict
# from typing import Tuple, Optional, Dict
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# try:
#     import wfdb  # type: ignore
# except Exception as e:  # pragma: no cover
#     wfdb = None
#
# from scipy.signal import butter, filtfilt, iirnotch, welch, medfilt
#
#
# # --------------------------- Public API --------------------------- #
#
# def ecg_hrv_pipeline_physionet(
#     record: str = "mitdb/100",
#     dur_s: int = 180,
#     theme: str = "light",
#     powerline_freq: float = 60.0,
#     pn_dir: Optional[str] = None,
# ) -> Tuple[Dict[str, float], np.ndarray, float]:
#     """Run the ECG + HRV pipeline on a PhysioNet record and plot results.
#
#     Parameters
#     ----------
#     record : str
#         Record spec. Examples: 'mitdb/100' or just '100' with pn_dir='mitdb'.
#     dur_s : int
#         Duration in seconds to analyze starting from t=0.
#     theme : str
#         'light' or 'dark' for plot styling.
#     powerline_freq : float
#         Powerline frequency for notch filtering (50 or 60 Hz).
#     pn_dir : Optional[str]
#         PhysioNet subdirectory (e.g. 'mitdb'). If None, the function attempts
#         to infer from the record string.
#
#     Returns
#     -------
#     dict
#         Dictionary of HRV metrics.
#     """
#     _apply_theme(theme)
#
#     if wfdb is None:
#         raise RuntimeError(
#             "WFDB not found. Install with `pip install wfdb` and try again."
#         )
#
#     # -------------------- Load ECG -------------------- #
#     sig, fs, t = _load_physionet_signal(record, pn_dir)
#
#     idx = (t >= 0) & (t < dur_s)
#     if not np.any(idx):
#         raise ValueError("Requested duration exceeds record length.")
#
#     x = sig[idx, 0].astype(float)  # use first channel
#     trow = t[idx]
#
#     # -------------------- Denoising -------------------- #
#     ecg_hp = hp_filter(x, fs, 0.5)
#     ecg_notch = notch_filter(ecg_hp, fs, powerline_freq)
#     ecg_bp = bp_filter(ecg_notch, fs, 5.0, 25.0)
#
#     # -------------------- QRS Enhancement + R-peaks -------------------- #
#     diff_sig = np.concatenate([[0.0], np.diff(ecg_bp)])
#     sq_sig = diff_sig ** 2
#     win_ms = 120.0
#     win_n = max(1, int(round(fs * win_ms / 1000.0)))
#     int_sig = movmean(sq_sig, win_n)
#
#     locs_R_pre, thr = detect_r_peaks(int_sig, fs)
#
#     search_radius = int(round(0.05 * fs))
#     locs_R = refine_r_peaks(ecg_bp, locs_R_pre, search_radius, fs)
#
#     # -------------------- HRV Metrics -------------------- #
#     if len(locs_R) < 2:
#         raise RuntimeError("Too few R-peaks detected. Try a longer segment.")
#
#     RR = np.diff(locs_R) / fs
#
#     # Keep NN and its time base aligned (fix for length mismatch)
#     mask_good = ~isoutlier_median(RR)
#     NN = RR[mask_good]
#     tRR = (locs_R[:-1] / fs)[mask_good]
#     if NN.size < 3:
#         print("Warning: Too few NN intervals—try a longer segment.")
#
#     metrics = compute_hrv_metrics(NN, tRR)
#
#     cv_rr = np.std(RR) / np.mean(RR)
#     ac_rr = autocorr_stat(RR, lag=1)
#     metrics.cv_rr = float(cv_rr)
#     metrics.ac_lag1 = float(ac_rr)
#     metrics.af_screen_positive = bool(
#         (cv_rr > 0.15) and (metrics.pnn50 > 15.0) and (ac_rr < 0.35)
#     )
#
#     # -------------------- Plotting -------------------- #
#     _plot_main(record, fs, dur_s, trow, x, ecg_hp, ecg_notch, ecg_bp, locs_R, int_sig, thr, RR)
#     _plot_secondary(metrics, NN, tRR)
#
#     # -------------------- Report -------------------- #
#     # -------------------- Report -------------------- #
#     m = asdict(metrics)
#     print("--- HRV METRICS ---")
#     for k, v in m.items():
#         print(f"{k}: {v}")
#
#     print(
#         f"\nAF-like irregularity screen: "
#         f"{'POSITIVE (heuristic)' if metrics.af_screen_positive else 'negative'}"
#     )
#     print(
#         (
#             "Mean NN = {meanNN_ms:.0f} ms | SDNN = {sdnn_ms:.0f} ms | "
#             "RMSSD = {rmssd_ms:.0f} ms | pNN50 = {pnn50:.1f} %"
#         ).format(**m)
#     )
#     print(
#         (
#             "LFnu = {lfnu:.1f} % | HFnu = {hfnu:.1f} % | "
#             "CV(RR) = {cv_rr:.2f} | Lag-1 autocorr = {ac_lag1:.2f}\n"
#         ).format(**m)
#     )
#
#     plt.show()
#
#     return m, locs_R, fs
#
#
# # --------------------------- Theme helpers --------------------------- #
#
# def _apply_theme(which: str) -> None:
#     which = (which or "light").lower()
#     if which == "dark":
#         fg = (0.96, 0.96, 0.96)
#         bg = (0.10, 0.10, 0.10)
#         gridc = (0.45, 0.45, 0.45)
#         legc = (0.16, 0.16, 0.16)
#     else:
#         fg = (0.00, 0.00, 0.00)
#         bg = (1.00, 1.00, 1.00)
#         gridc = (0.80, 0.80, 0.80)
#         legc = (1.00, 1.00, 1.00)
#
#     plt.rcParams.update({
#         "figure.facecolor": bg,
#         "axes.facecolor": bg,
#         "axes.edgecolor": fg,
#         "axes.labelcolor": fg,
#         "text.color": fg,
#         "axes.grid": True,
#         "grid.color": gridc,
#         "grid.linestyle": "-",
#         "grid.alpha": 1.0,
#         "axes.spines.top": True,
#         "axes.spines.right": True,
#         "axes.titlesize": 12,
#         "axes.labelsize": 11,
#         "legend.framealpha": 1.0,
#         "legend.facecolor": legc,
#         "legend.edgecolor": fg,
#     })
#
#
# # --------------------------- Loading --------------------------- #
#
# def _load_physionet_signal(record: str, pn_dir: Optional[str]) -> Tuple[np.ndarray, float, np.ndarray]:
#     """Load a signal using WFDB, attempting to infer pn_dir if omitted."""
#     rec = record
#     dir_guess = pn_dir
#     if "/" in record and pn_dir is None:
#         dir_guess, rec = record.split("/", 1)
#     try:
#         x, fields = wfdb.rdsamp(rec, pn_dir=dir_guess)
#     except Exception:
#         # second chance: maybe record already contains directory
#         x, fields = wfdb.rdsamp(record)
#
#     fs = float(fields.get("fs"))
#     n = x.shape[0]
#     t = np.arange(n) / fs
#     return x, fs, t
#
#
# # --------------------------- Filters --------------------------- #
#
# def hp_filter(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
#     """High-pass filter (Butterworth) or robust baseline removal fallback."""
#     try:
#         b, a = butter(2, fc / (fs / 2.0), btype="high")
#         return filtfilt(b, a, x)
#     except Exception:  # pragma: no cover
#         # Fallback: median filter baseline estimate then smooth
#         w = int(max(1, round(fs * 0.6)))
#         if w % 2 == 0:
#             w += 1
#         x_med = medfilt(x, kernel_size=w)
#         return x - movmean(x_med, w)
#
#
# def notch_filter(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
#     """Notch filter around powerline frequency (iirnotch)."""
#     try:
#         w0 = f0 / (fs / 2.0)
#         Q = 30.0
#         b, a = iirnotch(w0, w0 / Q)
#         return filtfilt(b, a, x)
#     except Exception:  # pragma: no cover
#         # Crude fallback: subtract best-fit sinusoid at f0
#         t = np.arange(x.size) / fs
#         ref = np.sin(2 * np.pi * f0 * t)
#         alpha = np.dot(ref, x) / np.dot(ref, ref)
#         return x - alpha * ref
#
#
# def bp_filter(x: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
#     """Bandpass filter emphasizing QRS energy."""
#     try:
#         b, a = butter(3, [f1 / (fs / 2.0), f2 / (fs / 2.0)], btype="bandpass")
#         return filtfilt(b, a, x)
#     except Exception:  # pragma: no cover
#         return x - movmean(x, int(round(fs * 0.15)))
#
#
# def movmean(x: np.ndarray, w: int) -> np.ndarray:
#     w = max(1, int(w))
#     k = np.ones(w) / w
#     return np.convolve(x, k, mode="same")
#
#
# # --------------------------- R-peak detection --------------------------- #
#
# def detect_r_peaks(int_sig: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
#     int_sig = np.asarray(int_sig).astype(float)
#     noise_est = np.percentile(int_sig, 60)
#     sig_est = np.percentile(int_sig, 98)
#     thr = 0.3 * noise_est + 0.7 * sig_est
#
#     ref = int(round(0.2 * fs))  # 200 ms refractory
#     cand = np.flatnonzero(int_sig > thr)
#     if cand.size == 0:
#         return np.array([], dtype=int), float(thr)
#
#     locs = []
#     k = 0
#     while k < cand.size:
#         idx = cand[k]
#         w_end = min(int_sig.size - 1, idx + ref)
#         localmax = idx + int(np.argmax(int_sig[idx:w_end + 1]))
#         locs.append(localmax)
#         # advance to next candidate beyond refractory
#         next_idx = np.searchsorted(cand, idx + ref + 1, side="left")
#         k = int(next_idx)
#
#     locs = np.unique(np.array(locs, dtype=int))
#     return locs, float(thr)
#
#
# def refine_r_peaks(ecg: np.ndarray, locs: np.ndarray, rad: int, fs: float) -> np.ndarray:
#     locs_ref = np.array(locs, dtype=int)
#     N = ecg.size
#     for i in range(locs_ref.size):
#         a = max(0, locs_ref[i] - rad)
#         b = min(N - 1, locs_ref[i] + rad)
#         mx = a + int(np.argmax(ecg[a:b + 1]))
#         locs_ref[i] = mx
#
#     locs_ref = np.sort(locs_ref)
#     if locs_ref.size > 1:
#         minsep = int(round(0.2 * fs))
#         keep = np.ones_like(locs_ref, dtype=bool)
#         j = 1
#         while j < locs_ref.size:
#             if (locs_ref[j] - locs_ref[j - 1]) < minsep:
#                 # keep larger amplitude
#                 if ecg[locs_ref[j]] >= ecg[locs_ref[j - 1]]:
#                     keep[j - 1] = False
#                 else:
#                     keep[j] = False
#             j += 1
#         locs_ref = locs_ref[keep]
#     return locs_ref
#
#
# # --------------------------- HRV metrics --------------------------- #
#
# @dataclass
# class HRVMetrics:
#     meanNN_ms: float
#     sdnn_ms: float
#     rmssd_ms: float
#     pnn50: float
#     lfnu: float
#     hfnu: float
#     lf_power: float
#     hf_power: float
#     total_power: float
#     lf_center_hz: float
#     hf_center_hz: float
#     SD1_ms: float
#     SD2_ms: float
#     cv_rr: float = math.nan
#     ac_lag1: float = math.nan
#     af_screen_positive: bool = False
#
#
# def compute_hrv_metrics(NN: np.ndarray, tNN: np.ndarray) -> HRVMetrics:
#     NN = np.asarray(NN, dtype=float)
#     meanNN = float(np.mean(NN))
#     sdnn = float(np.std(NN, ddof=0))
#     rmssd = float(np.sqrt(np.mean(np.diff(NN) ** 2)))
#     pnn50 = float(100.0 * np.mean(np.abs(np.diff(NN)) > 0.050))
#
#     lfnu, hfnu, lf_center, hf_center, lf_power, hf_power, total_power = hrv_freq(NN, tNN)
#     SD1, SD2 = poincare_SD1_SD2(NN)
#
#     return HRVMetrics(
#         meanNN_ms=meanNN * 1000.0,
#         sdnn_ms=sdnn * 1000.0,
#         rmssd_ms=rmssd * 1000.0,
#         pnn50=pnn50,
#         lfnu=lfnu,
#         hfnu=hfnu,
#         lf_power=lf_power,
#         hf_power=hf_power,
#         total_power=total_power,
#         lf_center_hz=lf_center,
#         hf_center_hz=hf_center,
#         SD1_ms=SD1 * 1000.0,
#         SD2_ms=SD2 * 1000.0,
#     )
#
#
# def hrv_freq(NN: np.ndarray, tNN: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
#     if NN.size < 4:
#         return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan
#
#     fs_hrv = 4.0
#     tu = np.arange(tNN[0], tNN[-1] + 1e-9, 1.0 / fs_hrv)
#     x = np.interp(tu, tNN, NN)
#     f, pxx = welch(x - np.mean(x), fs=fs_hrv, nperseg=256, noverlap=128, nfft=1024)
#
#     lf_band = (0.04, 0.15)
#     hf_band = (0.15, 0.40)
#
#     lf_idx = (f >= lf_band[0]) & (f < lf_band[1])
#     hf_idx = (f >= hf_band[0]) & (f <= hf_band[1])
#
#     lf_power = float(np.trapz(pxx[lf_idx], f[lf_idx]))
#     hf_power = float(np.trapz(pxx[hf_idx], f[hf_idx]))
#     total_power = float(np.trapz(pxx[(f >= lf_band[0]) & (f <= hf_band[1])], f[(f >= lf_band[0]) & (f <= hf_band[1])]))
#
#     denom = (lf_power + hf_power) if (lf_power + hf_power) > 0 else math.nan
#     lfnu = 100.0 * lf_power / denom if denom == denom else math.nan
#     hfnu = 100.0 * hf_power / denom if denom == denom else math.nan
#
#     lf_center = centroid(f[lf_idx], pxx[lf_idx])
#     hf_center = centroid(f[hf_idx], pxx[hf_idx])
#
#     return lfnu, hfnu, float(lf_center), float(hf_center), lf_power, hf_power, total_power
#
#
# def hrv_psd(NN: np.ndarray, tNN: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     if NN.size < 4:
#         f = np.linspace(0.0, 0.5, 512)
#         pxx = np.zeros_like(f)
#         return pxx, f
#
#     fs_hrv = 4.0
#     tu = np.arange(tNN[0], tNN[-1] + 1e-9, 1.0 / fs_hrv)
#     x = np.interp(tu, tNN, NN)
#     f, pxx = welch(x - np.mean(x), fs=fs_hrv, nperseg=256, noverlap=128, nfft=1024)
#     return pxx, f
#
#
# def centroid(f: np.ndarray, p: np.ndarray) -> float:
#     if f.size == 0 or p.size == 0 or np.sum(p) <= 0:
#         return math.nan
#     return float(np.sum(f * p) / np.sum(p))
#
#
# def poincare_SD1_SD2(NN: np.ndarray) -> Tuple[float, float]:
#     NN1 = NN[:-1]
#     NN2 = NN[1:]
#     diffs = (NN2 - NN1) / np.sqrt(2.0)
#     sums = (NN2 + NN1) / np.sqrt(2.0)
#     SD1 = float(np.std(diffs, ddof=0))
#     SD2 = float(np.std(sums, ddof=0))
#     return SD1, SD2
#
#
# def autocorr_stat(x: np.ndarray, lag: int) -> float:
#     x = np.asarray(x, dtype=float)
#     x = x - np.mean(x)
#     if x.size < lag + 1:
#         return math.nan
#     num = np.sum(x[:-lag] * x[lag:])
#     den = np.sum(x ** 2)
#     return float(num / den) if den != 0 else math.nan
#
#
# def isoutlier_median(x: np.ndarray, thresh: float = 3.5) -> np.ndarray:
#     """Median-based outlier detection (approx MATLAB isoutlier(...,'median'))."""
#     x = np.asarray(x, dtype=float)
#     med = np.median(x)
#     mad = np.median(np.abs(x - med))
#     if mad == 0:
#         return np.zeros_like(x, dtype=bool)
#     z = 0.6745 * (x - med) / mad
#     return np.abs(z) > thresh
#
#
# # --------------------------- Plotting --------------------------- #
#
# def _plot_main(
#     record: str,
#     fs: float,
#     dur_s: int,
#     trow: np.ndarray,
#     x: np.ndarray,
#     ecg_hp: np.ndarray,
#     ecg_notch: np.ndarray,
#     ecg_bp: np.ndarray,
#     locs_R: np.ndarray,
#     int_sig: np.ndarray,
#     thr: float,
#     RR: np.ndarray,
# ) -> None:
#     fig = plt.figure(figsize=(12, 8))
#     fig.canvas.manager.set_window_title("ECG & HRV Pipeline (PhysioNet)")
#
#     ax1 = plt.subplot(4, 1, 1)
#     ax1.plot(trow, x, color=(0.35, 0.35, 0.35), label="Raw ECG")
#     ax1.plot(trow, ecg_hp, label="HP")
#     ax1.plot(trow, ecg_notch, label="Notch")
#     ax1.plot(trow, ecg_bp, label="Bandpass")
#     ax1.legend(loc="upper right")
#     ax1.set_xlabel("Time (s)")
#     ax1.set_ylabel("mV")
#     ax1.set_title(f"Record {record}  |  fs = {int(fs)} Hz  |  Duration = {dur_s} s")
#
#     ax2 = plt.subplot(4, 1, 2)
#     ax2.plot(trow, ecg_notch)
#     r_times = locs_R / fs
#     r_vals = np.interp(r_times, trow, ecg_notch)
#     ax2.stem(r_times, r_vals, basefmt=" ", linefmt="-", markerfmt="o")
#     ax2.set_xlabel("Time (s)")
#     ax2.set_ylabel("mV")
#     ax2.set_title("Filtered ECG + Detected R-peaks")
#
#     ax3 = plt.subplot(4, 1, 3)
#     ax3.plot(trow, int_sig)
#     ax3.axhline(thr, linestyle="--", color="r")
#     ax3.set_xlabel("Time (s)")
#     ax3.set_ylabel("AU")
#     ax3.set_title("QRS Enhancement (Integrated Signal) + Threshold")
#
#     ax4 = plt.subplot(4, 1, 4)
#     ax4.plot(locs_R[:-1] / fs, RR * 1000.0, ".-")
#     ax4.set_xlabel("Time (s)")
#     ax4.set_ylabel("RR (ms)")
#     ax4.set_title("Tachogram (RR Intervals)")
#
#     fig.tight_layout()
#
#
# def _plot_secondary(metrics: HRVMetrics, NN: np.ndarray, tNN: np.ndarray) -> None:
#     fig = plt.figure(figsize=(11, 4))
#     fig.canvas.manager.set_window_title("HRV Frequency & Nonlinear")
#
#     # PSD
#     ax1 = plt.subplot(1, 3, 1)
#     pxx, f = hrv_psd(NN, tNN)
#     ax1.plot(f, pxx)
#     ax1.set_xlim(0.0, 0.5)
#     ax1.set_xlabel("Hz")
#     ax1.set_ylabel("Power (ms^2/Hz)")
#     ax1.set_title("HRV PSD (Welch)")
#
#     # Poincaré
#     ax2 = plt.subplot(1, 3, 2)
#     NN1 = NN[:-1]
#     NN2 = NN[1:]
#     ax2.plot(NN1 * 1000.0, NN2 * 1000.0, ".")
#     ax2.set_aspect("equal", adjustable="box")
#     ax2.set_xlabel("NN_n (ms)")
#     ax2.set_ylabel("NN_{n+1} (ms)")
#     ax2.set_title(f"Poincaré (SD1={metrics.SD1_ms:.1f} ms, SD2={metrics.SD2_ms:.1f} ms)")
#
#     # LF/HF normalized
#     ax3 = plt.subplot(1, 3, 3)
#     ax3.bar(["LFnu", "HFnu"], [metrics.lfnu, metrics.hfnu])
#     ax3.set_ylim(0, 100)
#     ax3.set_title("Normalized LF/HF Power (%)")
#
#     fig.tight_layout()
#
# #############################  evaluating %%%%%%%%%%%%%%%
#
#
# def evaluate_against_annotations(record: str,
#                                  locs_R: np.ndarray,
#                                  fs: float,
#                                  pn_dir: Optional[str] = None,
#                                  tol_ms: float = 50.0):
#     """
#     Compare detected R-peaks to PhysioNet annotations.
#     Returns a dict with Se, PPV, F1, counts, and timing error stats.
#     """
#     if wfdb is None:
#         raise RuntimeError("WFDB required. pip install wfdb")
#
#     # Infer pn_dir if embedded in record like 'mitdb/100'
#     rec = record
#     ann_dir = pn_dir
#     if "/" in record and pn_dir is None:
#         ann_dir, rec = record.split("/", 1)
#
#     # Most MIT-BIH records use annotation "atr" and symbol 'N','V', etc.
#     # R-peaks correspond to beat annotations (any beat symbol).
#     ann = wfdb.rdann(rec, 'atr', pn_dir=ann_dir)
#     ref_samples = np.asarray(ann.sample, dtype=int)
#
#     # Some records have two leads; annotations are aligned to the record sampling.
#     # We'll assume all beat annotations are R-peaks.
#     # Filter to analysis window if you truncated the signal:
#     # (Use the time of your last detected peak)
#     if locs_R.size:
#         max_sample = int(np.max(locs_R) + fs)  # small cushion
#         ref_samples = ref_samples[ref_samples < max_sample]
#
#     # Match with tolerance
#     tol_samples = int(round(tol_ms * 1e-3 * fs))
#     det = np.asarray(locs_R, dtype=int)
#     ref = np.asarray(ref_samples, dtype=int)
#
#     used_det = np.zeros(det.size, dtype=bool)
#     used_ref = np.zeros(ref.size, dtype=bool)
#     matches = []
#
#     j = 0
#     for i, r in enumerate(ref):
#         # advance j to first det not too far behind
#         while j < det.size and det[j] < r - tol_samples:
#             j += 1
#         # check candidates around r
#         k0 = max(0, j - 3)
#         k1 = min(det.size, j + 4)
#         if k0 >= k1:
#             continue
#         k = np.argmin(np.abs(det[k0:k1] - r)) + k0
#         if abs(det[k] - r) <= tol_samples and not used_det[k]:
#             used_det[k] = True
#             used_ref[i] = True
#             matches.append(det[k] - r)
#
#     TP = int(np.sum(used_ref))
#     FN = int(ref.size - TP)
#     FP = int(np.sum(~used_det))  # detections that didn't match any ref
#
#     Se = TP / (TP + FN) if (TP + FN) else float('nan')         # sensitivity (recall)
#     PPV = TP / (TP + FP) if (TP + FP) else float('nan')        # precision
#     F1 = 2*Se*PPV/(Se+PPV) if (Se+PPV) else float('nan')
#
#     # Timing error (ms) for matched beats
#     matches = np.asarray(matches, dtype=float)
#     mae_ms = np.mean(np.abs(matches)) * 1000.0 / fs if matches.size else float('nan')
#     med_ms = np.median(matches) * 1000.0 / fs if matches.size else float('nan')
#     mad_ms = np.median(np.abs(matches - np.median(matches))) * 1000.0 / fs if matches.size else float('nan')
#
#     return {
#         "TP": TP, "FP": FP, "FN": FN,
#         "Se": Se, "PPV": PPV, "F1": F1,
#         "timing_mae_ms": mae_ms,
#         "timing_median_ms": med_ms,
#         "timing_mad_ms": mad_ms,
#         "tol_ms": tol_ms,
#         "n_ref_beats": int(ref.size),
#         "n_det_beats": int(det.size),
#     }
#
# def hrv_from_reference_annotations(record: str,
#                                    fs: float,
#                                    pn_dir: Optional[str] = None,
#                                    dur_s: Optional[float] = None):
#     """Compute HRV metrics using reference R-peaks for comparison."""
#     rec = record
#     ann_dir = pn_dir
#     if "/" in record and pn_dir is None:
#         ann_dir, rec = record.split("/", 1)
#     ann = wfdb.rdann(rec, 'atr', pn_dir=ann_dir)
#     ref = np.asarray(ann.sample, dtype=int)
#
#     if dur_s is not None:
#         ref = ref[ref < int(dur_s * fs)]
#
#     RR = np.diff(ref) / fs
#     # emulate the same outlier removal applied to detected beats
#     mask_good = ~isoutlier_median(RR)
#     NN = RR[mask_good]
#     tRR = (ref[:-1] / fs)[mask_good]
#     return compute_hrv_metrics(NN, tRR)




##############################################################
####################  batch "mitdb/100", "mitdb/101", "mitdb/103", "mitdb/105" ###########
###################################################################################


"""
ECG + HRV pipeline on real PhysioNet data (Python)
--------------------------------------------------
Dependencies:
  pip install wfdb numpy scipy matplotlib

Usage examples:
  from ecg_hrv_pipeline_physionet import ecg_hrv_pipeline_physionet
  ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='light', powerline_freq=60)
  ecg_hrv_pipeline_physionet('mitdb/100', dur_s=180, theme='dark',  powerline_freq=60)

Notes:
- The function will download records automatically via WFDB if needed.
- If passing 'mitdb/100' fails on your setup, try: record='100', pn_dir='mitdb'.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    import wfdb  # type: ignore
except Exception as e:  # pragma: no cover
    wfdb = None

from scipy.signal import butter, filtfilt, iirnotch, welch, medfilt


# --------------------------- Public API --------------------------- #

# --------------------------- Public API --------------------------- #

def ecg_hrv_pipeline_physionet(
    record: str = "mitdb/100",
    dur_s: int = 180,
    theme: str = "light",
    powerline_freq: float = 60.0,
    pn_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Run the ECG + HRV pipeline on a PhysioNet record and plot results.

    Parameters
    ----------
    record : str
        Record spec. Examples: 'mitdb/100' or just '100' with pn_dir='mitdb'.
    dur_s : int
        Duration in seconds to analyze starting from t=0.
    theme : str
        'light' or 'dark' for plot styling.
    powerline_freq : float
        Powerline frequency for notch filtering (50 or 60 Hz).
    pn_dir : Optional[str]
        PhysioNet subdirectory (e.g., 'mitdb'). If None, the function attempts
        to infer from the record string.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray, float]
        - metrics: dict of HRV metrics
        - locs_R: numpy array of detected R-peak sample indices
        - fs: sampling frequency (Hz)
    """

    _apply_theme(theme)

    if wfdb is None:
        raise RuntimeError(
            "WFDB not found. Install with `pip install wfdb` and try again."
        )

    # -------------------- Load ECG -------------------- #
    sig, fs, t = _load_physionet_signal(record, pn_dir)

    idx = (t >= 0) & (t < dur_s)
    if not np.any(idx):
        raise ValueError("Requested duration exceeds record length.")

    x = sig[idx, 0].astype(float)  # use first channel
    trow = t[idx]

    # -------------------- Denoising -------------------- #
    ecg_hp = hp_filter(x, fs, 0.5)
    ecg_notch = notch_filter(ecg_hp, fs, powerline_freq)
    ecg_bp = bp_filter(ecg_notch, fs, 5.0, 25.0)

    # -------------------- QRS Enhancement + R-peaks -------------------- #
    diff_sig = np.concatenate([[0.0], np.diff(ecg_bp)])
    sq_sig = diff_sig ** 2
    win_ms = 120.0
    win_n = max(1, int(round(fs * win_ms / 1000.0)))
    int_sig = movmean(sq_sig, win_n)

    locs_R_pre, thr = detect_r_peaks(int_sig, fs)

    search_radius = int(round(0.05 * fs))
    locs_R = refine_r_peaks(ecg_bp, locs_R_pre, search_radius, fs)

    # -------------------- HRV Metrics -------------------- #
    if len(locs_R) < 2:
        raise RuntimeError("Too few R-peaks detected. Try a longer segment.")

    RR = np.diff(locs_R) / fs

    # Keep NN and its time base aligned (fix for length mismatch)
    mask_good = ~isoutlier_median(RR)
    NN = RR[mask_good]
    tRR = (locs_R[:-1] / fs)[mask_good]
    if NN.size < 3:
        print("Warning: Too few NN intervals—try a longer segment.")

    metrics = compute_hrv_metrics(NN, tRR)

    cv_rr = np.std(RR) / np.mean(RR)
    ac_rr = autocorr_stat(RR, lag=1)
    metrics.cv_rr = float(cv_rr)
    metrics.ac_lag1 = float(ac_rr)
    metrics.af_screen_positive = bool(
        (cv_rr > 0.15) and (metrics.pnn50 > 15.0) and (ac_rr < 0.35)
    )

    # -------------------- Plotting -------------------- #
    _plot_main(record, fs, dur_s, trow, x, ecg_hp, ecg_notch, ecg_bp, locs_R, int_sig, thr, RR)
    _plot_secondary(metrics, NN, tRR)

    # -------------------- Report -------------------- #
    # -------------------- Report -------------------- #
    m = asdict(metrics)
    print("--- HRV METRICS ---")
    for k, v in m.items():
        print(f"{k}: {v}")

    print(
        f"\nAF-like irregularity screen: "
        f"{'POSITIVE (heuristic)' if metrics.af_screen_positive else 'negative'}"
    )
    print(
        (
            "Mean NN = {meanNN_ms:.0f} ms | SDNN = {sdnn_ms:.0f} ms | "
            "RMSSD = {rmssd_ms:.0f} ms | pNN50 = {pnn50:.1f} %"
        ).format(**m)
    )
    print(
        (
            "LFnu = {lfnu:.1f} % | HFnu = {hfnu:.1f} % | "
            "CV(RR) = {cv_rr:.2f} | Lag-1 autocorr = {ac_lag1:.2f}\n"
        ).format(**m)
    )

    plt.show()

    return m, locs_R, fs


# --------------------------- Theme helpers --------------------------- #

def _apply_theme(which: str) -> None:
    which = (which or "light").lower()
    if which == "dark":
        fg = (0.96, 0.96, 0.96)
        bg = (0.10, 0.10, 0.10)
        gridc = (0.45, 0.45, 0.45)
        legc = (0.16, 0.16, 0.16)
    else:
        fg = (0.00, 0.00, 0.00)
        bg = (1.00, 1.00, 1.00)
        gridc = (0.80, 0.80, 0.80)
        legc = (1.00, 1.00, 1.00)

    plt.rcParams.update({
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "text.color": fg,
        "axes.grid": True,
        "grid.color": gridc,
        "grid.linestyle": "-",
        "grid.alpha": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.framealpha": 1.0,
        "legend.facecolor": legc,
        "legend.edgecolor": fg,
    })


# --------------------------- Loading --------------------------- #

def _load_physionet_signal(record: str, pn_dir: Optional[str]) -> Tuple[np.ndarray, float, np.ndarray]:
    """Load a signal using WFDB, attempting to infer pn_dir if omitted."""
    rec = record
    dir_guess = pn_dir
    if "/" in record and pn_dir is None:
        dir_guess, rec = record.split("/", 1)
    try:
        x, fields = wfdb.rdsamp(rec, pn_dir=dir_guess)
    except Exception:
        # second chance: maybe record already contains directory
        x, fields = wfdb.rdsamp(record)

    fs = float(fields.get("fs"))
    n = x.shape[0]
    t = np.arange(n) / fs
    return x, fs, t


# --------------------------- Filters --------------------------- #

def hp_filter(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """High-pass filter (Butterworth) or robust baseline removal fallback."""
    try:
        b, a = butter(2, fc / (fs / 2.0), btype="high")
        return filtfilt(b, a, x)
    except Exception:  # pragma: no cover
        # Fallback: median filter baseline estimate then smooth
        w = int(max(1, round(fs * 0.6)))
        if w % 2 == 0:
            w += 1
        x_med = medfilt(x, kernel_size=w)
        return x - movmean(x_med, w)


def notch_filter(x: np.ndarray, fs: float, f0: float) -> np.ndarray:
    """Notch filter around powerline frequency (iirnotch)."""
    try:
        w0 = f0 / (fs / 2.0)
        Q = 30.0
        b, a = iirnotch(w0, w0 / Q)
        return filtfilt(b, a, x)
    except Exception:  # pragma: no cover
        # Crude fallback: subtract best-fit sinusoid at f0
        t = np.arange(x.size) / fs
        ref = np.sin(2 * np.pi * f0 * t)
        alpha = np.dot(ref, x) / np.dot(ref, ref)
        return x - alpha * ref


def bp_filter(x: np.ndarray, fs: float, f1: float, f2: float) -> np.ndarray:
    """Bandpass filter emphasizing QRS energy."""
    try:
        b, a = butter(3, [f1 / (fs / 2.0), f2 / (fs / 2.0)], btype="bandpass")
        return filtfilt(b, a, x)
    except Exception:  # pragma: no cover
        return x - movmean(x, int(round(fs * 0.15)))


def movmean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# --------------------------- R-peak detection --------------------------- #

def detect_r_peaks(int_sig: np.ndarray, fs: float) -> Tuple[np.ndarray, float]:
    int_sig = np.asarray(int_sig).astype(float)
    noise_est = np.percentile(int_sig, 60)
    sig_est = np.percentile(int_sig, 98)
    thr = 0.3 * noise_est + 0.7 * sig_est

    ref = int(round(0.2 * fs))  # 200 ms refractory
    cand = np.flatnonzero(int_sig > thr)
    if cand.size == 0:
        return np.array([], dtype=int), float(thr)

    locs = []
    k = 0
    while k < cand.size:
        idx = cand[k]
        w_end = min(int_sig.size - 1, idx + ref)
        localmax = idx + int(np.argmax(int_sig[idx:w_end + 1]))
        locs.append(localmax)
        # advance to next candidate beyond refractory
        next_idx = np.searchsorted(cand, idx + ref + 1, side="left")
        k = int(next_idx)

    locs = np.unique(np.array(locs, dtype=int))
    return locs, float(thr)


def refine_r_peaks(ecg: np.ndarray, locs: np.ndarray, rad: int, fs: float) -> np.ndarray:
    locs_ref = np.array(locs, dtype=int)
    N = ecg.size
    for i in range(locs_ref.size):
        a = max(0, locs_ref[i] - rad)
        b = min(N - 1, locs_ref[i] + rad)
        mx = a + int(np.argmax(ecg[a:b + 1]))
        locs_ref[i] = mx

    locs_ref = np.sort(locs_ref)
    if locs_ref.size > 1:
        minsep = int(round(0.2 * fs))
        keep = np.ones_like(locs_ref, dtype=bool)
        j = 1
        while j < locs_ref.size:
            if (locs_ref[j] - locs_ref[j - 1]) < minsep:
                # keep larger amplitude
                if ecg[locs_ref[j]] >= ecg[locs_ref[j - 1]]:
                    keep[j - 1] = False
                else:
                    keep[j] = False
            j += 1
        locs_ref = locs_ref[keep]
    return locs_ref


# --------------------------- HRV metrics --------------------------- #

@dataclass
class HRVMetrics:
    meanNN_ms: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50: float
    lfnu: float
    hfnu: float
    lf_power: float
    hf_power: float
    total_power: float
    lf_center_hz: float
    hf_center_hz: float
    SD1_ms: float
    SD2_ms: float
    cv_rr: float = math.nan
    ac_lag1: float = math.nan
    af_screen_positive: bool = False


def compute_hrv_metrics(NN: np.ndarray, tNN: np.ndarray) -> HRVMetrics:
    NN = np.asarray(NN, dtype=float)
    meanNN = float(np.mean(NN))
    sdnn = float(np.std(NN, ddof=0))
    rmssd = float(np.sqrt(np.mean(np.diff(NN) ** 2)))
    pnn50 = float(100.0 * np.mean(np.abs(np.diff(NN)) > 0.050))

    lfnu, hfnu, lf_center, hf_center, lf_power, hf_power, total_power = hrv_freq(NN, tNN)
    SD1, SD2 = poincare_SD1_SD2(NN)

    return HRVMetrics(
        meanNN_ms=meanNN * 1000.0,
        sdnn_ms=sdnn * 1000.0,
        rmssd_ms=rmssd * 1000.0,
        pnn50=pnn50,
        lfnu=lfnu,
        hfnu=hfnu,
        lf_power=lf_power,
        hf_power=hf_power,
        total_power=total_power,
        lf_center_hz=lf_center,
        hf_center_hz=hf_center,
        SD1_ms=SD1 * 1000.0,
        SD2_ms=SD2 * 1000.0,
    )


def hrv_freq(NN: np.ndarray, tNN: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    if NN.size < 4:
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

    fs_hrv = 4.0
    tu = np.arange(tNN[0], tNN[-1] + 1e-9, 1.0 / fs_hrv)
    x = np.interp(tu, tNN, NN)
    f, pxx = welch(x - np.mean(x), fs=fs_hrv, nperseg=256, noverlap=128, nfft=1024)

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.40)

    lf_idx = (f >= lf_band[0]) & (f < lf_band[1])
    hf_idx = (f >= hf_band[0]) & (f <= hf_band[1])

    lf_power = float(np.trapz(pxx[lf_idx], f[lf_idx]))
    hf_power = float(np.trapz(pxx[hf_idx], f[hf_idx]))
    total_power = float(np.trapz(pxx[(f >= lf_band[0]) & (f <= hf_band[1])], f[(f >= lf_band[0]) & (f <= hf_band[1])]))

    denom = (lf_power + hf_power) if (lf_power + hf_power) > 0 else math.nan
    lfnu = 100.0 * lf_power / denom if denom == denom else math.nan
    hfnu = 100.0 * hf_power / denom if denom == denom else math.nan

    lf_center = centroid(f[lf_idx], pxx[lf_idx])
    hf_center = centroid(f[hf_idx], pxx[hf_idx])

    return lfnu, hfnu, float(lf_center), float(hf_center), lf_power, hf_power, total_power


def hrv_psd(NN: np.ndarray, tNN: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if NN.size < 4:
        f = np.linspace(0.0, 0.5, 512)
        pxx = np.zeros_like(f)
        return pxx, f

    fs_hrv = 4.0
    tu = np.arange(tNN[0], tNN[-1] + 1e-9, 1.0 / fs_hrv)
    x = np.interp(tu, tNN, NN)
    f, pxx = welch(x - np.mean(x), fs=fs_hrv, nperseg=256, noverlap=128, nfft=1024)
    return pxx, f


def centroid(f: np.ndarray, p: np.ndarray) -> float:
    if f.size == 0 or p.size == 0 or np.sum(p) <= 0:
        return math.nan
    return float(np.sum(f * p) / np.sum(p))


def poincare_SD1_SD2(NN: np.ndarray) -> Tuple[float, float]:
    NN1 = NN[:-1]
    NN2 = NN[1:]
    diffs = (NN2 - NN1) / np.sqrt(2.0)
    sums = (NN2 + NN1) / np.sqrt(2.0)
    SD1 = float(np.std(diffs, ddof=0))
    SD2 = float(np.std(sums, ddof=0))
    return SD1, SD2


def autocorr_stat(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    if x.size < lag + 1:
        return math.nan
    num = np.sum(x[:-lag] * x[lag:])
    den = np.sum(x ** 2)
    return float(num / den) if den != 0 else math.nan


def isoutlier_median(x: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """Median-based outlier detection (approx MATLAB isoutlier(...,'median'))."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x, dtype=bool)
    z = 0.6745 * (x - med) / mad
    return np.abs(z) > thresh


# --------------------------- Plotting --------------------------- #

def _plot_main(
    record: str,
    fs: float,
    dur_s: int,
    trow: np.ndarray,
    x: np.ndarray,
    ecg_hp: np.ndarray,
    ecg_notch: np.ndarray,
    ecg_bp: np.ndarray,
    locs_R: np.ndarray,
    int_sig: np.ndarray,
    thr: float,
    RR: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("ECG & HRV Pipeline (PhysioNet)")

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(trow, x, color=(0.35, 0.35, 0.35), label="Raw ECG")
    ax1.plot(trow, ecg_hp, label="HP")
    ax1.plot(trow, ecg_notch, label="Notch")
    ax1.plot(trow, ecg_bp, label="Bandpass")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("mV")
    ax1.set_title(f"Record {record}  |  fs = {int(fs)} Hz  |  Duration = {dur_s} s")

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(trow, ecg_notch)
    r_times = locs_R / fs
    r_vals = np.interp(r_times, trow, ecg_notch)
    ax2.stem(r_times, r_vals, basefmt=" ", linefmt="-", markerfmt="o")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("mV")
    ax2.set_title("Filtered ECG + Detected R-peaks")

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(trow, int_sig)
    ax3.axhline(thr, linestyle="--", color="r")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("AU")
    ax3.set_title("QRS Enhancement (Integrated Signal) + Threshold")

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(locs_R[:-1] / fs, RR * 1000.0, ".-")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("RR (ms)")
    ax4.set_title("Tachogram (RR Intervals)")

    fig.tight_layout()


def _plot_secondary(metrics: HRVMetrics, NN: np.ndarray, tNN: np.ndarray) -> None:
    fig = plt.figure(figsize=(11, 4))
    fig.canvas.manager.set_window_title("HRV Frequency & Nonlinear")

    # PSD
    ax1 = plt.subplot(1, 3, 1)
    pxx, f = hrv_psd(NN, tNN)
    ax1.plot(f, pxx)
    ax1.set_xlim(0.0, 0.5)
    ax1.set_xlabel("Hz")
    ax1.set_ylabel("Power (ms^2/Hz)")
    ax1.set_title("HRV PSD (Welch)")

    # Poincaré
    ax2 = plt.subplot(1, 3, 2)
    NN1 = NN[:-1]
    NN2 = NN[1:]
    ax2.plot(NN1 * 1000.0, NN2 * 1000.0, ".")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("NN_n (ms)")
    ax2.set_ylabel("NN_{n+1} (ms)")
    ax2.set_title(f"Poincaré (SD1={metrics.SD1_ms:.1f} ms, SD2={metrics.SD2_ms:.1f} ms)")

    # LF/HF normalized
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(["LFnu", "HFnu"], [metrics.lfnu, metrics.hfnu])
    ax3.set_ylim(0, 100)
    ax3.set_title("Normalized LF/HF Power (%)")

    fig.tight_layout()

#############################  evaluating %%%%%%%%%%%%%%%


def evaluate_against_annotations(record: str,
                                 locs_R: np.ndarray,
                                 fs: float,
                                 pn_dir: Optional[str] = None,
                                 tol_ms: float = 50.0):
    """
    Compare detected R-peaks to PhysioNet annotations.
    Returns a dict with Se, PPV, F1, counts, and timing error stats.
    """
    if wfdb is None:
        raise RuntimeError("WFDB required. pip install wfdb")

    # Infer pn_dir if embedded in record like 'mitdb/100'
    rec = record
    ann_dir = pn_dir
    if "/" in record and pn_dir is None:
        ann_dir, rec = record.split("/", 1)

    # Most MIT-BIH records use annotation "atr" and symbol 'N','V', etc.
    # R-peaks correspond to beat annotations (any beat symbol).
    ann = wfdb.rdann(rec, 'atr', pn_dir=ann_dir)
    ref_samples = np.asarray(ann.sample, dtype=int)

    # Some records have two leads; annotations are aligned to the record sampling.
    # We'll assume all beat annotations are R-peaks.
    # Filter to analysis window if you truncated the signal:
    # (Use the time of your last detected peak)
    if locs_R.size:
        max_sample = int(np.max(locs_R) + fs)  # small cushion
        ref_samples = ref_samples[ref_samples < max_sample]

    # Match with tolerance
    tol_samples = int(round(tol_ms * 1e-3 * fs))
    det = np.asarray(locs_R, dtype=int)
    ref = np.asarray(ref_samples, dtype=int)

    used_det = np.zeros(det.size, dtype=bool)
    used_ref = np.zeros(ref.size, dtype=bool)
    matches = []

    j = 0
    for i, r in enumerate(ref):
        # advance j to first det not too far behind
        while j < det.size and det[j] < r - tol_samples:
            j += 1
        # check candidates around r
        k0 = max(0, j - 3)
        k1 = min(det.size, j + 4)
        if k0 >= k1:
            continue
        k = np.argmin(np.abs(det[k0:k1] - r)) + k0
        if abs(det[k] - r) <= tol_samples and not used_det[k]:
            used_det[k] = True
            used_ref[i] = True
            matches.append(det[k] - r)

    TP = int(np.sum(used_ref))
    FN = int(ref.size - TP)
    FP = int(np.sum(~used_det))  # detections that didn't match any ref

    Se = TP / (TP + FN) if (TP + FN) else float('nan')         # sensitivity (recall)
    PPV = TP / (TP + FP) if (TP + FP) else float('nan')        # precision
    F1 = 2*Se*PPV/(Se+PPV) if (Se+PPV) else float('nan')

    # Timing error (ms) for matched beats
    matches = np.asarray(matches, dtype=float)
    mae_ms = np.mean(np.abs(matches)) * 1000.0 / fs if matches.size else float('nan')
    med_ms = np.median(matches) * 1000.0 / fs if matches.size else float('nan')
    mad_ms = np.median(np.abs(matches - np.median(matches))) * 1000.0 / fs if matches.size else float('nan')

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Se": Se, "PPV": PPV, "F1": F1,
        "timing_mae_ms": mae_ms,
        "timing_median_ms": med_ms,
        "timing_mad_ms": mad_ms,
        "tol_ms": tol_ms,
        "n_ref_beats": int(ref.size),
        "n_det_beats": int(det.size),
    }

def hrv_from_reference_annotations(record: str,
                                   fs: float,
                                   pn_dir: Optional[str] = None,
                                   dur_s: Optional[float] = None):
    """Compute HRV metrics using reference R-peaks for comparison."""
    rec = record
    ann_dir = pn_dir
    if "/" in record and pn_dir is None:
        ann_dir, rec = record.split("/", 1)
    ann = wfdb.rdann(rec, 'atr', pn_dir=ann_dir)
    ref = np.asarray(ann.sample, dtype=int)

    if dur_s is not None:
        ref = ref[ref < int(dur_s * fs)]

    RR = np.diff(ref) / fs
    # emulate the same outlier removal applied to detected beats
    mask_good = ~isoutlier_median(RR)
    NN = RR[mask_good]
    tRR = (ref[:-1] / fs)[mask_good]
    return compute_hrv_metrics(NN, tRR)


