

# ECG-processing-stack (Python)

Full ECG processing stack for **real ECG data** from [PhysioNet](https://physionet.org/):  
loading, denoising, QRS detection, HRV metrics (time/frequency/nonlinear), an AF-like irregularity screen, and detailed evaluation vs. expert annotations.

All implemented in:

- `ecg_hrv_pipeline_physionet.py` — main pipeline function and helpers  
- `main.py` — batch evaluation script with summary table

---

##  What’s inside

- **Data loading**: Direct download and parsing of PhysioNet records via `wfdb`
- **Denoising**: High-pass, notch (50/60 Hz), and bandpass filters
- **QRS detection**:
  - Pan–Tompkins–style enhancement (derivative → squaring → integration)
  - Adaptive threshold & refractory
  - Local-max refinement on filtered ECG
- **HRV analysis**:
  - Time domain: `meanNN`, `SDNN`, `RMSSD`, `pNN50`
  - Frequency domain (Welch PSD, LF/HF bands): `LFnu`, `HFnu`, band powers & centers
  - Nonlinear: Poincaré `SD1`, `SD2`
- **AF-like irregularity screen**:
  - Flags if variability, pNN50, and lag-1 autocorr meet heuristic criteria
- **Evaluation vs ground truth**:
  - Compares detected R-peaks to PhysioNet annotation files (Se, PPV, F1, timing error)
  - Compares HRV metrics to reference computed from annotated beats
- **Visualization**:
  - Filter stages and R-peak overlay
  - Integrated QRS enhancement trace + threshold
  - RR tachogram
  - HRV PSD
  - Poincaré plot

---

##  Quick start

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install wfdb numpy scipy matplotlib
   ```
3. Run:
   ```bash
   python main.py
   ```
   This will:
   - Process multiple PhysioNet records
   - Print HRV metrics per record
   - Print detection accuracy table
   - Save results to `qrs_eval_summary.csv`

---

##  Requirements

- Python 3.8+  
- Packages: `wfdb`, `numpy`, `scipy`, `matplotlib`

---

##  Usage

Example: run pipeline on one record with plots:
```python
from ecg_hrv_pipeline_physionet import ecg_hrv_pipeline_physionet

metrics, locs_R, fs = ecg_hrv_pipeline_physionet(
    record="mitdb/100",
    dur_s=180,
    theme="light",
    powerline_freq=60
)
```

Batch evaluation (see `main.py`):
```python
records = ["mitdb/100", "mitdb/101", "mitdb/103", "mitdb/105"]
```
Outputs a formatted table of:
```
Record    Se    PPV    F1    TP   FP   FN   ΔmeanNN(ms)  ΔSDNN(ms)
```
and a summary of mean/median Se, PPV, F1.

---

##  Outputs

**Console:**
- HRV metrics dictionary
- AF-like screen result
- Detection accuracy metrics (Se, PPV, F1, TP, FP, FN, timing errors)
- HRV differences vs reference

**CSV:**
- `qrs_eval_summary.csv` with per-record evaluation

**Figures:**
- ECG before/after filtering
- R-peaks
- QRS enhancement trace
- Tachogram
- HRV PSD
- Poincaré plot

---

##  Methods (high level)

1. **Load ECG** from PhysioNet record via `wfdb.rdsamp`
2. **Filter**:
   - High-pass (0.5 Hz) for baseline removal
   - Notch (50/60 Hz) for powerline
   - Bandpass (~5–25 Hz) for QRS emphasis
3. **QRS detection**:
   - Differentiate → square → moving integration (~120 ms)
   - Adaptive threshold & refractory
   - Refine peaks by local maximum
4. **RR/NN intervals**:
   - Outlier removal by median filter
5. **HRV metrics**:
   - Time-domain
   - Frequency-domain (Welch PSD, LF/HF integration)
   - Nonlinear (Poincaré SD1/SD2)
6. **AF-like screen**:
   - Heuristic thresholds on CV(RR), pNN50, autocorrelation
7. **Evaluation**:
   - Compare detected R-peaks to annotated beats (tolerance ±50 ms)
   - Compute Se, PPV, F1, timing errors
   - Compare HRV to reference from annotations

---

##  Repository layout

```text
.
├── ecg_hrv_pipeline_physionet.py   # Pipeline & helper functions
├── main.py                         # Batch evaluation & summary
└── README.md                       # This file
```

##  Citation

If you use this in research or projects, please cite:  
**“ECG + HRV Advanced Pipeline (Python, PhysioNet) — loading, denoising, QRS detection, HRV metrics, and evaluation vs annotations.”**

##  Related Work
This project is the Python version of my MATLAB ECG pipeline:  
[ECG-processing-stack (MATLAB)](https://github.com/rzrm99/ECG-processing-stack)


---

##  Medical Disclaimer

This software is for **educational and research purposes only**.  
It is **not** intended for clinical diagnosis, treatment, or decision-making.  
The AF-like irregularity screen is a simple heuristic and must not be relied upon for any medical conclusions.  
Use at your own risk.
