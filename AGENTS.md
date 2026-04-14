# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This repository contains standalone Python scripts for **oceanographic internal wave analysis** on continental shelves, using ADCP, CTD, and thermistor chain data. There is no web server, database, or build system — just scientific analysis scripts.

### Scripts

| Script | Purpose |
|---|---|
| `adcp_ctd_analysis.py` | ADCP current velocity + CTD buoyancy frequency, bandpass filtering, Welch PSD, wavelet analysis, wave direction |
| `st1.py` / `st2.py` / `st3.py` | Thermistor chain stations 1–3: isotherm depth interpolation, temperature field visualization, FFT spectrum |
| `st4-stat.py` | Station 4: TEOS-10 density, Brunt-Väisälä profile, Garrett-Munk spectral model comparison |
| `st-zugi.py` | Cross-station comparison: loads all 4 stations, matches sensor depths, synchronized temperature plots |

### Dependencies

Install via `pip install -r requirements.txt`. Core packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `PyWavelets`, `gsw`, `openpyxl`.

### Key caveats

- **No data files in repo.** All scripts reference hardcoded Windows paths (`C:\Документы\ДИПЛОМ\Химченко_данные\...`) for Excel/CSV/TXT input data. Scripts will error on file-not-found unless data is provided and paths are updated.
- **Interactive input.** Scripts `st1.py`–`st3.py`, `st4-stat.py`, and `st-zugi.py` call `input()` to prompt for isotherms or date ranges. Use `echo "value" | python3 script.py` to pipe input non-interactively.
- **Matplotlib backend.** In headless environments, set `matplotlib.use('Agg')` before importing `pyplot`, or set env var `MPLBACKEND=Agg`.
- **No linter configured.** Use `python3 -m py_compile <script>.py` for syntax checks.
- **No automated tests.** Validation is done by running scripts with data and inspecting plots.
