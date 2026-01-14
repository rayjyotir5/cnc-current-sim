# CNC mains current shift simulator

Generates a synthetic **single-phase mains input current** time series for a CNC machine over one shift, sampled at **100 Hz**.

The output is **instantaneous current samples** (signed), created by sampling a synthesized mains current waveform (fundamental + rectifier-like harmonics) whose envelope is driven by a simple shift state model (idle, rapid, cut, tool change, breaks, etc.).

## Usage

Generate an 8-hour CSV (100 Hz):

```bash
python3 simulate_cnc_shift_current.py --hours 8 --fs 100 --out shift_current.csv
```

Include additional columns (state, approximate power/voltage/PF/THD):

```bash
python3 simulate_cnc_shift_current.py --hours 8 --fs 100 --include-meta --out shift_current.csv
```

Write a compact `.npz` instead of CSV:

```bash
python3 simulate_cnc_shift_current.py --hours 8 --fs 100 --out-format npz --out shift_current.npz
```

## Interactive viewer (Plotly)

The viewer writes an interactive HTML plot you can zoom/pan in your browser.

Install Plotly in a virtual environment (recommended on Homebrew Python):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install plotly
```

View the whole shift as a 1‑second RMS envelope:

```bash
python view_shift_current_plotly.py --input shift_current.csv --mode envelope --fs 100 --open
```

View a raw window (instantaneous samples):

```bash
python view_shift_current_plotly.py --input shift_current.csv --mode raw --fs 100 --start 3600 --duration 30 --open
```

## Key parameters

- `--phases 3` (default): assumes a balanced 3‑phase CNC, but outputs the current of the **one phase you’re measuring**.
- `--v-rms`: per‑phase RMS voltage at the measurement point (e.g. `230` for 400 V L‑L systems, `277` for 480 V L‑L systems).
- `--line-freq`: mains frequency (50/60 Hz). With 100 Hz sampling, the fundamental will **alias** (e.g. 60 Hz → 40 Hz).
- `--include-power-on`: adds inrush/boot/warmup at the beginning of the shift.
- `--unplanned-off-events`: inserts power-off downtime periods (`state=off`) followed by a restart sequence.
- `--breaks-off`: makes scheduled breaks use `state=off` (near-zero current) instead of `state=break`.
- `--seed`: makes the shift repeatable.

## Output

CSV default columns:

- `timestamp_ms`: unix epoch in milliseconds (UTC)
- `current_a`: instantaneous current sample in amps (signed)

With `--include-meta`, additional columns are included: `state`, `p_total_w`, `v_rms`, `pf_disp`, `thd_i_over_i1`.
