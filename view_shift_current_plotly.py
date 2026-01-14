#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import webbrowser

import numpy as np


def _require_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except Exception:
        print(
            "Plotly is required for this viewer.\n\n"
            "Quick install (recommended: use a venv):\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install plotly\n\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _iso_parse(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_datetime_utc(ts_ms: np.ndarray) -> list[datetime]:
    return [datetime.fromtimestamp(float(x) / 1000.0, tz=timezone.utc) for x in ts_ms.tolist()]


@dataclass(frozen=True)
class RawWindow:
    ts_ms: np.ndarray
    current_a: np.ndarray


@dataclass(frozen=True)
class Envelope:
    ts_ms: np.ndarray
    irms_a: np.ndarray


def _read_csv_header(path: Path) -> tuple[list[str], dict[str, int]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [h.strip() for h in header]
    return header, {name: idx for idx, name in enumerate(header)}


def _read_csv_raw_window(
    path: Path,
    *,
    fs_hz: float,
    start_s: float,
    duration_s: float,
) -> RawWindow:
    header, col = _read_csv_header(path)
    if "current_a" not in col:
        raise SystemExit(f"CSV is missing required column 'current_a'. Found: {header}")
    ts_col = "timestamp_ms" if "timestamp_ms" in col else ("timestamp" if "timestamp" in col else "")
    if not ts_col:
        raise SystemExit(f"CSV is missing a timestamp column ('timestamp_ms' or 'timestamp'). Found: {header}")

    skip = int(round(max(0.0, start_s) * fs_hz))
    take = int(round(max(0.0, duration_s) * fs_hz))
    if take <= 0:
        raise SystemExit("--duration must be > 0")

    ts_out: list[int] = []
    i_out: list[float] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for _ in range(skip):
            if next(reader, None) is None:
                break
        for _ in range(take):
            row = next(reader, None)
            if row is None:
                break
            try:
                i_out.append(float(row[col["current_a"]]))
                if ts_col == "timestamp_ms":
                    ts_out.append(int(row[col[ts_col]]))
                else:
                    ts_out.append(int(_iso_parse(row[col[ts_col]]).timestamp() * 1000))
            except Exception:
                continue

    if not ts_out:
        raise SystemExit("No samples loaded (check --start/--duration and CSV format).")

    return RawWindow(ts_ms=np.asarray(ts_out, dtype=np.int64), current_a=np.asarray(i_out, dtype=np.float64))


def _read_csv_envelope_1s(
    path: Path,
    *,
    fs_hz: float,
) -> Envelope:
    header, col = _read_csv_header(path)
    if "current_a" not in col:
        raise SystemExit(f"CSV is missing required column 'current_a'. Found: {header}")
    ts_col = "timestamp_ms" if "timestamp_ms" in col else ("timestamp" if "timestamp" in col else "")
    if not ts_col:
        raise SystemExit(f"CSV is missing a timestamp column ('timestamp_ms' or 'timestamp'). Found: {header}")

    block = int(round(fs_hz))
    if block <= 0:
        raise SystemExit("--fs must be > 0")
    if not math.isclose(block, fs_hz, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("Envelope mode currently expects an integer sampling rate (e.g. --fs 100).")

    ts_out: list[int] = []
    irms_out: list[float] = []

    sum_sq = 0.0
    n = 0
    ts_block_first: int | None = None

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            try:
                i = float(row[col["current_a"]])
                if ts_block_first is None:
                    if ts_col == "timestamp_ms":
                        ts_block_first = int(row[col[ts_col]])
                    else:
                        ts_block_first = int(_iso_parse(row[col[ts_col]]).timestamp() * 1000)
            except Exception:
                continue

            sum_sq += i * i
            n += 1
            if n >= block:
                irms_out.append(math.sqrt(sum_sq / n))
                ts_out.append(int(ts_block_first) if ts_block_first is not None else 0)
                sum_sq = 0.0
                n = 0
                ts_block_first = None

    if n > 0:
        irms_out.append(math.sqrt(sum_sq / n))
        if ts_block_first is None:
            ts_block_first = ts_out[-1] + 1000 if ts_out else 0
        ts_out.append(int(ts_block_first))

    if not ts_out:
        raise SystemExit("No samples loaded from CSV.")

    return Envelope(ts_ms=np.asarray(ts_out, dtype=np.int64), irms_a=np.asarray(irms_out, dtype=np.float64))


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.load(path)
    ts_ms = d["timestamp_ms"].astype(np.int64, copy=False)
    i_a = d["current_a"].astype(np.float64, copy=False)
    fs_hz = float(d["fs_hz"]) if "fs_hz" in d else 100.0
    return ts_ms, i_a, fs_hz


def _envelope_from_arrays(ts_ms: np.ndarray, i_a: np.ndarray, fs_hz: float) -> Envelope:
    block = int(round(fs_hz))
    if block <= 0:
        raise SystemExit("--fs must be > 0")
    if not math.isclose(block, fs_hz, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit("Envelope mode currently expects an integer sampling rate (e.g. --fs 100).")
    n_blocks = int(i_a.size // block)
    if n_blocks <= 0:
        raise SystemExit("Not enough samples for envelope.")
    i2 = i_a[: n_blocks * block].reshape(n_blocks, block)
    irms = np.sqrt(np.mean(i2 * i2, axis=1))
    return Envelope(ts_ms=ts_ms[::block][:n_blocks], irms_a=irms)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive Plotly viewer for shift_current CSV/NPZ.")
    parser.add_argument("--input", type=Path, default=Path("shift_current.csv"))
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling rate (Hz). Used for CSV parsing and windowing.")
    parser.add_argument("--mode", choices=["envelope", "raw"], default="envelope")
    parser.add_argument("--start", type=float, default=0.0, help="Raw mode: window start (seconds from file start).")
    parser.add_argument("--duration", type=float, default=30.0, help="Raw mode: window duration (seconds).")
    parser.add_argument(
        "--x-axis",
        choices=["relative_s", "timestamp"],
        default="relative_s",
        help="X axis for plots: relative seconds or UTC timestamp.",
    )
    parser.add_argument("--out-html", type=Path, default=Path("shift_current_view.html"))
    parser.add_argument("--open", action="store_true", help="Open the HTML in your default browser.")

    args = parser.parse_args()

    _require_plotly()
    import plotly.graph_objects as go

    in_path = args.input
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    fs_hz = float(args.fs)
    if fs_hz <= 0:
        raise SystemExit("--fs must be > 0")

    if in_path.suffix.lower() == ".npz":
        ts_ms_all, i_a_all, fs_from_file = _read_npz(in_path)
        if args.mode == "envelope":
            env = _envelope_from_arrays(ts_ms_all, i_a_all, fs_from_file)
            ts_ms = env.ts_ms
            y = env.irms_a
            title = "Shift current envelope (1s RMS)"
            y_label = "Irms (A)"
        else:
            start_idx = int(round(max(0.0, float(args.start)) * fs_from_file))
            take = int(round(max(0.0, float(args.duration)) * fs_from_file))
            if take <= 0:
                raise SystemExit("--duration must be > 0")
            end_idx = min(i_a_all.size, start_idx + take)
            if end_idx <= start_idx:
                raise SystemExit("Raw window is empty (check --start/--duration).")
            ts_ms = ts_ms_all[start_idx:end_idx]
            y = i_a_all[start_idx:end_idx]
            title = "Shift current (instantaneous samples)"
            y_label = "Current (A)"
    else:
        if args.mode == "envelope":
            env = _read_csv_envelope_1s(in_path, fs_hz=fs_hz)
            ts_ms = env.ts_ms
            y = env.irms_a
            title = "Shift current envelope (1s RMS)"
            y_label = "Irms (A)"
        else:
            win = _read_csv_raw_window(in_path, fs_hz=fs_hz, start_s=float(args.start), duration_s=float(args.duration))
            ts_ms = win.ts_ms
            y = win.current_a
            title = "Shift current (instantaneous samples)"
            y_label = "Current (A)"

    if args.x_axis == "timestamp":
        x = _to_datetime_utc(ts_ms)
        x_label = "Time (UTC)"
    else:
        x0 = int(ts_ms[0])
        x = (ts_ms.astype(np.float64) - x0) / 1000.0
        x_label = "Time (s from window start)" if args.mode == "raw" else "Time (s from shift start)"

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=args.mode))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
    )

    out_html = args.out_html
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Wrote {out_html}")
    if args.open:
        webbrowser.open(out_html.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
