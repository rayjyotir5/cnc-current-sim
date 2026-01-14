#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class Segment:
    state: str
    duration_s: float


def _parse_start_time(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _iso_from_unix_ms(unix_ms: int) -> str:
    dt = datetime.fromtimestamp(unix_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _generate_segments(
    rng: np.random.Generator,
    shift_seconds: float,
    include_power_on: bool,
    breaks_off: bool,
    unplanned_off_events: int,
) -> list[Segment]:
    segments: list[Segment] = []

    def append(state: str, duration_s: float) -> None:
        if duration_s <= 0:
            return
        if segments and segments[-1].state == state:
            last = segments[-1]
            segments[-1] = Segment(state=last.state, duration_s=last.duration_s + float(duration_s))
            return
        segments.append(Segment(state=state, duration_s=float(duration_s)))

    planned_breaks: list[tuple[float, float]] = [
        (2 * 3600.0, 15 * 60.0),
        (4 * 3600.0, 30 * 60.0),
        (6 * 3600.0, 15 * 60.0),
    ]
    planned_breaks = [(s, d) for (s, d) in planned_breaks if 0.0 < s < shift_seconds]

    break_windows = [(s, s + d) for (s, d) in planned_breaks]

    def in_break_window(t_s: float) -> bool:
        return any(start <= t_s < end for (start, end) in break_windows)

    def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
        return (a0 < b1) and (b0 < a1)

    reserved: list[tuple[float, float]] = []

    def is_free(start_s: float, end_s: float) -> bool:
        if start_s < 0.0 or end_s > shift_seconds or end_s <= start_s:
            return False
        return not any(overlaps(start_s, end_s, r0, r1) for (r0, r1) in reserved)

    def reserve(start_s: float, end_s: float) -> None:
        reserved.append((start_s, end_s))

    events: list[tuple[float, str, float]] = []

    def add_event(start_s: float, state: str, dur_s: float) -> None:
        if dur_s <= 0.0 or start_s >= shift_seconds:
            return
        start_s = max(0.0, float(start_s))
        dur_s = float(min(dur_s, shift_seconds - start_s))
        events.append((start_s, state, dur_s))
        reserve(start_s, start_s + dur_s)

    def sample_restart_durations() -> tuple[float, float, float]:
        inrush_d = float(rng.uniform(2.0, 6.0))
        boot_d = float(rng.uniform(30.0, 120.0))
        idle_d = float(rng.uniform(15.0, 90.0))
        return inrush_d, boot_d, idle_d

    def add_restart_sequence(start_s: float, inrush_d: float, boot_d: float, idle_d: float) -> float:
        add_event(start_s, "inrush", inrush_d)
        add_event(start_s + inrush_d, "boot", boot_d)
        add_event(start_s + inrush_d + boot_d, "idle", idle_d)
        return inrush_d + boot_d + idle_d

    for start_s, dur_s in planned_breaks:
        break_state = "off" if breaks_off else "break"
        add_event(float(start_s), break_state, float(dur_s))
        if breaks_off:
            inrush_d, boot_d, idle_d = sample_restart_durations()
            add_restart_sequence(float(start_s + dur_s), inrush_d, boot_d, idle_d)

    if shift_seconds >= 10 * 60.0:
        alarm_duration_s = float(rng.uniform(60.0, 240.0))
        for _ in range(50):
            alarm_start_s = float(rng.uniform(0.8 * shift_seconds, min(0.98 * shift_seconds, shift_seconds - 1.0)))
            if is_free(alarm_start_s, alarm_start_s + alarm_duration_s) and not in_break_window(alarm_start_s):
                add_event(alarm_start_s, "alarm", alarm_duration_s)
                break

    unplanned_off_events = max(0, int(unplanned_off_events))
    if unplanned_off_events > 0 and shift_seconds >= 30 * 60.0:
        start_min_s = 10 * 60.0
        end_margin_s = 10 * 60.0
        for _ in range(unplanned_off_events):
            for _try in range(250):
                off_d = float(rng.uniform(3 * 60.0, 35 * 60.0))
                inrush_d, boot_d, idle_d = sample_restart_durations()
                restart_total = inrush_d + boot_d + idle_d
                latest_start = shift_seconds - (off_d + restart_total + end_margin_s)
                if latest_start <= start_min_s:
                    break
                off_start = float(rng.uniform(start_min_s, latest_start))
                if not is_free(off_start, off_start + off_d + restart_total):
                    continue
                add_event(off_start, "off", off_d)
                add_restart_sequence(off_start + off_d, inrush_d, boot_d, idle_d)
                break

    events.sort(key=lambda e: e[0])
    event_idx = 0

    t = 0.0

    def add_with_events(state: str, duration_s: float) -> None:
        nonlocal t, event_idx
        remaining = float(duration_s)
        while remaining > 0.0 and t < shift_seconds:
            while event_idx < len(events) and t >= events[event_idx][0] - 1e-9:
                ev_start_s, ev_state, ev_dur_s = events[event_idx]
                ev_remaining = min(ev_dur_s, shift_seconds - t)
                append(ev_state, ev_remaining)
                t += ev_remaining
                event_idx += 1
                if remaining <= 0.0 or t >= shift_seconds:
                    return

            next_event_start_s = events[event_idx][0] if event_idx < len(events) else math.inf
            chunk = min(remaining, shift_seconds - t, max(0.0, next_event_start_s - t))
            if chunk <= 0.0:
                continue
            append(state, chunk)
            t += chunk
            remaining -= chunk

    if include_power_on:
        add_with_events("inrush", rng.uniform(2.0, 6.0))
        add_with_events("boot", rng.uniform(60.0, 180.0))
        add_with_events("warmup", rng.uniform(8 * 60.0, 15 * 60.0))
    else:
        add_with_events("idle", rng.uniform(60.0, 180.0))

    while t < shift_seconds:
        if event_idx < len(events) and t >= events[event_idx][0] - 1e-9:
            add_with_events("idle", 0.1)
            continue

        add_with_events("load_unload", rng.uniform(45.0, 140.0))
        num_ops = int(rng.integers(3, 8))
        for _ in range(num_ops):
            if t >= shift_seconds:
                break
            if rng.random() < 0.35:
                add_with_events("tool_change", rng.uniform(7.0, 20.0))
            add_with_events("rapid", rng.uniform(1.0, 12.0))
            add_with_events("cut", rng.uniform(20.0, 180.0))
            if rng.random() < 0.4:
                add_with_events("dwell", rng.uniform(0.5, 4.0))

        if rng.random() < 0.25 and (shift_seconds - t) > 10.0:
            add_with_events("idle", rng.uniform(5.0, 25.0))

        if (shift_seconds - t) < 5.0:
            break

    if segments and sum(s.duration_s for s in segments) > shift_seconds:
        total = sum(s.duration_s for s in segments)
        over = total - shift_seconds
        last = segments[-1]
        segments[-1] = Segment(state=last.state, duration_s=max(0.0, last.duration_s - over))
        segments = [s for s in segments if s.duration_s > 0]

    return segments


@dataclass(frozen=True)
class GlobalProcesses:
    supply_ratio_a: np.ndarray
    supply_ratio_f: np.ndarray
    supply_ratio_phi: np.ndarray
    sensor_offset_a: np.ndarray
    sensor_offset_f: np.ndarray
    sensor_offset_phi: np.ndarray
    phase0: float


def _make_global_processes(rng: np.random.Generator) -> GlobalProcesses:
    supply_ratio_f = rng.uniform(1 / 3600.0, 1 / 60.0, size=5)
    supply_ratio_a = rng.normal(0.0, 0.004, size=5)
    supply_ratio_phi = rng.uniform(-math.pi, math.pi, size=5)

    sensor_offset_f = rng.uniform(1 / 7200.0, 1 / 300.0, size=3)
    sensor_offset_a = rng.normal(0.0, 0.08, size=3)
    sensor_offset_phi = rng.uniform(-math.pi, math.pi, size=3)

    return GlobalProcesses(
        supply_ratio_a=supply_ratio_a,
        supply_ratio_f=supply_ratio_f,
        supply_ratio_phi=supply_ratio_phi,
        sensor_offset_a=sensor_offset_a,
        sensor_offset_f=sensor_offset_f,
        sensor_offset_phi=sensor_offset_phi,
        phase0=float(rng.uniform(-math.pi, math.pi)),
    )


def _supply_voltage_ratio(t_s: np.ndarray, gp: GlobalProcesses) -> np.ndarray:
    ratio = np.ones_like(t_s, dtype=np.float64)
    for a, f, phi in zip(gp.supply_ratio_a, gp.supply_ratio_f, gp.supply_ratio_phi, strict=True):
        ratio += a * np.sin(2.0 * math.pi * f * t_s + phi)
    return _clip(ratio, 0.9, 1.1)


def _sensor_offset_a(t_s: np.ndarray, gp: GlobalProcesses) -> np.ndarray:
    offset = np.zeros_like(t_s, dtype=np.float64)
    for a, f, phi in zip(gp.sensor_offset_a, gp.sensor_offset_f, gp.sensor_offset_phi, strict=True):
        offset += a * np.sin(2.0 * math.pi * f * t_s + phi)
    return offset


def _segment_power_process_w(
    rng: np.random.Generator,
    state: str,
    t_local_s: np.ndarray,
    rated_total_power_w: float,
    base_power_w: float,
) -> np.ndarray:
    n = t_local_s.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if state == "off":
        mean = float(rng.uniform(0.0, 8.0))
        flicker = 0.15 * mean * np.sin(2.0 * math.pi * (1 / 120.0) * t_local_s + rng.uniform(-math.pi, math.pi))
        jitter = rng.normal(0.0, 0.8, size=n)
        return np.maximum(0.0, mean + flicker + jitter)

    if state == "inrush":
        mean = base_power_w * rng.uniform(0.9, 1.2)
        spike = rated_total_power_w * rng.uniform(0.05, 0.15) * np.exp(-t_local_s / rng.uniform(0.6, 1.8))
        return mean + spike

    if state in {"boot", "break"}:
        mean = base_power_w * rng.uniform(0.85, 1.05)
        jitter = rng.normal(0.0, 0.02 * mean, size=n)
        return mean + jitter

    if state == "idle":
        mean = base_power_w * rng.uniform(0.95, 1.1)
        jitter = rng.normal(0.0, 0.03 * mean, size=n)
        return mean + jitter

    if state == "warmup":
        mean = base_power_w * rng.uniform(1.0, 1.15)
        warm_spindle = rated_total_power_w * rng.uniform(0.06, 0.14) * (0.5 + 0.5 * np.sin(2.0 * math.pi * (1 / 40.0) * t_local_s))
        jitter = rng.normal(0.0, 0.02 * mean, size=n)
        return mean + warm_spindle + jitter

    if state == "load_unload":
        mean = base_power_w * rng.uniform(1.0, 1.2)
        p = mean + rng.normal(0.0, 0.02 * mean, size=n)
        num_moves = int(rng.integers(2, 10))
        for _ in range(num_moves):
            center = rng.uniform(0.0, max(0.0, t_local_s[-1]))
            width = rng.uniform(0.15, 1.0)
            amp = rated_total_power_w * rng.uniform(0.01, 0.05)
            p += amp * np.exp(-0.5 * ((t_local_s - center) / width) ** 2)
        return p

    if state == "tool_change":
        mean = base_power_w * rng.uniform(1.05, 1.25)
        p = mean + rng.normal(0.0, 0.02 * mean, size=n)
        p += rated_total_power_w * rng.uniform(0.03, 0.08) * (0.5 + 0.5 * np.sin(2.0 * math.pi * (1 / rng.uniform(1.0, 3.0)) * t_local_s))
        return p

    if state == "rapid":
        mean = base_power_w * rng.uniform(1.1, 1.25)
        servo = rated_total_power_w * rng.uniform(0.05, 0.18)
        dur = max(0.01, float(t_local_s[-1]))
        tri = 1.0 - np.abs(2.0 * (t_local_s / dur) - 1.0)
        p = mean + servo * tri
        p += rng.normal(0.0, 0.03 * (mean + servo), size=n)
        return p

    if state == "cut":
        mean = rated_total_power_w * rng.uniform(0.25, 0.8) + base_power_w
        slow_f = rng.uniform(1 / 30.0, 1 / 6.0)
        fast_f = rng.uniform(0.8, 6.0)
        slow = mean * rng.uniform(0.03, 0.09) * np.sin(2.0 * math.pi * slow_f * t_local_s + rng.uniform(-math.pi, math.pi))
        fast = mean * rng.uniform(0.008, 0.03) * np.sin(2.0 * math.pi * fast_f * t_local_s + rng.uniform(-math.pi, math.pi))
        chatter = mean * rng.uniform(0.0, 0.015) * np.sin(
            2.0 * math.pi * rng.uniform(15.0, 45.0) * t_local_s + rng.uniform(-math.pi, math.pi)
        )
        noise = rng.normal(0.0, 0.01 * mean, size=n)
        p = mean + slow + fast + chatter + noise
        entry_len = min(int(0.35 * n), int(rng.uniform(0.2, 1.5) * 100))
        if entry_len > 0:
            ramp = np.linspace(0.0, 1.0, entry_len, dtype=np.float64)
            p[:entry_len] = base_power_w + (p[:entry_len] - base_power_w) * ramp
        return p

    if state == "dwell":
        mean = base_power_w * rng.uniform(1.0, 1.12)
        jitter = rng.normal(0.0, 0.02 * mean, size=n)
        return mean + jitter

    if state == "alarm":
        mean = base_power_w * rng.uniform(0.95, 1.05)
        jitter = rng.normal(0.0, 0.02 * mean, size=n)
        spike = rated_total_power_w * rng.uniform(0.02, 0.06) * np.exp(-t_local_s / rng.uniform(3.0, 12.0))
        return mean + jitter + spike

    mean = base_power_w
    jitter = rng.normal(0.0, 0.02 * mean, size=n)
    return mean + jitter


def _aux_power_w(
    t_abs_s: np.ndarray,
    state: str,
    rng: np.random.Generator,
    hydro_period_s: float,
    hydro_on_s: float,
    hydro_phase_s: float,
    conveyor_period_s: float,
    conveyor_on_s: float,
    conveyor_phase_s: float,
) -> np.ndarray:
    if state == "off":
        return np.zeros_like(t_abs_s, dtype=np.float64)

    coolant_w = 450.0 if state in {"cut"} else 0.0
    coolant = np.full_like(t_abs_s, coolant_w, dtype=np.float64)

    hydro_phase = (t_abs_s + hydro_phase_s) % hydro_period_s
    hydro_on = hydro_phase < hydro_on_s
    hydro_w = 1400.0
    hydro = hydro_w * hydro_on.astype(np.float64)

    conveyor = np.zeros_like(t_abs_s, dtype=np.float64)
    if state in {"cut", "rapid", "tool_change"}:
        conv_phase = (t_abs_s + conveyor_phase_s) % conveyor_period_s
        conveyor_on = conv_phase < conveyor_on_s
        conveyor_w = 220.0
        conveyor = conveyor_w * conveyor_on.astype(np.float64)

    mist_w = 260.0 if state in {"cut"} and rng.random() < 0.7 else 0.0
    mist = np.full_like(t_abs_s, mist_w, dtype=np.float64)

    return coolant + hydro + conveyor + mist


def _pf_disp_from_load(load_frac: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pf_min = 0.65
    pf_max = 0.96
    pf = pf_min + (pf_max - pf_min) * np.sqrt(_clip(load_frac, 0.0, 1.0))
    pf += rng.normal(0.0, 0.01, size=pf.size)
    return _clip(pf, 0.2, 0.995)


def _thd_from_load(load_frac: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    thd_min = 0.15
    thd_max = 0.55
    thd = thd_max - (thd_max - thd_min) * np.sqrt(_clip(load_frac, 0.0, 1.0))
    thd += rng.normal(0.0, 0.02, size=thd.size)
    return _clip(thd, 0.05, 0.9)


def _override_pf_thd_for_state(
    state: str, t_local_s: np.ndarray, pf: np.ndarray, thd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if state == "off":
        return np.ones_like(pf), np.zeros_like(thd)
    if state == "inrush":
        pf0 = 0.08
        pf1 = 0.55
        tau = 1.2
        pf_inrush = pf1 - (pf1 - pf0) * np.exp(-t_local_s / tau)
        thd_inrush = 0.8 - 0.35 * (1.0 - np.exp(-t_local_s / 1.0))
        return _clip(pf_inrush, 0.05, 0.9), _clip(thd_inrush, 0.2, 0.95)
    if state in {"boot", "break", "idle"}:
        thd2 = _clip(thd * 1.15, 0.1, 0.95)
        pf2 = _clip(pf * 0.9, 0.2, 0.995)
        return pf2, thd2
    return pf, thd


def _synthesize_current_a(
    t_abs_s: np.ndarray,
    p_total_w: np.ndarray,
    v_rms_nominal: float,
    phases: int,
    line_freq_hz: float,
    rated_total_power_w: float,
    rng: np.random.Generator,
    gp: GlobalProcesses,
    state: str,
    t_local_s: np.ndarray,
    harmonics: np.ndarray,
    harmonic_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    v_rms = v_rms_nominal * _supply_voltage_ratio(t_abs_s, gp)

    p_phase_w = p_total_w / float(phases)
    rated_phase_w = rated_total_power_w / float(phases)

    load_frac = _clip(np.abs(p_phase_w) / max(1.0, rated_phase_w), 0.0, 2.0)
    pf_disp = _pf_disp_from_load(load_frac, rng)
    thd = _thd_from_load(load_frac, rng)
    pf_disp, thd = _override_pf_thd_for_state(state, t_local_s, pf_disp, thd)

    sign = np.sign(p_phase_w)
    sign[sign == 0] = 1.0

    eps = 1e-6
    i1_rms = np.abs(p_phase_w) / (v_rms * pf_disp + eps)
    ih_total_rms = thd * i1_rms

    weights = harmonic_weights / np.linalg.norm(harmonic_weights)
    ih_rms = ih_total_rms[:, None] * weights[None, :]

    omega = 2.0 * math.pi * line_freq_hz
    theta = omega * t_abs_s + gp.phase0
    phi = np.arccos(pf_disp)

    i = sign * (math.sqrt(2.0) * i1_rms) * np.sin(theta - phi)
    for idx, h in enumerate(harmonics.tolist()):
        i += (math.sqrt(2.0) * ih_rms[:, idx]) * np.sin(h * (theta - phi))

    i += rng.normal(0.0, 0.02 + 0.004 * np.maximum(0.0, i1_rms), size=i.size)
    i += _sensor_offset_a(t_abs_s, gp)

    meta = {
        "v_rms": v_rms,
        "p_total_w": p_total_w,
        "pf_disp": pf_disp,
        "thd": thd,
        "i1_rms": i1_rms,
    }
    return i, meta


def _iter_shift_rows_csv(
    *,
    rng: np.random.Generator,
    segments: list[Segment],
    target_samples: int,
    start_unix_ms: int,
    fs_hz: float,
    v_rms_nominal: float,
    phases: int,
    line_freq_hz: float,
    rated_total_power_w: float,
    base_power_w: float,
    timestamp_format: Literal["unix_ms", "iso"],
    include_meta: bool,
) -> Iterable[list[str]]:
    dt_s = 1.0 / fs_hz
    ms_per_sample = 1000.0 / fs_hz
    gp = _make_global_processes(rng)

    harmonics = np.array([5, 7, 11, 13, 17, 19], dtype=np.int64)
    harmonic_weights = np.array([1.0, 0.85, 0.35, 0.25, 0.15, 0.1], dtype=np.float64)

    hydro_period_s = float(rng.uniform(45.0, 75.0))
    hydro_on_s = float(rng.uniform(7.0, 16.0))
    hydro_phase_s = float(rng.uniform(0.0, hydro_period_s))
    conveyor_period_s = float(rng.uniform(140.0, 260.0))
    conveyor_on_s = float(rng.uniform(6.0, 14.0))
    conveyor_phase_s = float(rng.uniform(0.0, conveyor_period_s))

    sample_index = 0

    def emit_state(state: str, n: int) -> Iterable[list[str]]:
        nonlocal sample_index
        if n <= 0:
            return

        idx = sample_index + np.arange(n, dtype=np.int64)
        t_abs_s = idx.astype(np.float64) * dt_s
        t_local_s = np.arange(n, dtype=np.float64) * dt_s
        ts_ms = start_unix_ms + np.round(idx.astype(np.float64) * ms_per_sample).astype(np.int64)

        p_process = _segment_power_process_w(
            rng=rng,
            state=state,
            t_local_s=t_local_s,
            rated_total_power_w=rated_total_power_w,
            base_power_w=base_power_w,
        )
        p_aux = _aux_power_w(
            t_abs_s=t_abs_s,
            state=state,
            rng=rng,
            hydro_period_s=hydro_period_s,
            hydro_on_s=hydro_on_s,
            hydro_phase_s=hydro_phase_s,
            conveyor_period_s=conveyor_period_s,
            conveyor_on_s=conveyor_on_s,
            conveyor_phase_s=conveyor_phase_s,
        )
        p_total = np.maximum(0.0, p_process + p_aux)

        i_a, meta = _synthesize_current_a(
            t_abs_s=t_abs_s,
            p_total_w=p_total,
            v_rms_nominal=v_rms_nominal,
            phases=phases,
            line_freq_hz=line_freq_hz,
            rated_total_power_w=rated_total_power_w,
            rng=rng,
            gp=gp,
            state=state,
            t_local_s=t_local_s,
            harmonics=harmonics,
            harmonic_weights=harmonic_weights,
        )

        if timestamp_format == "iso":
            ts = [_iso_from_unix_ms(int(x)) for x in ts_ms.tolist()]
        else:
            ts = [str(int(x)) for x in ts_ms.tolist()]

        if include_meta:
            for row_i in range(n):
                yield [
                    ts[row_i],
                    f"{i_a[row_i]:.6f}",
                    state,
                    f"{meta['p_total_w'][row_i]:.3f}",
                    f"{meta['v_rms'][row_i]:.3f}",
                    f"{meta['pf_disp'][row_i]:.4f}",
                    f"{meta['thd'][row_i]:.4f}",
                ]
        else:
            for row_i in range(n):
                yield [ts[row_i], f"{i_a[row_i]:.6f}"]

        sample_index += n

    for seg in segments:
        if sample_index >= target_samples:
            break
        n = int(round(seg.duration_s * fs_hz))
        if n <= 0:
            continue
        n = min(n, target_samples - sample_index)
        if n <= 0:
            break
        yield from emit_state(seg.state, n)

    if sample_index < target_samples:
        yield from emit_state("idle", target_samples - sample_index)


def _write_csv(out_path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_npz(
    out_path: Path,
    *,
    rng: np.random.Generator,
    segments: list[Segment],
    target_samples: int,
    start_unix_ms: int,
    fs_hz: float,
    v_rms_nominal: float,
    phases: int,
    line_freq_hz: float,
    rated_total_power_w: float,
    base_power_w: float,
) -> None:
    dt_s = 1.0 / fs_hz
    gp = _make_global_processes(rng)

    harmonics = np.array([5, 7, 11, 13, 17, 19], dtype=np.int64)
    harmonic_weights = np.array([1.0, 0.85, 0.35, 0.25, 0.15, 0.1], dtype=np.float64)

    hydro_period_s = float(rng.uniform(45.0, 75.0))
    hydro_on_s = float(rng.uniform(7.0, 16.0))
    hydro_phase_s = float(rng.uniform(0.0, hydro_period_s))
    conveyor_period_s = float(rng.uniform(140.0, 260.0))
    conveyor_on_s = float(rng.uniform(6.0, 14.0))
    conveyor_phase_s = float(rng.uniform(0.0, conveyor_period_s))

    ms_per_sample = 1000.0 / fs_hz
    ts_ms = start_unix_ms + np.round(np.arange(target_samples, dtype=np.float64) * ms_per_sample).astype(np.int64)
    i_a = np.empty(target_samples, dtype=np.float32)

    sample_index = 0
    for seg in segments:
        if sample_index >= target_samples:
            break
        n = int(round(seg.duration_s * fs_hz))
        if n <= 0:
            continue
        n = min(n, target_samples - sample_index)
        if n <= 0:
            break

        idx = sample_index + np.arange(n, dtype=np.int64)
        t_abs_s = idx.astype(np.float64) * dt_s
        t_local_s = np.arange(n, dtype=np.float64) * dt_s

        p_process = _segment_power_process_w(
            rng=rng,
            state=seg.state,
            t_local_s=t_local_s,
            rated_total_power_w=rated_total_power_w,
            base_power_w=base_power_w,
        )
        p_aux = _aux_power_w(
            t_abs_s=t_abs_s,
            state=seg.state,
            rng=rng,
            hydro_period_s=hydro_period_s,
            hydro_on_s=hydro_on_s,
            hydro_phase_s=hydro_phase_s,
            conveyor_period_s=conveyor_period_s,
            conveyor_on_s=conveyor_on_s,
            conveyor_phase_s=conveyor_phase_s,
        )
        p_total = np.maximum(0.0, p_process + p_aux)

        i_seg, _ = _synthesize_current_a(
            t_abs_s=t_abs_s,
            p_total_w=p_total,
            v_rms_nominal=v_rms_nominal,
            phases=phases,
            line_freq_hz=line_freq_hz,
            rated_total_power_w=rated_total_power_w,
            rng=rng,
            gp=gp,
            state=seg.state,
            t_local_s=t_local_s,
            harmonics=harmonics,
            harmonic_weights=harmonic_weights,
        )
        i_a[sample_index : sample_index + n] = i_seg.astype(np.float32)
        sample_index += n

    if sample_index < target_samples:
        n = target_samples - sample_index
        idx = sample_index + np.arange(n, dtype=np.int64)
        t_abs_s = idx.astype(np.float64) * dt_s
        t_local_s = np.arange(n, dtype=np.float64) * dt_s

        p_process = _segment_power_process_w(
            rng=rng,
            state="idle",
            t_local_s=t_local_s,
            rated_total_power_w=rated_total_power_w,
            base_power_w=base_power_w,
        )
        p_aux = _aux_power_w(
            t_abs_s=t_abs_s,
            state="idle",
            rng=rng,
            hydro_period_s=hydro_period_s,
            hydro_on_s=hydro_on_s,
            hydro_phase_s=hydro_phase_s,
            conveyor_period_s=conveyor_period_s,
            conveyor_on_s=conveyor_on_s,
            conveyor_phase_s=conveyor_phase_s,
        )
        p_total = np.maximum(0.0, p_process + p_aux)

        i_seg, _ = _synthesize_current_a(
            t_abs_s=t_abs_s,
            p_total_w=p_total,
            v_rms_nominal=v_rms_nominal,
            phases=phases,
            line_freq_hz=line_freq_hz,
            rated_total_power_w=rated_total_power_w,
            rng=rng,
            gp=gp,
            state="idle",
            t_local_s=t_local_s,
            harmonics=harmonics,
            harmonic_weights=harmonic_weights,
        )
        i_a[sample_index:] = i_seg.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        timestamp_ms=ts_ms,
        current_a=i_a,
        fs_hz=float(fs_hz),
        line_freq_hz=float(line_freq_hz),
        v_rms=float(v_rms_nominal),
        phases=int(phases),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate single-phase mains input current for a CNC over a shift, sampled at 100 Hz.\n\n"
            "This generates *instantaneous* current samples by sampling a synthesized mains current waveform\n"
            "(fundamental + rectifier-like harmonics) whose envelope is driven by a shift-length process model."
        )
    )
    parser.add_argument("--out", type=Path, default=Path("shift_current.csv"))
    parser.add_argument("--out-format", choices=["csv", "npz"], default="csv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling rate (Hz). Default: 100")
    parser.add_argument("--line-freq", type=float, default=60.0, help="Mains frequency (Hz). Typically 50 or 60.")
    parser.add_argument("--v-rms", type=float, default=230.0, help="Per-phase RMS voltage at measurement point.")
    parser.add_argument(
        "--phases",
        type=int,
        default=3,
        choices=[1, 3],
        help="Assume machine is 1φ or balanced 3φ; output is the current of one measured phase.",
    )
    parser.add_argument(
        "--rated-power-w",
        type=float,
        default=18000.0,
        help="Approx. machine rated total real power (W) used to scale load states.",
    )
    parser.add_argument(
        "--base-power-w",
        type=float,
        default=1400.0,
        help="Base always-on power draw (W) for controls/fans/chiller baseline (not including cycling auxiliaries).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help='Shift start time. ISO8601 (e.g. "2026-01-14T08:00:00Z") or unix_ms integer. Default: now (UTC).',
    )
    parser.add_argument("--include-power-on", action="store_true", help="Include power-on inrush/boot/warmup segments.")
    parser.add_argument(
        "--breaks-off",
        action="store_true",
        help="Model scheduled breaks with the machine powered off (near-zero current), followed by a restart sequence.",
    )
    parser.add_argument(
        "--unplanned-off-events",
        type=int,
        default=1,
        help="Number of unplanned power-off downtime events to insert (0 to disable).",
    )
    parser.add_argument(
        "--timestamp-format",
        choices=["unix_ms", "iso"],
        default="unix_ms",
        help="Timestamp column format for CSV output.",
    )
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include extra columns (state, power, voltage, PF, THD).",
    )

    args = parser.parse_args()

    shift_seconds = float(args.hours) * 3600.0
    if shift_seconds <= 0:
        raise SystemExit("--hours must be > 0")

    fs_hz = float(args.fs)
    if fs_hz <= 0:
        raise SystemExit("--fs must be > 0")

    target_samples = int(round(shift_seconds * fs_hz))
    if target_samples <= 0:
        raise SystemExit("shift is too short for the chosen --fs")

    start_unix_ms = _parse_start_time(args.start) if args.start else int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    rng = np.random.default_rng(args.seed)
    segments = _generate_segments(
        rng=rng,
        shift_seconds=shift_seconds,
        include_power_on=bool(args.include_power_on),
        breaks_off=bool(args.breaks_off),
        unplanned_off_events=int(args.unplanned_off_events),
    )

    if args.out_format == "npz":
        _write_npz(
            args.out,
            rng=rng,
            segments=segments,
            target_samples=target_samples,
            start_unix_ms=start_unix_ms,
            fs_hz=fs_hz,
            v_rms_nominal=float(args.v_rms),
            phases=int(args.phases),
            line_freq_hz=float(args.line_freq),
            rated_total_power_w=float(args.rated_power_w),
            base_power_w=float(args.base_power_w),
        )
        return 0

    if args.timestamp_format == "iso":
        ts_col = "timestamp"
    else:
        ts_col = "timestamp_ms"

    if args.include_meta:
        header = [ts_col, "current_a", "state", "p_total_w", "v_rms", "pf_disp", "thd_i_over_i1"]
    else:
        header = [ts_col, "current_a"]

    rows = _iter_shift_rows_csv(
        rng=rng,
        segments=segments,
        target_samples=target_samples,
        start_unix_ms=start_unix_ms,
        fs_hz=fs_hz,
        v_rms_nominal=float(args.v_rms),
        phases=int(args.phases),
        line_freq_hz=float(args.line_freq),
        rated_total_power_w=float(args.rated_power_w),
        base_power_w=float(args.base_power_w),
        timestamp_format=args.timestamp_format,
        include_meta=bool(args.include_meta),
    )
    _write_csv(args.out, header=header, rows=rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
