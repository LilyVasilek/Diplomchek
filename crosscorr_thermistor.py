# -*- coding: utf-8 -*-
"""Кросс-корреляция изотерм (по замечаниям руководителя).

1) Одна термокоса (T4), изотермы на разных глубинах — матрицы r и временного лага.
2) Одна глубина, термокосы T1–T4 — лаг прихода сигнала между косами (цуг ±1 ч).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
DT_SEC = 30.0
WAVE_MIN_HEIGHT_M = 0.5
DEFAULT_SURGE_START = "2023-06-21 21:20"
DEFAULT_SURGE_END = "2023-06-21 23:20"
DEPTH_COVERAGE_MIN = 0.85

# Имена листов по фактической структуре st*.xlsx (st4 отличается от st1–3).
STATIONS = (
    ("st1.xlsx", "dep1", "ss1", "temp1", "T1"),
    ("st2.xlsx", "dep1", "ss", "temp1", "T2"),
    ("st3.xlsx", "dep1", "ss10s", "temp1", "T3"),
    ("st4.xlsx", "dep_n", "ss", "TV", "T4"),
)


def _find_xlsx(name: str) -> Path:
    for folder in (SCRIPT_DIR, SCRIPT_DIR / "term_blocks"):
        p = folder / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Не найден {name} в {SCRIPT_DIR} или term_blocks/")


def common_continuous_interval(*arrays: np.ndarray) -> tuple[int | None, int | None]:
    combined_valid = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        combined_valid &= ~np.isnan(arr)
    padded = np.concatenate(([False], combined_valid, [False]))
    d = np.diff(padded.astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if len(starts) == 0:
        return None, None
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def load_station_30s(xlsx_name: str, sheet_dep: str, sheet_time: str, sheet_temp: str):
    path = _find_xlsx(xlsx_name)
    xl = pd.ExcelFile(path)
    if sheet_dep not in xl.sheet_names:
        raise ValueError(f"{xlsx_name}: нет листа '{sheet_dep}', доступны: {xl.sheet_names}")
    if sheet_time not in xl.sheet_names:
        raise ValueError(f"{xlsx_name}: нет листа времени '{sheet_time}', доступны: {xl.sheet_names}")
    if sheet_temp not in xl.sheet_names:
        raise ValueError(f"{xlsx_name}: нет листа температуры '{sheet_temp}', доступны: {xl.sheet_names}")
    depths = pd.read_excel(path, sheet_name=sheet_dep, header=None).values.astype(float)
    temps = pd.read_excel(path, sheet_name=sheet_temp, header=None).values.astype(float)
    time = pd.to_datetime(pd.read_excel(path, sheet_name=sheet_time, header=None).iloc[:, 0])
    df_t = pd.DataFrame(temps, index=time)
    df_d = pd.DataFrame(depths, index=time)
    grid_t = df_t.resample("30s").mean()
    depths_30s = df_d.resample("30s").mean().values
    temps_30s = grid_t.values
    time_30s = grid_t.index
    median_depths = np.nanmedian(depths_30s, axis=0)
    return time_30s, temps_30s, median_depths, depths_30s


def iso_depth_series(temps_30s: np.ndarray, median_depths: np.ndarray, t_iso: float) -> np.ndarray:
    z_iso = np.full(temps_30s.shape[0], np.nan)
    for t in range(temps_30s.shape[0]):
        row_t = temps_30s[t, :]
        row_z = median_depths
        ok = np.isfinite(row_t) & np.isfinite(row_z)
        if np.sum(ok) < 2:
            continue
        z_iso[t] = interp1d(
            row_t[ok],
            row_z[ok],
            bounds_error=False,
            fill_value=np.nan,
        )(t_iso)
    return z_iso


def temp_at_depth_row(t_row: np.ndarray, z_row: np.ndarray, z_target: float) -> float:
    ok = np.isfinite(t_row) & np.isfinite(z_row)
    if np.sum(ok) < 2:
        return np.nan
    z = z_row[ok]
    t = t_row[ok]
    order = np.argsort(z)
    z, t = z[order], t[order]
    if z_target < z.min() or z_target > z.max():
        return np.nan
    return float(interp1d(z, t, bounds_error=False, fill_value=np.nan)(z_target))


def depth_series_at_z(
    time_30s: pd.DatetimeIndex,
    temps_30s: np.ndarray,
    depths_30s: np.ndarray,
    z_target: float,
) -> np.ndarray:
    out = np.full(len(time_30s), np.nan)
    for i in range(len(time_30s)):
        out[i] = temp_at_depth_row(temps_30s[i, :], depths_30s[i, :], z_target)
    return out


def normalized_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Нормированная кросс-корреляция; возвращает lags, corr, r_max, lag_at_max (в отсчётах)."""
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a[mask], dtype=float)
    b = np.asarray(b[mask], dtype=float)
    n = len(a)
    if n < 16:
        raise ValueError("Слишком мало точек для кросс-корреляции")
    a = a - np.mean(a)
    b = b - np.mean(b)
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        raise ValueError("Нулевая дисперсия ряда")
    a /= sa
    b /= sb
    full = correlate(a, b, mode="full", method="fft") / n
    lags = np.arange(-n + 1, n, dtype=int)
    mid = n - 1
    lo = max(0, mid - max_lag_samples)
    hi = min(len(full), mid + max_lag_samples + 1)
    corr = full[lo:hi]
    lags_trim = lags[lo:hi]
    imax = int(np.argmax(corr))
    return lags_trim, corr, float(corr[imax]), int(lags_trim[imax])


def prepare_anomaly(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    m = np.nanmean(s)
    if not np.isfinite(m):
        m = 0.0
    return s - m


def _fill_short_gaps(series: np.ndarray, max_gap: int = 4) -> np.ndarray:
    s = pd.Series(np.asarray(series, dtype=float))
    return s.interpolate(method="linear", limit=max_gap).values


def _temp_series_at_depth(
    rec: dict,
    t_index: pd.DatetimeIndex,
    z_target: float,
) -> np.ndarray:
    md = rec["md"]
    j = int(np.argmin(np.abs(md - z_target)))
    z_s = float(md[j])
    out = np.full(len(t_index), np.nan)
    idx = rec["time"].get_indexer(t_index)
    for k, ii in enumerate(idx):
        if ii < 0:
            continue
        out[k] = temp_at_depth_row(rec["temps"][ii, :], rec["depths_ts"][ii, :], z_s)
    return out


def detect_waves(z_shifted, dt_seconds, min_period_min=3.0, min_height_m=WAVE_MIN_HEIGHT_M):
    """Волны между соседними минимумами глубины изотермы."""
    minima, _ = find_peaks(-np.asarray(z_shifted, dtype=float))
    waves = []
    for i in range(len(minima) - 1):
        i0, i1 = int(minima[i]), int(minima[i + 1])
        period_min = (i1 - i0) * dt_seconds / 60.0
        if period_min < min_period_min:
            continue
        seg = z_shifted[i0 : i1 + 1]
        imax = i0 + int(np.argmax(seg))
        zmax = z_shifted[imax]
        h_wave = 0.5 * (zmax - z_shifted[i0] + zmax - z_shifted[i1])
        if h_wave >= min_height_m:
            waves.append((i0, i1, imax, float(h_wave), float(period_min)))
    return waves


def _sensor_depths_all_four(
    loaded: dict,
    tol_m: float = 0.5,
    t_index: pd.DatetimeIndex | None = None,
    t0=None,
    t1=None,
    min_frac: float = DEPTH_COVERAGE_MIN,
) -> list[float]:
    """Глубины, где на T1–T4 есть датчик (±tol) и достаточно данных в окне."""
    names = list(loaded.keys())
    if len(names) < 4:
        return []
    ref_depths = np.asarray(loaded["T4"]["md"], dtype=float)
    ref_depths = ref_depths[np.isfinite(ref_depths)]
    matched = []
    for z_ref in ref_depths:
        z_use = [float(z_ref)]
        for name in names:
            if name == "T4":
                continue
            md = np.asarray(loaded[name]["md"], dtype=float)
            md = md[np.isfinite(md)]
            if md.size == 0:
                z_use = []
                break
            j = int(np.argmin(np.abs(md - z_ref)))
            if abs(md[j] - z_ref) > tol_m:
                z_use = []
                break
            z_use.append(float(md[j]))
        if len(z_use) != 4:
            continue
        if t_index is not None and t0 is not None and t1 is not None:
            mask_t = (t_index >= t0) & (t_index <= t1)
            if np.sum(mask_t) < 32:
                continue
            ok_depth = True
            for name, z_s in zip(names, z_use):
                rec = loaded[name]
                idx = rec["time"].get_indexer(t_index[mask_t])
                n_ok = 0
                for ii in idx:
                    if ii < 0:
                        continue
                    v = temp_at_depth_row(
                        rec["temps"][ii, :], rec["depths_ts"][ii, :], z_s,
                    )
                    if np.isfinite(v):
                        n_ok += 1
                if n_ok / max(len(idx), 1) < min_frac:
                    ok_depth = False
                    break
            if not ok_depth:
                continue
        matched.append(float(z_ref))
    return sorted(set(round(z, 2) for z in matched))


def _iso_label(t_c: float) -> str:
    return f"{t_c:.1f} C"


def _annotate_matrix(ax, matrix: np.ndarray, fmt: str = "{:.1f}", text_color: str = "white"):
    """Подписи значений в ячейках матрицы."""
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            txt = fmt.format(val)
            vmax = np.nanmax(np.abs(matrix))
            color = text_color if vmax > 0 and abs(val) > 0.35 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color, fontweight="bold")


def _read_isotherms_st4() -> list[float]:
    print("\n--- T4: три изотермы (общий непрерывный участок) ---")
    iso_in = []
    for i in range(3):
        iso_in.append(float(input(f"  Изотерма {i + 1}, град C: ")))
    iso_values = sorted(iso_in, reverse=True)
    print(f"  Используются (сверху вниз): {[ _iso_label(t) for t in iso_values ]}")
    return iso_values


def _read_max_lag(prompt: str, default: float) -> float:
    raw = input(f"{prompt} (Enter = {default:g}): ").strip()
    return float(raw) if raw else float(default)


def xcorr_st4_ray(iso_values: list[float], max_lag_min: float = 90.0):
    """Кросс-корреляция изотерм st4 на общем участке (разные глубины, одна коса)."""
    time_30s, temps_30s, median_depths, _depths_30s = load_station_30s(
        "st4.xlsx", "dep_n", "ss", "TV"
    )
    iso_depths = {T: iso_depth_series(temps_30s, median_depths, T) for T in iso_values}
    arrays = [iso_depths[T] for T in iso_values]
    c_start, c_end = common_continuous_interval(*arrays)
    if c_start is None:
        raise RuntimeError("Нет общего непрерывного участка для выбранных изотерм st4.")

    seg = slice(c_start, c_end)
    t_seg = time_30s[seg]
    max_lag = int(max_lag_min * 60 / DT_SEC)
    labels = [_iso_label(T) for T in iso_values]
    z_means = [float(np.nanmean(iso_depths[T][seg])) for T in iso_values]
    series = {lb: prepare_anomaly(iso_depths[T][seg]) for lb, T in zip(labels, iso_values)}

    n = len(labels)
    lag_mat = np.full((n, n), np.nan)
    r_mat = np.full((n, n), np.nan)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i == j:
                lag_mat[i, j] = 0.0
                r_mat[i, j] = 1.0
                continue
            try:
                _lags, _corr, r_max, lag_s = normalized_xcorr(
                    series[li], series[lj], max_lag_samples=max_lag,
                )
                lag_mat[i, j] = lag_s * DT_SEC / 60.0
                r_mat[i, j] = r_max
            except ValueError:
                pass

    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            lag_ij = lag_mat[i, j]
            if np.isfinite(lag_ij):
                pair_rows.append({
                    "верх": labels[i],
                    "низ": labels[j],
                    "dz_м": z_means[j] - z_means[i],
                    "лаг_мин": lag_ij,
                    "r": r_mat[i, j],
                })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    im0 = axes[0].imshow(r_mat, vmin=-1, vmax=1, cmap="RdBu_r", origin="upper")
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Слой (глубже)")
    axes[0].set_ylabel("Слой (мельче)")
    axes[0].set_title("Сходство колебаний глубин изотерм, r")
    _annotate_matrix(axes[0], r_mat, fmt="{:.2f}")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    cbar0.set_label("r")

    im1 = axes[1].imshow(lag_mat, cmap="viridis", origin="upper")
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Слой (глубже)")
    axes[1].set_ylabel("Слой (мельче)")
    axes[1].set_title("Лаг при макс. r: мин (+ — глубже позже)")
    _annotate_matrix(axes[1], lag_mat, fmt="{:.0f}")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    cbar1.set_label("мин")
    fig.suptitle(
        f"T4, одна коса — изотермы на разных глубинах\n"
        f"{t_seg[0]:%d.%m.%Y %H:%M} – {t_seg[-1]:%d.%m.%Y %H:%M}, "
        f"шаг 30 с, поиск лага ±{max_lag_min:g} мин",
        fontsize=11,
    )
    out = SCRIPT_DIR / "xcorr_st4_ray_isotherms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report_path = SCRIPT_DIR / "xcorr_st4_report.txt"
    rep = [
        "T4: КРОСС-КОРРЕЛЯЦИЯ ИЗОТЕРМ",
        f"Участок: {t_seg[0]:%d.%m.%Y %H:%M} - {t_seg[-1]:%d.%m.%Y %H:%M}",
        f"Точек: {len(t_seg)}, макс. сдвиг: +/-{max_lag_min:g} мин",
        "",
        "Средняя глубина изотермы:",
    ]
    for lb, zm in zip(labels, z_means):
        rep.append(f"  {lb}: {zm:.2f} м")
    rep.append("")
    rep.append("Как читать матрицы:")
    rep.append("  Строка — более мелкий (теплее) слой, столбец — более глубокий.")
    rep.append("  Лаг > 0: глубокий слой смещается позже мелкого (фаза с глубиной).")
    rep.append("")
    if pair_rows:
        rep.append("Пары слоёв (мелкий → глубокий):")
        for pr in pair_rows:
            rep.append(
                f"  {pr['верх']} → {pr['низ']}:  Δz={pr['dz_м']:.2f} м,  "
                f"лаг={pr['лаг_мин']:+.1f} мин,  r={pr['r']:.2f}"
            )
        rep.append("")
    rep.append("Матрица лага, мин (строка i, столбец j):")
    hdr = "        " + "".join(f"{lb:>10s}" for lb in labels)
    rep.append(hdr)
    for i, li in enumerate(labels):
        row = f"  {li:>8s}"
        for j in range(n):
            v = lag_mat[i, j]
            row += f"{v:10.1f}" if np.isfinite(v) else "       n/a"
        rep.append(row)
    rep.append("")
    rep.append("Матрица корреляции r:")
    rep.append(hdr)
    for i, li in enumerate(labels):
        row = f"  {li:>8s}"
        for j in range(n):
            v = r_mat[i, j]
            row += f"{v:10.2f}" if np.isfinite(v) else "       n/a"
        rep.append(row)
    rep.append(f"\nГрафик: {out}")
    report_path.write_text("\n".join(rep), encoding="utf-8")

    print("\n" + "=" * 60)
    print("T4: РЕЗУЛЬТАТ")
    print("=" * 60)
    print("\n".join(rep))
    return lag_mat, r_mat, labels, t_seg


def _read_surge_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    print("\n--- Короткий участок: цуг на T4 ---")
    raw_s = input(
        f"  Начало цуга (ГГГГ-ММ-ДД ЧЧ:ММ, Enter = {DEFAULT_SURGE_START}): "
    ).strip()
    raw_e = input(
        f"  Конец цуга (ГГГГ-ММ-ДД ЧЧ:ММ, Enter = {DEFAULT_SURGE_END}): "
    ).strip()
    t0 = pd.to_datetime(raw_s) if raw_s else pd.to_datetime(DEFAULT_SURGE_START)
    t1 = pd.to_datetime(raw_e) if raw_e else pd.to_datetime(DEFAULT_SURGE_END)
    if t1 <= t0:
        raise ValueError("Конец цуга должен быть позже начала.")
    print(f"  Цуг: {t0:%d.%m.%Y %H:%M} — {t1:%d.%m.%Y %H:%M}")
    return t0, t1


def _plot_surge_field_t4(
    time_30s: pd.DatetimeIndex,
    temps_30s: np.ndarray,
    median_depths: np.ndarray,
    surge_start: pd.Timestamp,
    surge_end: pd.Timestamp,
    z_top: float,
    z_max: float = 18.0,
) -> Path:
    """Три панели: час до / цуг / час после (ширина 1:2:1)."""
    windows = [
        (surge_start - pd.Timedelta(hours=1), surge_start, "час до"),
        (surge_start, surge_end, "цуг"),
        (surge_end, surge_end + pd.Timedelta(hours=1), "час после"),
    ]
    hours = [(w1 - w0).total_seconds() / 3600.0 for w0, w1, _ in windows]
    side = hours[0] if hours[0] > 0 else 1.0
    widths = [max(0.2, h / side) for h in hours] + [0.07]
    fig = plt.figure(figsize=(11.5, 5.0))
    gs = fig.add_gridspec(1, 4, width_ratios=widths, wspace=0.38)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    chunks = []
    for w0, w1, _ in windows:
        m = (time_30s >= w0) & (time_30s <= w1)
        if np.any(m):
            chunks.append(temps_30s[m, :])
    if chunks:
        t_lo = float(np.nanpercentile(np.concatenate(chunks), 2))
        t_hi = float(np.nanpercentile(np.concatenate(chunks), 98))
    else:
        t_lo, t_hi = 14.0, 20.0
    levels = np.arange(int(np.floor(t_lo)), int(np.ceil(t_hi)) + 1, 1.0)
    last_cf = None
    for j, (ax, (w0, w1, cap)) in enumerate(zip(axes, windows)):
        m = (time_30s >= w0) & (time_30s <= w1)
        if not np.any(m):
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center", transform=ax.transAxes)
            continue
        t_sel = time_30s[m]
        TT, DD = np.meshgrid(t_sel, median_depths)
        last_cf = ax.contourf(TT, DD, temps_30s[m, :].T, levels=levels, cmap="RdYlBu_r", extend="both")
        ax.invert_yaxis()
        ax.set_title(f"{cap}\n{w0:%H:%M}–{w1:%H:%M}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if j == 0:
            ax.set_ylabel("Глубина, м")
        ax.set_ylim(z_max, z_top)
        if j > 0:
            ax.tick_params(labelleft=False)
    if last_cf is not None:
        fig.colorbar(last_cf, cax=cax, label="T, °C")
    fig.suptitle(
        f"T4: поле вокруг цуга ({surge_start:%d.%m.%Y %H:%M}–{surge_end:%H:%M})",
        fontsize=11,
    )
    out = SCRIPT_DIR / "xcorr_surge_field_T4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_surge_isotherms_t4(
    time_30s: pd.DatetimeIndex,
    iso_depths: dict[float, np.ndarray],
    surge_start: pd.Timestamp,
    surge_end: pd.Timestamp,
) -> Path:
    m = (time_30s >= surge_start) & (time_30s <= surge_end)
    t = time_30s[m]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for t_iso, z in iso_depths.items():
        ax.plot(t, z[m], lw=1.0, label=_iso_label(t_iso))
    ax.invert_yaxis()
    ax.set_ylabel("Глубина изотермы, м")
    ax.set_xlabel("Время")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.set_title(f"T4: изотермы только в цуге ({surge_start:%H:%M}–{surge_end:%H:%M})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = SCRIPT_DIR / "xcorr_surge_isotherms_T4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def xcorr_stations_surge(
    iso_values: list[float],
    wave_iso: float,
    surge_start: pd.Timestamp,
    surge_end: pd.Timestamp,
    max_lag_min: float = 60.0,
    depth_tol_m: float = 0.5,
):
    """Задержки между T1–T4: цуг, общие глубины датчиков, сопоставление волн T4."""
    loaded = {}
    for xlsx, sh_dep, sh_time, sh_temp, title in STATIONS:
        t, temps, md, depths_ts = load_station_30s(xlsx, sh_dep, sh_time, sh_temp)
        loaded[title] = {"time": t, "temps": temps, "depths_ts": depths_ts, "md": md}

    t4 = loaded["T4"]
    z_top = float(np.nanmin(t4["md"]))
    _plot_surge_field_t4(
        t4["time"], t4["temps"], t4["md"], surge_start, surge_end, z_top=z_top,
    )
    print("  Сохранено: xcorr_surge_field_T4.png")

    iso_depths_t4 = {
        T: iso_depth_series(t4["temps"], t4["md"], T) for T in iso_values
    }
    _plot_surge_isotherms_t4(t4["time"], iso_depths_t4, surge_start, surge_end)
    print("  Сохранено: xcorr_surge_isotherms_T4.png")

    t_win = pd.date_range(
        surge_start - pd.Timedelta(hours=1),
        surge_end + pd.Timedelta(hours=1),
        freq="30s",
    )
    t_common = t4["time"]
    for rec in loaded.values():
        t_common = t_common.intersection(rec["time"])
    t_common = t_common.intersection(t_win)
    if len(t_common) < 64:
        raise RuntimeError("Мало общего времени в окне ±1 ч вокруг цуга.")

    depth_list = _sensor_depths_all_four(
        loaded, tol_m=depth_tol_m, t_index=t_common,
        t0=surge_start, t1=surge_end, min_frac=DEPTH_COVERAGE_MIN,
    )
    if not depth_list:
        raise RuntimeError(
            "Нет глубин с датчиками на всех четырёх косах и достаточным покрытием в цуге."
        )
    print(f"  Глубины (датчик на T1–T4), м: {depth_list}")

    m_surge = (t_common >= surge_start) & (t_common <= surge_end)
    t_surge = t_common[m_surge]
    z_wave = iso_depth_series(t4["temps"], t4["md"], wave_iso)
    idx_c = t4["time"].get_indexer(t_common)
    z_seg = np.asarray(z_wave[idx_c], dtype=float)[m_surge]
    z_shift = z_seg - np.nanmean(z_seg)
    waves = detect_waves(z_shift, dt_seconds=DT_SEC)
    print(f"  Волн на T4 ({_iso_label(wave_iso)}) в цуге: {len(waves)}")

    z_main = depth_list[len(depth_list) // 2]
    depth_scores = []
    for z_ref in depth_list:
        fracs = []
        for name in ("T1", "T2", "T3", "T4"):
            s = _temp_series_at_depth(loaded[name], t_surge, z_ref)
            fracs.append(float(np.sum(np.isfinite(s))) / max(len(s), 1))
        depth_scores.append((min(fracs), z_ref))
    z_main = max(depth_scores)[1]
    print(f"  Глубина для сопоставления волн (макс. покрытие): {z_main:.2f} м")

    names = ["T1", "T2", "T3", "T4"]
    temp_surge = {
        name: _fill_short_gaps(_temp_series_at_depth(loaded[name], t_surge, z_main))
        for name in names
    }
    max_lag = int(max_lag_min * 60 / DT_SEC)
    wave_rows = []
    for wnum, (i0, i1, imax, h_m, period_min) in enumerate(waves, start=1):
        half = max(int(0.5 * (i1 - i0)), 8)
        w0 = max(0, imax - half)
        w1 = min(len(z_shift), imax + half + 1)
        ref = prepare_anomaly(temp_surge["T4"][w0:w1])
        t_evt = t_surge[imax]
        for other in ("T1", "T2", "T3"):
            seg = prepare_anomaly(temp_surge[other][w0:w1])
            try:
                _lags, _corr, r_max, lag_s = normalized_xcorr(
                    ref, seg, max_lag_samples=max_lag,
                )
                lag_min = lag_s * DT_SEC / 60.0
            except ValueError:
                lag_min, r_max = np.nan, np.nan
            wave_rows.append({
                "волна": wnum,
                "время_T4": t_evt,
                "H_м": h_m,
                "T_мин": period_min,
                "глубина_м": z_main,
                "пара": f"T4–{other}",
                "лаг_мин": lag_min,
                "r": r_max,
            })

    if wave_rows:
        df_w = pd.DataFrame(wave_rows)
        print("\n" + "=" * 60)
        print("СОПОСТАВЛЕНИЕ ВОЛН ЦУГА МЕЖДУ КОСАМИ")
        print("=" * 60)
        print("Положительный лаг: сигнал на второй косе позже T4.")
        print(df_w.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        p_w = SCRIPT_DIR / "xcorr_surge_waves.csv"
        df_w.to_csv(p_w, index=False, encoding="utf-8-sig")
        print(f"\nCSV волн: {p_w}")

    series_full = {
        name: prepare_anomaly(_fill_short_gaps(_temp_series_at_depth(loaded[name], t_common, z_main)))
        for name in names
    }

    n_st = len(names)
    lag_mat = np.full((n_st, n_st), np.nan)
    r_mat = np.full((n_st, n_st), np.nan)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                lag_mat[i, j] = 0.0
                r_mat[i, j] = 1.0
                continue
            try:
                _lags, _corr, r_max, lag_s = normalized_xcorr(
                    series_full[ni], series_full[nj], max_lag_samples=max_lag,
                )
                lag_mat[i, j] = lag_s * DT_SEC / 60.0
                r_mat[i, j] = r_max
            except ValueError:
                pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(r_mat, vmin=-1, vmax=1, cmap="RdBu_r", origin="upper")
    axes[0].set_xticks(range(n_st))
    axes[0].set_yticks(range(n_st))
    axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].set_yticklabels(names)
    axes[0].set_title(f"Одна глубина {z_main:.1f} м — сходство T на косах")
    _annotate_matrix(axes[0], r_mat, fmt="{:.2f}")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    cb0.set_label("r")
    im1 = axes[1].imshow(lag_mat, cmap="viridis", origin="upper")
    axes[1].set_xticks(range(n_st))
    axes[1].set_yticks(range(n_st))
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    axes[1].set_yticklabels(names)
    axes[1].set_title("Лаг при макс. r, мин (+ — столбец позже)")
    _annotate_matrix(axes[1], lag_mat, fmt="{:.0f}")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    cb1.set_label("мин")
    fig.suptitle(
        f"T1–T4, глубина {z_main:.1f} м\n"
        f"{t_common[0]:%d.%m %H:%M} – {t_common[-1]:%H:%M}, цуг "
        f"{surge_start:%H:%M}–{surge_end:%H:%M}",
        fontsize=10,
    )
    tag = f"{z_main:.1f}".replace(".", "p")
    out = SCRIPT_DIR / f"xcorr_surge_z{tag}m.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Матрица на {z_main:.1f} м: {out.name}")


def xcorr_stations_same_depth(
    depth_list: list[float] | None = None,
    max_lag_min: float = 120.0,
    depth_tol_m: float = 1.0,
):
    """Кросс-корреляция T(z,t) на одной глубине между термокосами."""
    loaded = {}
    for xlsx, sh_dep, sh_time, sh_temp, title in STATIONS:
        try:
            t, temps, md, depths_ts = load_station_30s(xlsx, sh_dep, sh_time, sh_temp)
            loaded[title] = {"time": t, "temps": temps, "depths_ts": depths_ts, "md": md}
        except FileNotFoundError:
            print(f"  Пропуск {title}: нет файла {xlsx}")

    if len(loaded) < 2:
        raise RuntimeError("Нужно минимум 2 станции с данными st*.xlsx")

    t_common = None
    for rec in loaded.values():
        t_common = rec["time"] if t_common is None else t_common.intersection(rec["time"])
    if t_common is None or len(t_common) < 128:
        raise RuntimeError("Слишком мало общего времени между станциями.")

    if depth_list is None:
        depth_list = _sensor_depths_all_four(
            loaded, tol_m=depth_tol_m, t_index=t_common,
            t0=t_common[0], t1=t_common[-1],
        )
        if not depth_list:
            raise RuntimeError(
                "Нет глубин с датчиками на всех четырёх косах. Задайте глубины вручную."
            )
        print(f"  Глубины (T1–T4), м: {[round(z, 1) for z in depth_list]}")

    names = list(loaded.keys())
    max_lag = int(max_lag_min * 60 / DT_SEC)
    summary_rows = []

    for z_target in depth_list:
        series = {}
        for name, rec in loaded.items():
            idx = rec["time"].get_indexer(t_common)
            temps_al = rec["temps"][idx, :]
            depths_al = rec["depths_ts"][idx, :]
            t_series = np.array(
                [temp_at_depth_row(temps_al[i], depths_al[i], z_target) for i in range(len(t_common))],
                dtype=float,
            )
            series[name] = prepare_anomaly(t_series)

        n_st = len(names)
        lag_mat = np.full((n_st, n_st), np.nan)
        r_mat = np.full((n_st, n_st), np.nan)
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i == j:
                    lag_mat[i, j] = 0.0
                    r_mat[i, j] = 1.0
                    continue
                try:
                    _lags, _corr, r_max, lag_s = normalized_xcorr(
                        series[ni], series[nj], max_lag_samples=max_lag,
                    )
                    lag_mat[i, j] = lag_s * DT_SEC / 60.0
                    r_mat[i, j] = r_max
                    if i < j:
                        summary_rows.append(
                            {
                                "z_m": z_target,
                                "станция_A": ni,
                                "станция_B": nj,
                                "лаг_мин": lag_s * DT_SEC / 60.0,
                                "r_max": r_max,
                            }
                        )
                except ValueError:
                    pass

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        im0 = axes[0].imshow(r_mat, vmin=-1, vmax=1, cmap="RdBu_r", origin="upper")
        axes[0].set_xticks(range(n_st))
        axes[0].set_yticks(range(n_st))
        axes[0].set_xticklabels(names, rotation=35, ha="right")
        axes[0].set_yticklabels(names)
        axes[0].set_xlabel("Термокоса (столбец)")
        axes[0].set_ylabel("Термокоса (строка)")
        axes[0].set_title(f"Корреляция, глубина {z_target:.1f} м")
        _annotate_matrix(axes[0], r_mat, fmt="{:.2f}")
        cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
        cbar0.set_label("r")

        im1 = axes[1].imshow(lag_mat, cmap="viridis", origin="upper")
        axes[1].set_xticks(range(n_st))
        axes[1].set_yticks(range(n_st))
        axes[1].set_xticklabels(names, rotation=35, ha="right")
        axes[1].set_yticklabels(names)
        axes[1].set_xlabel("Термокоса (столбец)")
        axes[1].set_ylabel("Термокоса (строка)")
        axes[1].set_title("Сдвиг при макс. корреляции, мин")
        _annotate_matrix(axes[1], lag_mat, fmt="{:.0f}")
        cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
        cbar1.set_label("мин")
        fig.suptitle(
            f"T1–T4, одна глубина {z_target:.1f} м\n"
            f"{t_common[0]:%d.%m.%Y %H:%M} – {t_common[-1]:%H:%M}, лаг ±{max_lag_min:g} мин",
            fontsize=10,
        )
        tag = f"{z_target:.1f}".replace(".", "p")
        out = SCRIPT_DIR / f"xcorr_stations_z{tag}m.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Глубина {z_target:.1f} м: сохранено {out.name}")

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df = df.rename(columns={
            "z_m": "Глубина_м",
            "станция_A": "T_A",
            "станция_B": "T_B",
            "лаг_мин": "Сдвиг_мин",
            "r_max": "Корреляция_r",
        })
        print("\n" + "=" * 60)
        print("T1-T4: СВОДКА СДВИГОВ")
        print("=" * 60)
        print("Сдвиг, мин: положительный - T_B позже T_A.")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        csv_path = SCRIPT_DIR / "xcorr_stations_lags.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nТаблица CSV: {csv_path}")


def _print_main_menu():
    print("\n" + "=" * 60)
    print("КРОСС-КОРРЕЛЯЦИЯ ТЕРМОКОС")
    print("=" * 60)
    print("  1 - T4: одна коса, изотермы на разных глубинах (лаг, r)")
    print("  2 - T1–T4: одна глубина, разные косы, цуг ±1 ч (задержка прихода)")
    print("  3 - Оба расчета")
    print("  0 - Выход")


def _ask_mode() -> int:
    _print_main_menu()
    while True:
        raw = input("Выберите пункт (0-3, Enter = 3): ").strip() or "3"
        if raw in ("0", "1", "2", "3"):
            return int(raw)
        print("  Введите 0, 1, 2 или 3.")


def run_st4_block():
    iso_values = _read_isotherms_st4()
    max_lag_ray = _read_max_lag("  Макс. сдвиг по времени, мин", 90.0)
    xcorr_st4_ray(iso_values, max_lag_min=max_lag_ray)


def run_stations_block():
    surge_start, surge_end = _read_surge_window()
    iso_values = _read_isotherms_st4()
    raw_w = input(
        f"  Изотерма для поиска волн (Enter = {iso_values[1]:.1f} C): "
    ).strip()
    wave_iso = float(raw_w) if raw_w else float(iso_values[1])
    max_lag_st = _read_max_lag("  Макс. сдвиг по времени, мин", 60.0)
    xcorr_stations_surge(
        iso_values=iso_values,
        wave_iso=wave_iso,
        surge_start=surge_start,
        surge_end=surge_end,
        max_lag_min=max_lag_st,
    )


def run_crosscorr(mode: int | None = None) -> None:
    """Запуск кросс-корреляции (mode 0–3; None — интерактивное меню)."""
    if mode is None:
        mode = _ask_mode()
    if mode == 0:
        print("Выход.")
        return
    if mode in (1, 3):
        run_st4_block()
    if mode in (2, 3):
        run_stations_block()
    print("\nКросс-корреляция: готово.")


def main():
    run_crosscorr()


if __name__ == "__main__":
    main()
