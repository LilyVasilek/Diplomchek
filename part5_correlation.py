# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import correlate, detrend

# =========================================================
# 15. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ТЕРМОКОС
#     A) Автокорреляции по глубинным уровням
#     B) Кросс-корреляции внутри каждой термокосы
#     C) Кросс-корреляции между термокосами (полная запись)
#     D) Цуг 20.06 23:30–21.06 01:00 — синхронные ряды + CCF + лаги
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DT_SECONDS = 30.0
MAX_LAG_MIN = 4000.0
MAX_LAG_STEPS = int((MAX_LAG_MIN * 60.0) / DT_SECONDS)

# Окно цуга (как в part4_short_window).
SHORT_SURGE_START = pd.to_datetime("2023-06-20 23:30")
SHORT_SURGE_END = pd.to_datetime("2023-06-21 01:00")
REF_STATION = "T4"  # опорная термокоса для фазовых сдвигов

# Цуг смотрим в зоне термоклина (10–13 м): общая глубина — середина интервала.
SURGE_Z_THERMOCLINE_LO_M = 10.0
SURGE_Z_THERMOCLINE_HI_M = 13.0
SURGE_Z_REF_M = 0.5 * (SURGE_Z_THERMOCLINE_LO_M + SURGE_Z_THERMOCLINE_HI_M)

# Имена листов по фактической структуре st*.xlsx (st4 отличается от st1–3).
STATIONS = (
    ("st1.xlsx", "dep1", "ss1", "temp1", "T1"),
    ("st2.xlsx", "dep1", "ss", "temp1", "T2"),
    ("st3.xlsx", "dep1", "ss10s", "temp1", "T3"),
    ("st4.xlsx", "dep_n", "ss", "TV", "T4"),
)

_LEVEL_ROLES = (
    "мелководный датчик (ближе к поверхности)",
    "среднеглубинный датчик",
    "глубинный датчик (ближе ко дну)",
)


def _level_label(sensor_idx, depth_m, role):
    """Подпись уровня: номер датчика, глубина и физический смысл."""
    n = int(sensor_idx) + 1
    if np.isfinite(depth_m):
        return f"датчик №{n}, z = {depth_m:.1f} м\n({role})"
    return f"датчик №{n}\n({role})"


def _pair_label(st, i0, i1):
    """Краткая подпись пары для легенды (без переносов)."""
    z0 = st["level_depths"][i0]
    z1 = st["level_depths"][i1]
    n0, n1 = st["level_sensor_n"][i0], st["level_sensor_n"][i1]
    if np.isfinite(z0) and np.isfinite(z1):
        return f"№{n0} ({z0:.1f} м) — №{n1} ({z1:.1f} м)"
    return f"№{n0} — №{n1}"


def _load_station_frames(xlsx_name, dep_sheet, time_sheet, temp_sheet):
    xlsx_path = os.path.join(BASE_DIR, xlsx_name)
    if not os.path.isfile(xlsx_path):
        raise FileNotFoundError(f"Нет файла: {xlsx_path}")

    temps = pd.read_excel(xlsx_path, sheet_name=temp_sheet, header=None)
    times = pd.to_datetime(
        pd.read_excel(xlsx_path, sheet_name=time_sheet, header=None).iloc[:, 0],
        errors="coerce",
    )
    depths = pd.read_excel(xlsx_path, sheet_name=dep_sheet, header=None)

    valid_time = times.notna()
    if not np.any(valid_time):
        raise ValueError(f"В {xlsx_name} нет валидной временной оси.")

    temps = temps.loc[valid_time].reset_index(drop=True)
    depths = depths.loc[valid_time].reset_index(drop=True)
    times = times.loc[valid_time].reset_index(drop=True)

    n_cols = int(min(temps.shape[1], depths.shape[1]))
    if n_cols < 1:
        raise ValueError(f"В {xlsx_name} нет столбцов температур/глубин.")

    temps = temps.iloc[:, :n_cols]
    depths = depths.iloc[:, :n_cols]

    df_t = pd.DataFrame(temps.values.astype(float), index=times)
    df_d = pd.DataFrame(depths.values.astype(float), index=times)

    df_t = df_t.groupby(level=0).mean().sort_index()
    df_d = df_d.groupby(level=0).mean().sort_index()

    df_t_30 = df_t.resample("30s").mean()
    df_d_30 = df_d.resample("30s").mean()
    med_depths = np.nanmedian(df_d_30.values, axis=0)
    return df_t_30, med_depths


def _pick_level_indices(med_depths):
    z = np.asarray(med_depths, dtype=float)
    valid = np.where(np.isfinite(z))[0]
    if valid.size < 3:
        raise ValueError("Недостаточно датчиков с валидной глубиной (нужно >= 3).")

    z_valid = z[valid]
    i_top = valid[int(np.argmin(z_valid))]
    i_bot = valid[int(np.argmax(z_valid))]
    z_mid_target = float(0.5 * (z_valid.min() + z_valid.max()))
    i_mid = valid[int(np.argmin(np.abs(z_valid - z_mid_target)))]

    idx = [i_top, i_mid, i_bot]
    if len(set(idx)) < 3:
        ordered = list(valid[np.argsort(z_valid)])
        idx = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]

    if len(set(idx)) < 3:
        uniq = []
        for j in valid[np.argsort(z_valid)]:
            if j not in uniq:
                uniq.append(int(j))
            if len(uniq) == 3:
                break
        idx = uniq
    return tuple(int(i) for i in idx)


def _series_at_depth(df_t_30, med_depths, z_ref):
    """T(z_ref, t) — линейная интерполяция по вертикальному профилю на каждом шаге."""
    z = np.asarray(med_depths, dtype=float)
    z_ref = float(z_ref)
    out = np.full(len(df_t_30), np.nan)
    for i, row in enumerate(df_t_30.to_numpy(dtype=float)):
        m = np.isfinite(row) & np.isfinite(z)
        if np.count_nonzero(m) < 2:
            continue
        zv, tv = z[m], row[m]
        order = np.argsort(zv)
        zv, tv = zv[order], tv[order]
        if z_ref < zv[0] or z_ref > zv[-1]:
            continue
        out[i] = np.interp(z_ref, zv, tv)
    return out


def _closest_sensor_note(med_depths, z_ref):
    z = np.asarray(med_depths, dtype=float)
    m = np.isfinite(z)
    if not np.any(m):
        return "—"
    idx = int(np.where(m)[0][np.argmin(np.abs(z[m] - z_ref))])
    return f"ближ. датчик №{idx + 1}, z = {z[idx]:.1f} м"


def _fill_series(s):
    x = pd.Series(np.asarray(s, dtype=float))
    x = x.interpolate(limit_direction="both")
    x = x.ffill().bfill()
    return x.to_numpy(dtype=float)


def _prepare_series(s, *, detrend_linear=True, normalize=True):
    """Ряд для корреляции: интерполяция пропусков, detrend, нормировка на σ."""
    arr = _fill_series(s)
    if arr.size < 8 or np.nanstd(arr) <= 0:
        return None
    if detrend_linear:
        arr = detrend(arr, type="linear")
    if normalize:
        std = float(np.std(arr))
        if std <= 0:
            return None
        arr = arr / std
    return arr


def _anomaly_for_plot(s):
    """Температурная аномалия для графика (°C): линейный тренд снят, без деления на σ."""
    arr = _fill_series(s)
    if arr.size < 2:
        return None
    return detrend(arr, type="linear")


def _normalized_xcorr(x, y, max_lag_steps):
    """
    Нормированная взаимная корреляция R(τ).
    τ = 0 — совпадение по времени; |R| ≤ 1.
    Положительный τ: второй ряд (y) сдвинут вперёд относительно первого (x),
    т.е. пик y наступает позже → x опережает y.
    """
    n = int(min(len(x), len(y)))
    if n < 8:
        return None, None
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)
    den = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if den <= 0:
        return None, None
    c_full = correlate(x, y, mode="full", method="auto") / den
    lags = np.arange(-n + 1, n, dtype=int)
    m = np.abs(lags) <= int(max_lag_steps)
    return lags[m], c_full[m]


def _lag_minutes(lags):
    return (lags.astype(float) * DT_SECONDS) / 60.0


def _peak_lag_info(lags, ccf):
    i_peak = int(np.nanargmax(np.abs(ccf)))
    lag_min = float(_lag_minutes(lags)[i_peak])
    r_peak = float(ccf[i_peak])
    return lag_min, r_peak, float(np.abs(ccf[i_peak]))


def _fmt_surge_axis(ax, t0, t1):
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelsize=8)


# ---------------------------------------------------------
# Загрузка всех станций
# ---------------------------------------------------------
station_data = []
for xlsx, dep_sheet, time_sheet, temp_sheet, title in STATIONS:
    try:
        df_t_30, med_depths = _load_station_frames(
            xlsx, dep_sheet, time_sheet, temp_sheet,
        )
        idx_top, idx_mid, idx_bot = _pick_level_indices(med_depths)
        idx_triplet = (idx_top, idx_mid, idx_bot)

        level_series = []
        level_text = []
        level_depths = []
        level_sensor_n = []
        for role, idx in zip(_LEVEL_ROLES, idx_triplet):
            z = float(med_depths[idx]) if np.isfinite(med_depths[idx]) else np.nan
            raw = df_t_30.iloc[:, idx].to_numpy(dtype=float)
            level_series.append(_prepare_series(raw))
            level_text.append(_level_label(idx, z, role))
            level_depths.append(z)
            level_sensor_n.append(int(idx) + 1)

        mean_raw = np.nanmean(df_t_30.to_numpy(dtype=float), axis=1)
        mean_series = _prepare_series(mean_raw)

        station_data.append(
            {
                "title": title,
                "file": xlsx,
                "time": df_t_30.index,
                "df_t": df_t_30,
                "med_depths": med_depths,
                "mean_raw": mean_raw,
                "levels": level_series,
                "level_labels": level_text,
                "level_depths": level_depths,
                "level_sensor_n": level_sensor_n,
                "mean_series": mean_series,
            }
        )
        depths_str = ", ".join(
            f"№{n} ({z:.1f} м)" if np.isfinite(z) else f"№{n}"
            for n, z in zip(level_sensor_n, level_depths)
        )
        print(f"{title}: N = {len(df_t_30)}, уровни: {depths_str}")
    except Exception as exc:
        print(f"{title}: пропуск ({exc}).")

if not station_data:
    raise RuntimeError("Не удалось загрузить данные ни по одной термокосе.")

SURGE_Z_REF_M = float(SURGE_Z_REF_M)
for st in station_data:
    st["t_at_zref"] = _series_at_depth(st["df_t"], st["med_depths"], SURGE_Z_REF_M)
print(
    f"Цуг: зона термоклина z = {SURGE_Z_THERMOCLINE_LO_M:.0f}–{SURGE_Z_THERMOCLINE_HI_M:.0f} м, "
    f"сравнение на z = {SURGE_Z_REF_M:.1f} м (интерполяция T по профилю)"
)

# ---------------------------------------------------------
# A) Автокорреляции
# ---------------------------------------------------------
fig_a, axs_a = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs_a = axs_a.ravel()
colors = ("red", "blue", "limegreen")

for i, st in enumerate(station_data[:4]):
    ax = axs_a[i]
    drawn = 0
    for j, s in enumerate(st["levels"]):
        if s is None:
            continue
        lags, acf = _normalized_xcorr(s, s, MAX_LAG_STEPS)
        if lags is None:
            continue
        ax.plot(_lag_minutes(lags), acf, lw=1.2, color=colors[j], label=st["level_labels"][j])
        drawn += 1
    ax.axhline(0.0, color="magenta", lw=1.0)
    ax.set_title(f"{st['title']} — автокорреляция T′")
    ax.grid(True, alpha=0.3)
    if drawn > 0:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

for k in range(len(station_data), 4):
    axs_a[k].set_visible(False)

for ax in axs_a:
    ax.set_xlim(-MAX_LAG_MIN, MAX_LAG_MIN)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Временной лаг τ, мин")
    ax.set_ylabel("R(τ)")

fig_a.suptitle(
    "Автокорреляция температурных аномалий на трёх глубинах\n"
    "(мелководный / среднеглубинный / глубинный датчик)",
    y=1.01,
    fontsize=11,
)
fig_a.tight_layout(rect=(0, 0, 1, 0.96))
out_a = os.path.join(BASE_DIR, "st4_fig13_autocorr_thermistor_levels.png")
fig_a.savefig(out_a, dpi=170, bbox_inches="tight")
plt.close(fig_a)
print(f"Автокорреляции: {out_a}")

# ---------------------------------------------------------
# B) Кросс-корреляции внутри каждой термокосы
# ---------------------------------------------------------
fig_b, axs_b = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs_b = axs_b.ravel()
pairs = ((0, 1), (0, 2), (1, 2))
pair_colors = ("#8e44ad", "#d35400", "#16a085")

for i, st in enumerate(station_data[:4]):
    ax = axs_b[i]
    for c, (i0, i1) in zip(pair_colors, pairs):
        s0, s1 = st["levels"][i0], st["levels"][i1]
        if s0 is None or s1 is None:
            continue
        lags, ccf = _normalized_xcorr(s0, s1, MAX_LAG_STEPS)
        if lags is None:
            continue
        ax.plot(
            _lag_minutes(lags), ccf, lw=1.1, color=c,
            label=_pair_label(st, i0, i1),
        )
    ax.axhline(0.0, color="magenta", lw=1.0)
    ax.set_title(
        f"{st['title']}: вертикальная связь между датчиками\n"
        f"(одна термокоса, три глубины)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

for k in range(len(station_data), 4):
    axs_b[k].set_visible(False)

for ax in axs_b:
    ax.set_xlim(-MAX_LAG_MIN, MAX_LAG_MIN)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Сдвиг по времени τ, мин")
    ax.set_ylabel("R(τ) — коэфф. корреляции")

fig_b.suptitle(
    "Кросс-корреляция между датчиками на разных глубинах одной термокосы\n"
    "Каждая панель — одна коса (T1…T4). Кривые — R(τ) для пары датчиков; "
    "τ = 0 — колебания T на этих глубинах совпадают по фазе",
    y=1.02,
    fontsize=10,
)
fig_b.tight_layout(rect=(0, 0, 1, 0.96))
out_b = os.path.join(BASE_DIR, "st4_fig14_crosscorr_within_thermistor.png")
fig_b.savefig(out_b, dpi=170, bbox_inches="tight")
plt.close(fig_b)
print(f"Внутренние CCF: {out_b}")

# ---------------------------------------------------------
# C) Кросс-корреляции между термокосами (полная запись)
# ---------------------------------------------------------
common_index = station_data[0]["time"]
for st in station_data[1:]:
    common_index = common_index.intersection(st["time"])

if len(common_index) < 16:
    print("Мало общих отсчётов: межстанционные CCF (полная запись) не построены.")
else:
    aligned = {}
    for st in station_data:
        s = pd.Series(st["mean_series"], index=st["time"]).reindex(common_index)
        aligned[st["title"]] = _prepare_series(s.to_numpy(dtype=float))

    fig_c, ax_c = plt.subplots(figsize=(12, 6))
    pair_idx = 0
    titles = [st["title"] for st in station_data]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            t1, t2 = titles[i], titles[j]
            s1, s2 = aligned[t1], aligned[t2]
            if s1 is None or s2 is None:
                continue
            lags, ccf = _normalized_xcorr(s1, s2, MAX_LAG_STEPS)
            if lags is None:
                continue
            ax_c.plot(
                _lag_minutes(lags), ccf, lw=1.15,
                label=f"{t1} vs {t2}",
                color=plt.cm.tab10(pair_idx % 10),
            )
            lag_min, r_peak, r_abs = _peak_lag_info(lags, ccf)
            print(
                f"  {t1} vs {t2}: max|R| = {r_abs:.3f}, R = {r_peak:+.3f} "
                f"при τ = {lag_min:+.1f} мин "
                f"({t1} {'опережает' if lag_min > 0 else 'отстаёт' if lag_min < 0 else 'в фазе'} {t2})"
            )
            pair_idx += 1

    ax_c.axhline(0.0, color="magenta", lw=1.0)
    ax_c.axvline(0.0, color="0.5", lw=0.8, ls=":")
    ax_c.set_xlim(-MAX_LAG_MIN, MAX_LAG_MIN)
    ax_c.set_ylim(-1.0, 1.0)
    ax_c.set_xlabel("Временной лаг τ, мин")
    ax_c.set_ylabel("R(τ)")
    ax_c.set_title(
        "Кросс-корреляция между термокосами\n"
        "(средняя T по всем датчикам, полная одновременная запись)"
    )
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    fig_c.tight_layout()
    out_c = os.path.join(BASE_DIR, "st4_fig15_crosscorr_between_thermistors.png")
    fig_c.savefig(out_c, dpi=170)
    plt.close(fig_c)
    print(f"Межстанционные CCF (полная запись): {out_c}")

# ---------------------------------------------------------
# D) ЦУГ — синхронные ряды, CCF и столбики лагов
# ---------------------------------------------------------
print(
    f"\n--- Цуг на всех термокосах: "
    f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%d.%m.%Y %H:%M} ---"
)

surge_by_station = {}
for st in station_data:
    mask = (st["time"] >= SHORT_SURGE_START) & (st["time"] <= SHORT_SURGE_END)
    t_win = st["time"][mask]
    if len(t_win) < 8:
        print(f"  {st['title']}: мало точек в окне цуга ({len(t_win)}).")
        continue
    raw = st["t_at_zref"][mask]
    if np.count_nonzero(np.isfinite(raw)) < 8:
        print(
            f"  {st['title']}: мало данных T(z={SURGE_Z_REF_M:.1f} м, термоклин) в окне цуга."
        )
        continue
    temp_c = _fill_series(raw)
    surge_by_station[st["title"]] = {
        "time": t_win,
        "raw": raw,
        "temp_c": temp_c,
        "sensor_note": _closest_sensor_note(st["med_depths"], SURGE_Z_REF_M),
    }

if len(surge_by_station) < 2:
    print("Недостаточно термокос с данными в окне цуга для сравнения.")
else:
    # D1) Температура T(z, t) в зоне термоклина — все термокосы на одном окне
    fig_d1, axs_d1 = plt.subplots(len(station_data), 1, figsize=(12, 2.4 * len(station_data)), sharex=True)
    if len(station_data) == 1:
        axs_d1 = [axs_d1]

    for ax, st in zip(axs_d1, station_data):
        pack = surge_by_station.get(st["title"])
        if pack is None:
            ax.text(0.5, 0.5, "нет данных", transform=ax.transAxes, ha="center")
            ax.set_ylabel("T, °C")
            ax.set_title(st["title"])
            continue
        t_plot = pack["temp_c"]
        ax.plot(pack["time"], t_plot, color="teal", lw=1.0)
        t_mean = float(np.nanmean(t_plot))
        ax.axhline(t_mean, color="0.55", lw=0.8, ls="--", label=f"T̄ = {t_mean:.2f} °C")
        ax.set_ylabel("T, °C")
        ax.set_title(
            f"{st['title']}: T(z = {SURGE_Z_REF_M:.1f} м, термоклин), {pack['sensor_note']}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    _fmt_surge_axis(axs_d1[-1], SHORT_SURGE_START, SHORT_SURGE_END)
    axs_d1[-1].set_xlabel("Время (UTC)")
    fig_d1.suptitle(
        f"Температура в зоне термоклина (z = {SURGE_Z_REF_M:.1f} м) в окне цуга\n"
        f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%d.%m.%Y %H:%M}",
        fontsize=11,
        y=1.01,
    )
    fig_d1.tight_layout(rect=(0, 0, 1, 0.97))
    out_d1 = os.path.join(BASE_DIR, "st4_fig16_surge_T_thermocline_all_stations.png")
    fig_d1.savefig(out_d1, dpi=170, bbox_inches="tight")
    plt.close(fig_d1)
    print(f"Температура в окне цуга (термоклин): {out_d1}")

    # Общий индекс времени в окне (пересечение всех станций)
    surge_common = None
    for title in surge_by_station:
        t_idx = surge_by_station[title]["time"]
        surge_common = t_idx if surge_common is None else surge_common.intersection(t_idx)

    ref_pack = surge_by_station.get(REF_STATION)
    if ref_pack is None or surge_common is None or len(surge_common) < 8:
        print(f"  Нет общего окна или опорной {REF_STATION} для CCF цуга.")
    else:
        ref_proc = _prepare_series(
            pd.Series(ref_pack["raw"], index=ref_pack["time"])
            .reindex(surge_common)
            .to_numpy(),
        )
        max_lag_surge = min(MAX_LAG_STEPS, max(4, len(surge_common) // 2))

        # D2) CCF в окне цуга: T4 vs остальные (на z = SURGE_Z_REF_M)
        fig_d2, ax_d2 = plt.subplots(figsize=(11, 5.5))
        lag_records = []
        ccf_colors = ("#e74c3c", "#3498db", "#2ecc71")

        for cidx, st in enumerate(station_data):
            title = st["title"]
            if title == REF_STATION or title not in surge_by_station:
                continue
            other = _prepare_series(
                pd.Series(
                    surge_by_station[title]["raw"],
                    index=surge_by_station[title]["time"],
                ).reindex(surge_common).to_numpy(),
            )
            if ref_proc is None or other is None:
                continue
            lags, ccf = _normalized_xcorr(ref_proc, other, max_lag_surge)
            if lags is None:
                continue
            lag_min, r_peak, r_abs = _peak_lag_info(lags, ccf)
            ax_d2.plot(
                _lag_minutes(lags), ccf, lw=1.3,
                color=ccf_colors[len(lag_records) % len(ccf_colors)],
                label=f"{REF_STATION} vs {title}",
            )
            lag_records.append({
                "pair": f"{REF_STATION} vs {title}",
                "other": title,
                "tau_min": lag_min,
                "R_peak": r_peak,
                "abs_R": r_abs,
            })
            ahead = REF_STATION if lag_min > 0 else title if lag_min < 0 else "в фазе"
            print(
                f"  Цуг CCF {REF_STATION} vs {title}: max|R| = {r_abs:.3f}, "
                f"R = {r_peak:+.3f}, τ = {lag_min:+.1f} мин → {ahead}"
            )

        ax_d2.axhline(0.0, color="magenta", lw=1.0)
        ax_d2.axvline(0.0, color="0.5", lw=0.8, ls=":")
        tau_max = _lag_minutes(np.array([max_lag_surge]))[0]
        ax_d2.set_xlim(-tau_max, tau_max)
        ax_d2.set_ylim(-1.0, 1.0)
        ax_d2.set_xlabel("Временной лаг τ, мин")
        ax_d2.set_ylabel("R(τ)")
        ax_d2.set_title(
            f"Кросс-корреляция в окне цуга, термоклин z = {SURGE_Z_REF_M:.1f} м "
            f"(опорная {REF_STATION})\n"
            f"{SHORT_SURGE_START:%d.%m %H:%M} — {SHORT_SURGE_END:%d.%m %H:%M}"
        )
        ax_d2.grid(True, alpha=0.3)
        ax_d2.legend(fontsize=9, loc="upper right")
        fig_d2.tight_layout()
        out_d2 = os.path.join(BASE_DIR, "st4_fig17_surge_crosscorr_vs_T4.png")
        fig_d2.savefig(out_d2, dpi=170)
        plt.close(fig_d2)
        print(f"CCF цуга vs {REF_STATION}: {out_d2}")

        # D3) Фазовый сдвиг τ и R в максимуме |R| (не карта распространения в пространстве)
        if lag_records:
            fig_d3, (ax_lag, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
            labels = [r["other"] for r in lag_records]
            taus = [r["tau_min"] for r in lag_records]
            rvals = [r["R_peak"] for r in lag_records]
            bar_colors = ["#c0392b" if t > 0 else "#2980b9" if t < 0 else "#7f8c8d" for t in taus]

            ax_lag.bar(labels, taus, color=bar_colors, edgecolor="black", lw=0.6)
            ax_lag.axhline(0.0, color="black", lw=0.8)
            ax_lag.set_ylabel(
                f"τ при max|R|, мин\n(+ → {REF_STATION} опережает по времени)"
            )
            ax_lag.set_xlabel("Термокоса")
            ax_lag.set_title("Фазовый сдвиг относительно T4")
            ax_lag.grid(True, axis="y", alpha=0.3)

            ax_r.bar(labels, rvals, color="#16a085", edgecolor="black", lw=0.6)
            ax_r.axhline(0.0, color="magenta", lw=0.8)
            ax_r.set_ylim(-1.0, 1.0)
            ax_r.set_ylabel("R(τ) в максимуме")
            ax_r.set_xlabel("Термокоса")
            ax_r.set_title("Когерентность в пике корреляции")
            ax_r.grid(True, axis="y", alpha=0.3)

            fig_d3.suptitle(
                f"Фазовый сдвиг цуга в термоклине (z = {SURGE_Z_REF_M:.1f} м) "
                f"относительно {REF_STATION}\n"
                f"(окно {SHORT_SURGE_START:%d.%m %H:%M}–{SHORT_SURGE_END:%H:%M})",
                fontsize=10,
            )
            fig_d3.tight_layout(rect=(0, 0, 1, 0.93))
            out_d3 = os.path.join(BASE_DIR, "st4_fig18_surge_phase_lag_vs_T4.png")
            fig_d3.savefig(out_d3, dpi=170, bbox_inches="tight")
            plt.close(fig_d3)
            print(f"Фазовый сдвиг цуга (fig18): {out_d3}")
