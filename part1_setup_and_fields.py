# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import detrend, find_peaks
import gsw
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Палитра для полей температуры (контрастная, хорошо читается на графиках).
TEMP_FIELD_CMAP = "turbo"
TEMP_FIELD_NLEVELS = 40

# Мин. высота волны H (м).
WAVE_MIN_HEIGHT_M = 0.5

# Для этих изотерм не строим только отдельный рисунок plot_single (не fig08).
STANDALONE_SPECTRUM_SKIP = {23.0}


def skip_standalone_isotherm_spectrum(T_iso):
    """True — отдельный спектр (st4_fig08_iso*.png) для этой изотермы не нужен."""
    return any(np.isclose(float(T_iso), float(t)) for t in STANDALONE_SPECTRUM_SKIP)

# =========================================================
# ЧТЕНИЕ ДАННЫХ
# =========================================================
xlsx_path = os.path.join(BASE_DIR, "st4.xlsx")
sheet_dep, sheet_time, sheet_temp = "dep_n", "ss", "TV"
depths = pd.read_excel(xlsx_path, sheet_name=sheet_dep, header=None).values
temps = pd.read_excel(xlsx_path, sheet_name=sheet_temp, header=None).values
time = pd.to_datetime(pd.read_excel(xlsx_path, sheet_name=sheet_time, header=None).iloc[:, 0])
dfT, dfD = pd.DataFrame(temps, index=time), pd.DataFrame(depths, index=time)

# =========================================================
# 0. СХЕМА ТЕРМОКОСЫ
# =========================================================
median_depths_raw = np.nanmedian(depths, axis=0)


def plot_scheme(depths_arr, outpath=None, show=True):
    md = np.nanmedian(depths_arr, axis=0)
    md = md[np.isfinite(md)]
    z_bot = float(np.max(md))  # дно — по самому глубокому (нижнему) датчику
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.plot([0, 0], [0.0, z_bot], color="black", lw=2)
    ax.hlines(0, -0.5, 0.5, color="navy", lw=2)
    ax.text(0.6, 0, "Уровень моря", va="center", ha="left", color="navy")
    ax.hlines(z_bot, -0.5, 0.5, color="saddlebrown", lw=3)
    ax.text(0.6, z_bot, "Дно", va="center", ha="left", color="saddlebrown")
    ax.scatter(np.zeros_like(md), md, s=100, c="red", zorder=5)
    for i, d in enumerate(md):
        ax.text(0.1, d, f"{i+1}\n{d:.1f} м", va="center", ha="left", fontsize=9)
    ax.set_ylim(z_bot + 0.15, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("Глубина, м")
    ax.set_title("Схема термокосы №4 с датчиками")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
    if show:
        plt.savefig(os.path.join(BASE_DIR, "st4_fig01.png"), dpi=150)
        plt.close("all")
    return fig, ax


_THERMISTOR_STATIONS = (
    ("st1.xlsx", "dep1", "Термокоса №1"),
    ("st2.xlsx", "dep1", "Термокоса №2"),
    ("st3.xlsx", "dep1", "Термокоса №3"),
    ("st4.xlsx", "dep_n", "Термокоса №4"),
)


def _load_station_median_depths(xlsx_name, sheet_dep):
    path = os.path.join(BASE_DIR, xlsx_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    arr = pd.read_excel(path, sheet_name=sheet_dep, header=None).values.astype(float)
    return np.nanmedian(arr, axis=0)


def plot_all_thermistor_schemes(outpath=None):
    """Схемы термокос №1–4 на одном рисунке (глубины из st1–st4.xlsx)."""
    chains = []
    for xlsx, sheet, title in _THERMISTOR_STATIONS:
        md = _load_station_median_depths(xlsx, sheet)
        md = md[np.isfinite(md)]
        if md.size == 0:
            raise ValueError(f"Нет глубин датчиков в {xlsx}")
        chains.append({"title": title, "depths": md, "file": xlsx})

    z_max_all = max(float(np.max(c["depths"])) for c in chains)
    x_centers = np.arange(len(chains), dtype=float) * 3.0
    cable_half = 0.35

    fig, ax = plt.subplots(figsize=(14, 9))
    for x0, chain in zip(x_centers, chains):
        md = chain["depths"]
        z_bot = float(np.max(md))  # дно — глубина самого нижнего датчика
        ax.plot([x0, x0], [0.0, z_bot], color="black", lw=2, zorder=1)
        ax.hlines(0.0, x0 - cable_half, x0 + cable_half, color="navy", lw=2)
        ax.hlines(z_bot, x0 - cable_half, x0 + cable_half, color="saddlebrown", lw=3)
        ax.text(x0 + 0.55, z_bot, "Дно", va="center", ha="left", color="saddlebrown", fontsize=8)
        ax.scatter(np.full_like(md, x0), md, s=90, c="red", zorder=5, edgecolors="white", lw=0.4)
        for i, d in enumerate(md):
            ax.text(x0 + 0.42, d, f"{i + 1}\n{d:.1f} м", va="center", ha="left", fontsize=8)
        ax.text(x0, -0.35, chain["title"], ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.hlines(0.0, x_centers[0] - 1.2, x_centers[-1] + 1.2, color="navy", lw=1.5, alpha=0.35)
    ax.text(
        (x_centers[0] + x_centers[-1]) / 2,
        -0.08,
        "Уровень моря",
        ha="center",
        va="top",
        color="navy",
        fontsize=10,
    )
    ax.set_xlim(x_centers[0] - 1.5, x_centers[-1] + 2.2)
    ax.set_ylim(z_max_all + 0.8, -0.9)
    ax.set_xticks([])
    ax.set_ylabel("Глубина, м")
    ax.set_title("Схемы термокос станций 1–4 (медианные глубины датчиков)")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()

    if outpath is None:
        outpath = os.path.join(BASE_DIR, "thermistor_schemes_1_4.png")
    fig.savefig(outpath, dpi=200)
    fig.savefig(os.path.join(BASE_DIR, "st4_fig00_thermistor_schemes_1_4.png"), dpi=150)
    plt.close(fig)
    print(f"Схемы термокос 1–4: {outpath}, st4_fig00_thermistor_schemes_1_4.png")
    return fig


plot_scheme(depths)
plot_all_thermistor_schemes()

# =========================================================
# 1. УСРЕДНЕНИЕ ДО 30 СЕКУНД
# =========================================================
grid_30s = dfT.resample("30s").mean()
temps_30s = grid_30s.values
depths_30s = dfD.resample("30s").mean().values
time_30s = grid_30s.index
median_depths = np.nanmedian(depths_30s, axis=0)

print(f"Усреднение: {len(time)} → {len(time_30s)} точек (шаг 30 с)")

# =========================================================
# 2. ПОЛЕ ВРЕМЕННОЙ ИЗМЕНЧИВОСТИ ТЕМПЕРАТУРЫ
# =========================================================
TT, DD = np.meshgrid(time_30s, median_depths)

_t_vmin = float(np.nanpercentile(temps_30s, 2))
_t_vmax = float(np.nanpercentile(temps_30s, 98))
_t_lev = np.linspace(_t_vmin, _t_vmax, TEMP_FIELD_NLEVELS)
plt.figure(figsize=(12, 6))
plt.contourf(TT, DD, temps_30s.T, _t_lev, cmap=TEMP_FIELD_CMAP, extend="both")
plt.gca().invert_yaxis()
plt.colorbar(label="Температура, °C")
plt.ylabel("Глубина, м")
plt.xlabel("Дата")
plt.title("Временная изменчивость температуры")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig02.png"), dpi=150)
plt.close("all")

# =========================================================
# 3. ПОЛЕ ПЛОТНОСТИ (TEOS-10)
# =========================================================
lat, lon, g = 44.5, 37.98, 9.81
p_dbar = gsw.p_from_z(-median_depths, lat)
SP = 18 * np.ones_like(temps_30s)
rho = np.zeros_like(temps_30s)
for i in range(len(median_depths)):
    SA = gsw.SA_from_SP(SP[:, i], p_dbar[i], lon, lat)
    CT = gsw.CT_from_t(SA, temps_30s[:, i], p_dbar[i])
    rho[:, i] = gsw.rho(SA, CT, p_dbar[i])

plt.figure(figsize=(12, 6))
plt.contourf(TT, DD, rho.T, 20, cmap="plasma")
plt.gca().invert_yaxis()
plt.colorbar(label="Плотность, кг/м³")
plt.ylabel("Глубина, м")
plt.xlabel("Дата")
plt.title("Временная изменчивость плотности (TEOS-10)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig03.png"), dpi=150)
plt.close("all")

# =========================================================
# 4. ПРОФИЛЬ ЧАСТОТЫ ВЯЙСЯЛЯ–БРЕНТА  N(z) = √( g/ρ₀(z) · dρ/dz )
# =========================================================
rho0 = np.nanmean(rho, axis=0)
drho_dz = np.gradient(rho0, median_depths)
N2 = (g / rho0) * drho_dz
N_profile = np.sqrt(np.clip(N2, 0, None))           # рад/с
N_profile_cph = N_profile * 3600.0 / (2.0 * np.pi)  # цикл/час

N_max_cph  = float(np.nanmax(N_profile_cph))
N_max_rads = float(np.nanmax(N_profile))
z_max = median_depths[np.nanargmax(N_profile_cph)]
T_min_N = 60.0 / N_max_cph  # период в минутах

print(f"\nN_max = {N_max_rads:.4e} рад/с  =  {N_max_cph:.1f} цикл/час"
      f"  (T = {T_min_N:.1f} мин,  z = {z_max:.1f} м)")

plt.figure(figsize=(5, 7))
plt.plot(N_profile_cph, median_depths, color="darkcyan", lw=1.5, label="N(z)")
plt.scatter(N_max_cph, z_max, s=40, color="red", zorder=5,
            label=f"$N_{{max}}$ = {N_max_cph:.1f} цикл/час\n"
                  f"(T = {T_min_N:.1f} мин,  z = {z_max:.1f} м)")
plt.gca().invert_yaxis()
plt.xlabel("N(z), цикл/час")
plt.ylabel("Глубина, м")
plt.title("Профиль частоты Вяйсяля–Брента")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig04.png"), dpi=150)
plt.close("all")

# =========================================================
# 5–6. ВЫБОР ТРЁХ ИЗОТЕРМ
# =========================================================
T_min = np.nanmin(temps_30s)
T_max = np.nanmax(temps_30s)
T_range = T_max - T_min
T_third = T_range / 3.0

print(f"\nДиапазон температур: {T_min:.1f} – {T_max:.1f} °C")
print(f"  {T_min:.1f} – {T_min + T_third:.1f} °C")
print(f"  {T_min + T_third:.1f} – {T_min + 2*T_third:.1f} °C")
print(f"  {T_min + 2*T_third:.1f} – {T_max:.1f} °C")

iso_input = []
for i in range(3):
    iso_input.append(float(input(f"\nВведите изотерму {i+1} (°C): ")))

iso_sorted = sorted(iso_input, reverse=True)
iso_values = iso_sorted

print("\nИзотермы (по убыванию температуры):")
for tv in iso_values:
    print(f"  {tv}°C")

iso_depths = {}
for T_iso in iso_values:
    z_iso = np.full(len(time_30s), np.nan)
    for t in range(len(time_30s)):
        z_iso[t] = interp1d(temps_30s[t, :], median_depths,
                            bounds_error=False, fill_value=np.nan)(T_iso)
    iso_depths[T_iso] = z_iso


def plot_period_heatmap(
    time_arr,
    depth_levels,
    temp_field,
    start_time,
    end_time,
    out_file=None,
    n_levels=TEMP_FIELD_NLEVELS,
    cmap=TEMP_FIELD_CMAP,
    fig_size=(12, 5.5),
    dpi=150,
    ax=None,
    vmin=None,
    vmax=None,
    title=None,
    show_ylabel=True,
):
    """Тепловая карта T(z,t) на интервале без линий изотерм."""
    t0 = pd.to_datetime(start_time)
    t1 = pd.to_datetime(end_time)
    mask = (time_arr >= t0) & (time_arr <= t1)
    if not np.any(mask):
        raise ValueError(f"В интервале {t0} — {t1} нет данных.")

    t_sel = time_arr[mask]
    temps_sel = temp_field[mask, :]
    TT_sel, DD_sel = np.meshgrid(t_sel, depth_levels)

    if vmin is None:
        vmin = float(np.nanmin(temps_sel))
    if vmax is None:
        vmax = float(np.nanmax(temps_sel))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    levels = np.linspace(vmin, vmax, n_levels)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    cf = ax.contourf(
        TT_sel, DD_sel, temps_sel.T, levels=levels, cmap=cmap, extend="both",
    )
    ax.invert_yaxis()
    if show_ylabel:
        ax.set_ylabel("Глубина, м")
    ax.set_xlabel("Время")
    if title is None:
        title = (
            f"Температурное поле\n"
            f"{t0.strftime('%d.%m.%Y %H:%M')} — {t1.strftime('%d.%m.%Y %H:%M')}"
        )
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.grid(True, alpha=0.25)

    if standalone:
        fig.colorbar(cf, ax=ax, label="Температура, °C")
        fig.tight_layout()
        if out_file is None:
            out_file = "st4_fig12.png"
        out_path = out_file if os.path.isabs(out_file) else os.path.join(BASE_DIR, out_file)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"Тепловая карта (без изотерм): {out_path}")
        return out_path
    return cf


def plot_period_heatmap_with_isotherms(
    time_arr,
    depth_levels,
    temp_field,
    iso_depths_dict,
    iso_vals,
    start_time,
    end_time,
    fig_size=(9, 6),
    line_width=0.7,
    legend_fontsize=7,
    temp_min=None,
    temp_max=None,
    ax=None,
    show=True,
):
    """Тепловая карта температуры с изотермами только на выбранном интервале."""
    t0 = pd.to_datetime(start_time)
    t1 = pd.to_datetime(end_time)
    mask = (time_arr >= t0) & (time_arr <= t1)
    if not np.any(mask):
        raise ValueError(f"В интервале {start_time} — {end_time} нет данных.")

    t_sel = time_arr[mask]
    temps_sel = temp_field[mask, :]
    TT_sel, DD_sel = np.meshgrid(t_sel, depth_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    if temp_min is not None and temp_max is not None and temp_max > temp_min:
        levels = np.linspace(temp_min, temp_max, TEMP_FIELD_NLEVELS)
        cf = ax.contourf(
            TT_sel, DD_sel, temps_sel.T, levels=levels, cmap=TEMP_FIELD_CMAP, extend="both",
        )
    else:
        t_lo = float(np.nanpercentile(temps_sel, 2))
        t_hi = float(np.nanpercentile(temps_sel, 98))
        levels = np.linspace(t_lo, t_hi, TEMP_FIELD_NLEVELS)
        cf = ax.contourf(
            TT_sel, DD_sel, temps_sel.T, levels=levels, cmap=TEMP_FIELD_CMAP, extend="both",
        )
    ax.invert_yaxis()
    fig.colorbar(cf, ax=ax, label="Температура, °C")

    cmap = plt.cm.get_cmap("Greys")
    for idx, T_iso in enumerate(iso_vals):
        if T_iso in iso_depths_dict:
            z_all = iso_depths_dict[T_iso]
        else:
            nearest_key = min(iso_depths_dict.keys(), key=lambda k: abs(float(k) - float(T_iso)))
            z_all = iso_depths_dict[nearest_key]
        z_sel = z_all[mask]
        ax.plot(
            t_sel,
            z_sel,
            color=cmap(0.25 + 0.7 * (idx / max(len(iso_vals) - 1, 1))),
            lw=line_width,
            label=f"{T_iso}°C",
        )

    ax.set_ylabel("Глубина, м")
    ax.set_xlabel("Время")
    ax.set_title(
        f"Температурное поле с изотермами ({t0.strftime('%d.%m.%Y %H:%M')} — "
        f"{t1.strftime('%d.%m.%Y %H:%M')})"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=legend_fontsize, ncol=2)
    if show:
        fig.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig05.png"), dpi=150)
        plt.close("all")

# =========================================================
# 7. ПОЛЕ ТЕМПЕРАТУРЫ С ВЫДЕЛЕННЫМИ ИЗОТЕРМАМИ
# =========================================================
fig7, ax7 = plt.subplots(figsize=(12, 6))
cf = ax7.contourf(TT, DD, temps_30s.T, _t_lev, cmap=TEMP_FIELD_CMAP, extend="both")
ax7.invert_yaxis()
plt.colorbar(cf, ax=ax7, label="Температура, °C")
label_positions = [0.25, 0.50, 0.75]
iso_field_color = "#FFFFFF"
for idx, T_iso in enumerate(iso_values):
    z_iso = iso_depths[T_iso]
    ic = iso_field_color
    ax7.plot(time_30s, z_iso, color=ic, lw=0.8, label=f"{T_iso}°C")
    valid_idx = np.where(~np.isnan(z_iso))[0]
    if len(valid_idx) > 0:
        pos = int(len(valid_idx) * label_positions[idx])
        pos = min(pos, len(valid_idx) - 1)
        ti = valid_idx[pos]
        ax7.annotate(f" {T_iso}°C", xy=(time_30s[ti], z_iso[ti]),
                     color=ic, fontsize=9, fontweight="bold", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, ec="none"))
ax7.set_ylabel("Глубина, м")
ax7.set_xlabel("Дата")
ax7.set_title("Временная изменчивость температуры с изотермами")
ax7.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
# Подписи уже нанесены прямо на линии, отдельная легенда не нужна.
fig7.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig06.png"), dpi=150)
plt.close("all")

# =========================================================
# 8. ОБЩИЙ НЕПРЕРЫВНЫЙ УЧАСТОК + АННОТАЦИЯ
# =========================================================


def common_continuous_interval(*arrays):
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
    best = np.argmax(lengths)
    return starts[best], ends[best]


def all_continuous_segments(arr, min_len=1):
    """Все непрерывные участки без NaN: [(start, end), ...], end — как срез Python."""
    isnan = np.isnan(np.asarray(arr, dtype=float))
    segments, start = [], None
    for i, val in enumerate(isnan):
        if not val and start is None:
            start = i
        elif val and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None
    if start is not None and len(isnan) - start >= min_len:
        segments.append((start, len(isnan)))
    return segments


def longest_continuous_segment(arr):
    segments = all_continuous_segments(arr, min_len=1)
    if not segments:
        return None, None
    return segments[np.argmax([e - s for s, e in segments])]


iso_arrays = [iso_depths[T] for T in iso_values]
c_start, c_end = common_continuous_interval(*iso_arrays)

if c_start is None:
    raise RuntimeError("Нет общего непрерывного участка для трёх изотерм!")

dt = 30
seg_len = c_end - c_start
seg_hours = (seg_len - 1) * dt / 3600.0

print(f"\nОбщий непрерывный участок: {seg_len} точек, {seg_hours:.1f} часов")

fig8, ax8 = plt.subplots(figsize=(14, 5))
for idx, T_iso in enumerate(iso_values):
    z_iso = iso_depths[T_iso]
    z_mean_seg = np.nanmean(z_iso[c_start:c_end])
    ax8.plot(time_30s, z_iso, lw=1, color=f"C{idx}",
             label=f"{T_iso}°C, z̄ = {z_mean_seg:.1f} м")
ax8.axvspan(time_30s[c_start], time_30s[c_end - 1],
            color="orange", alpha=0.2, label="Общий участок")
ax8.invert_yaxis()
ax8.set_ylabel("Глубина, м")
ax8.set_xlabel("Время")
ax8.set_title("Колебания изотерм с выделенным общим участком")
ax8.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9, loc="best")

annotation_text = f"Продолжительность: {seg_hours:.1f} ч"
ax8.text(0.02, 0.02, annotation_text, transform=ax8.transAxes,
         fontsize=10, va="bottom", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.8))
fig8.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig07.png"), dpi=150)
plt.close("all")

# =========================================================
# ПАРАМЕТРЫ СПЕКТРАЛЬНОГО АНАЛИЗА
# =========================================================
Fn = (1 / dt) * 3600 / 2
Omega = 7.2921e-5
fin = (2 * Omega * np.sin(np.deg2rad(lat)) / (2 * np.pi)) * 3600
C_M = 204.0


def garrett_munk_psd(f_cph, N_cph, fin_hz, C_M_val=None):
    """Модель Гарретта–Манка: S(f) = C_M·f_in·√(f²−f_in²) / (N·f³), f_in < f < N (ч⁻¹)."""
    if C_M_val is None:
        C_M_val = C_M
    f_cph = np.asarray(f_cph, dtype=float)
    S_GM = np.zeros_like(f_cph)
    N_cph = float(N_cph)
    fin_hz = float(fin_hz)
    if not (np.isfinite(N_cph) and N_cph > fin_hz > 0):
        return S_GM
    mg = (f_cph > fin_hz) & (f_cph < N_cph)
    S_GM[mg] = (
        C_M_val * (fin_hz * np.sqrt(f_cph[mg] ** 2 - fin_hz ** 2))
        / (N_cph * f_cph[mg] ** 3)
    )
    return S_GM


def _ascii_loglog_bars(f, y, n_cols=76, n_rows=18):
    """Вертикальные столбцы в лог-бинах по f; по вертикали — log10(y)."""
    f = np.asarray(f, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (f > 0) & (y > 0) & np.isfinite(f) & np.isfinite(y)
    f, y = f[m], y[m]
    if f.size == 0:
        return ["(нет положительных значений для графика)"]
    lf = np.log10(f)
    ly = np.log10(y)
    f_lo, f_hi = lf.min(), lf.max()
    y_lo, y_hi = ly.min(), ly.max()
    span = y_hi - y_lo
    if span <= 0:
        span = 0.1
        y_lo, y_hi = y_hi - span, y_hi + span
    pad = 0.06 * span
    y_lo -= pad
    y_hi += pad
    edges = np.linspace(f_lo, f_hi, n_cols + 1)
    y_bin = np.full(n_cols, np.nan)
    for i in range(n_cols):
        lo, hi = edges[i], edges[i + 1]
        sel = (lf >= lo) & (lf < hi) if i < n_cols - 1 else (lf >= lo) & (lf <= hi)
        if np.any(sel):
            y_bin[i] = np.nanmax(ly[sel])
    grid = [[" " for _ in range(n_cols)] for _ in range(n_rows)]
    for j in range(n_cols):
        if not np.isfinite(y_bin[j]):
            continue
        r = int(round((y_hi - y_bin[j]) / (y_hi - y_lo) * (n_rows - 1)))
        r = int(np.clip(r, 0, n_rows - 1))
        for k in range(r, n_rows):
            grid[k][j] = "#"
    lines = []
    for i in range(n_rows):
        y_tick = y_hi - (y_hi - y_lo) * (i / max(n_rows - 1, 1))
        tick = f"{10 ** y_tick:.2e}"[:10].ljust(11) if i % 2 == 0 else " " * 11
        lines.append(tick + "".join(grid[i]))
    lines.append(
        " " * 11
        + f"f_min={10 ** f_lo:.4e}  f_max={10 ** f_hi:.4e} ч⁻¹  (# = уровень по max в лог-бине)"
    )
    return lines


def _log_bin_aggregate(freq, values, n_bins=56, aggr="mean"):
    """Логарифмические интервалы по частоте; в каждом — mean или max."""
    freq = np.asarray(freq, dtype=float)
    values = np.asarray(values, dtype=float)
    m = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    freq, values = freq[m], values[m]
    if freq.size == 0:
        return None
    lo, hi = np.log10(freq.min()), np.log10(freq.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.logspace(lo, hi, n_bins + 1)
    f_c, out = [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        sel = (freq >= a) & (freq < b) if i < n_bins - 1 else (freq >= a) & (freq <= b)
        if not np.any(sel):
            continue
        block = values[sel]
        if aggr == "max":
            out.append(np.nanmax(block))
        else:
            out.append(np.nanmean(block))
        f_c.append(np.sqrt(a * b))
    return np.array(f_c), np.array(out)


def _log_bin_psd_gm(freq, psd, sgm, n_bins=48):
    """Одинаковые лог-бины по f; в каждом — среднее PSD и среднее S_GM."""
    m = (
        (freq > 0)
        & np.isfinite(freq)
        & np.isfinite(psd)
        & np.isfinite(sgm)
        & (sgm > 0)
    )
    if not np.any(m):
        return None
    fq, p1, s1 = freq[m], psd[m], sgm[m]
    lo, hi = np.log10(fq.min()), np.log10(fq.max())
    if hi <= lo:
        hi = lo + 1e-9
    edges = np.logspace(lo, hi, n_bins + 1)
    fc, pv, gv = [], [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        sel = (fq >= a) & (fq < b) if i < n_bins - 1 else (fq >= a) & (fq <= b)
        if not np.any(sel):
            continue
        fc.append(np.sqrt(a * b))
        pv.append(np.nanmean(p1[sel]))
        gv.append(np.nanmean(s1[sel]))
    return np.array(fc), np.array(pv), np.array(gv)


def _write_spectra_detail_txt(
    spec_list, fin_hz, N_mean_cph, seg_len, seg_hours, dt_s, txt_path=None
):
    """Подробный текстовый вывод амплитудных спектров и PSD в UTF-8 txt."""
    if txt_path is None:
        txt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "st4_spectra_detail.txt"
        )

    sep = "=" * 88
    with open(txt_path, "w", encoding="utf-8") as fp:

        def w(line=""):
            fp.write(line + "\n")

        w()
        w(sep)
        w(
            "СПЕКТРЫ (общий участок): "
            f"{seg_len} отсчётов, {seg_hours:.2f} ч, шаг Δt = {dt_s} с"
        )
        w(f"Найквист (по часам): F_n = {0.5 * 3600 / dt_s:.6f} ч⁻¹")
        w(f"f_in = {fin_hz:.6f} ч⁻¹,  N_max (профиль) = {N_mean_cph:.6f} ч⁻¹  [справочно]")
        w(
            "На графиках амплитуд и PSD — пунктир N_max (профиль В–Б); "
            "серые пунктиры: T = 0.1…1.0 ч (шаг 0.1), целые 1…17 ч и T = 17.1 ч."
        )
        w(sep)

        for block in spec_list:
            T_iso = block["T_iso"]
            z_mean = block["z_mean"]
            N_used = block["N_used"]
            gm_slope_str = block["gm_slope_str"]
            N_pts = block["N_pts"]
            Fv = block["Fv"]
            amplitude = block["amplitude"]
            f_psd = block["f_psd"]
            Pxx = block["Pxx"]
            S_GM = block["S_GM"]

            w()
            w(f"--- Изотерма {T_iso} °C ---")
            w(
                f"    z_mean = {z_mean:.3f} м,  N_max = {N_used:.6f} ч⁻¹,  "
                f"наклон Г–М (log-log) = {gm_slope_str}"
            )
            w(f"    N_fft = {N_pts},  длина ряда после detrend: {N_pts} точек")

            mamp = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
            if np.any(mamp):
                ia = int(np.nanargmax(amplitude[mamp]))
                fa_peak = Fv[mamp][ia]
                a_peak = amplitude[mamp][ia]
                w(
                    f"    Амплитудный спектр: max 2|F| = {a_peak:.4e} м при f = {fa_peak:.6e} ч⁻¹"
                )
                w("    ASCII (вертикаль # = log10 амплитуды по лог-бинам f):")
                for line in _ascii_loglog_bars(
                    Fv[mamp], amplitude[mamp], n_cols=76, n_rows=16
                ):
                    w("    " + line)
                agg_amp = _log_bin_aggregate(
                    Fv[mamp], amplitude[mamp], n_bins=52, aggr="mean"
                )
                if agg_amp is not None:
                    fc, av = agg_amp
                    w("    Таблица (лог-бины по f, среднее 2|F| по бину):")
                    w("      " + "f, 1/h".ljust(14) + "  " + "2|F|, м".ljust(14))
                    for i in range(len(fc)):
                        w(f"      {fc[i]:.6e}    {av[i]:.6e}")
            else:
                w("    Амплитудный спектр: нет положительных отсчётов")

            mpsd = (f_psd > 0) & np.isfinite(Pxx) & (Pxx > 0)
            if np.any(mpsd):
                ip = int(np.nanargmax(Pxx[mpsd]))
                fp_peak = f_psd[mpsd][ip]
                p_peak = Pxx[mpsd][ip]
                w(
                    f"    PSD: max = {p_peak:.4e} м²·ч при f = {fp_peak:.6e} ч⁻¹"
                )
                w("    ASCII PSD (log-log столбцы):")
                for line in _ascii_loglog_bars(
                    f_psd[mpsd], Pxx[mpsd], n_cols=76, n_rows=16
                ):
                    w("    " + line)

                agg_psd = _log_bin_aggregate(
                    f_psd[mpsd], Pxx[mpsd], n_bins=56, aggr="mean"
                )
                if agg_psd is not None:
                    fc2, pv2 = agg_psd
                    w(
                        "    Таблица PSD (все f>0, лог-бины по частоте, среднее PSD в бине):"
                    )
                    w("      " + "f, ч⁻¹".ljust(16) + "  " + "PSD, м²·ч".ljust(16))
                    for i in range(len(fc2)):
                        w(f"      {fc2[i]:.6e}    {pv2[i]:.6e}")

                mgm = mpsd & (S_GM > 0) & np.isfinite(S_GM)
                if np.any(mgm):
                    rat = Pxx[mgm] / S_GM[mgm]
                    w(
                        f"    Сравнение с Г–М в диапазоне f_in<f<N_max: "
                        f"min(PSD/S_GM)={np.nanmin(rat):.3e}, max={np.nanmax(rat):.3e}, "
                        f"median={np.nanmedian(rat):.3e}"
                    )
                    tgm = _log_bin_psd_gm(f_psd, Pxx, S_GM, n_bins=48)
                    if tgm is not None:
                        fc_p, pv, gv = tgm
                        w(
                            "    Таблица PSD и S_GM (общие лог-бины по f, среднее в бине):"
                        )
                        w(
                            "      "
                            + "f, ч⁻¹".ljust(16)
                            + "  "
                            + "PSD, м²·ч".ljust(16)
                            + "  "
                            + "S_GM".ljust(16)
                            + "  "
                            + "PSD/S_GM"
                        )
                        for i in range(len(fc_p)):
                            ratio = pv[i] / gv[i] if gv[i] > 0 else np.nan
                            w(
                                f"      {fc_p[i]:.6e}    {pv[i]:.6e}    {gv[i]:.6e}    {ratio:.4e}"
                            )
                else:
                    w(
                        "    Модель Г–М: нет положительных S_GM на сетке PSD для этой изотермы"
                    )
            else:
                w("    PSD: нет положительных отсчётов")

        w()
        w(sep)
        w()

    print(f"Подробный вывод спектров записан в файл:\n  {os.path.abspath(txt_path)}")


def _format_period_hours_label(T_h):
    """Подпись периода T (ч) для вертикальной линии на спектре."""
    T_h = float(T_h)
    if not np.isfinite(T_h) or T_h <= 0:
        return "—"
    if T_h >= 10:
        return f"{T_h:.1f} ч"
    if T_h >= 1:
        return f"{T_h:.2f} ч"
    return f"{T_h:.3f} ч"


def _add_spectrum_period_lines_hours(ax, periods_h, labels=None, color="0.45", lw=1.1):
    """Пунктир f = 1/T (ч⁻¹); на линии — подпись периода в часах."""
    from matplotlib.transforms import blended_transform_factory

    pairs = []
    for i, T_h in enumerate(periods_h):
        T_h = float(T_h)
        if not np.isfinite(T_h) or T_h <= 0:
            continue
        lbl = labels[i] if labels is not None else _format_period_hours_label(T_h)
        pairs.append((T_h, lbl))
    if not pairs:
        return

    xlo, xhi = ax.get_xlim()
    visible = []
    for T_h, lbl in pairs:
        f_cph = 1.0 / T_h
        if f_cph <= xlo or f_cph >= xhi:
            continue
        visible.append((T_h, lbl))
    if not visible:
        return

    visible.sort(key=lambda x: x[0], reverse=True)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    n = len(visible)
    step = 0.92 / max(n, 1)
    fs = 7 if n > 14 else 8
    for row, (T_h, lbl) in enumerate(visible):
        f_cph = 1.0 / T_h
        ax.axvline(f_cph, color=color, ls="--", lw=lw, alpha=0.85, zorder=4)
        ax.text(
            f_cph,
            0.98 - step * row,
            lbl,
            transform=trans,
            rotation=90,
            va="top",
            ha="center",
            fontsize=fs,
            color=color,
            clip_on=True,
        )


def _add_amp_spectrum_period_reference_lines(ax, t_max_h=17.1, color="0.45", lw=1.0):
    """Серые пунктиры периода: T = 0.1…1.0 ч (шаг 0.1), целые 1…17 ч и T = t_max_h."""
    t_max_h = float(t_max_h)
    raw = list(np.round(np.arange(0.1, 1.0 + 1e-9, 0.1), 1))
    raw += list(range(1, int(np.floor(t_max_h)) + 1))
    if t_max_h > int(np.floor(t_max_h)) + 1e-6:
        raw.append(t_max_h)
    periods = sorted({round(float(t), 4) for t in raw})

    labels = []
    for t in periods:
        if t < 1 or abs(t - round(t)) > 1e-6:
            labels.append(f"{t:.1f} ч")
        else:
            labels.append(f"{int(round(t))} ч")

    _add_spectrum_period_lines_hours(ax, periods, labels, color=color, lw=lw)


def _write_top_repeated_isos_psd_txt(
    blocks,
    *,
    dt_s,
    best_iso,
    t_start_str,
    t_end_str,
    iso_repeats_sorted,
    top_k,
    txt_path=None,
):
    """PSD изотерм с наибольшей повторяемостью волн — подробный UTF-8 txt."""
    if txt_path is None:
        txt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "st4_top_repeated_isotherms_psd.txt",
        )

    with open(txt_path, "w", encoding="utf-8") as fp:

        def w(line=""):
            fp.write(line + "\n")

        w("PSD изотерм с наибольшей повторяемостью волн")
        w(
            f"(участок изотермы {best_iso:.1f} °C для PSD; "
            f"волны на изотерме анализа wave_iso, h>={WAVE_MIN_HEIGHT_M} м, T>=3 мин)"
        )
        w(f"Изотерма участка PSD: {best_iso:.1f} °C")
        w(f"Интервал: {t_start_str} — {t_end_str}")
        w(f"Δt = {dt_s} с,  F_n (Найквист) = {0.5 * 3600 / dt_s:.6f} ч⁻¹")
        w()
        w(f"Топ-{top_k} по числу совпадений с волнами на лучшей изотерме:")
        for T_iso, cnt in iso_repeats_sorted[:top_k]:
            w(f"  {T_iso:.1f} °C: {cnt} волн")
        w()

        if not blocks:
            w("(Нет ни одной изотермы с достаточной длиной ряда для расчёта PSD.)")
        for b in blocks:
            w("=" * 72)
            T_iso = b["T_iso"]
            w(f"Изотерма {T_iso:.1f} °C  |  совпадений с волнами: {b['repeat_count']}")
            w(f"N = {b['n_pts']} отсчётов (после detrend), PSD в м²·ч, f в ч⁻¹")
            w(
                f"Наклон log10(PSD) vs log10(f): {b['slope']:.4f}; "
                f"intercept = {b['intercept']:.4f}"
            )
            fv, pv = b["f_cph"], b["Pxx"]
            ip = int(np.nanargmax(pv))
            w(f"Максимум PSD: {pv[ip]:.6e} м²·ч при f = {fv[ip]:.6e} ч⁻¹")
            w()
            w("ASCII (лог-бины по f, по вертикали log10(PSD), # = max в бине):")
            for line in _ascii_loglog_bars(fv, pv, n_cols=76, n_rows=16):
                w("  " + line)
            w()
            agg = _log_bin_aggregate(fv, pv, n_bins=56, aggr="mean")
            if agg is not None:
                fc, vals = agg
                w("Таблица (лог-бины по f, среднее PSD в бине):")
                w("  " + "f, ч⁻¹".ljust(16) + "  " + "PSD, м²·ч".ljust(16))
                for i in range(len(fc)):
                    w(f"  {fc[i]:.6e}    {vals[i]:.6e}")
            w()

    print(
        "Текстовый файл PSD изотерм с наибольшей повторяемостью:\n"
        f"  {os.path.abspath(txt_path)}"
    )


