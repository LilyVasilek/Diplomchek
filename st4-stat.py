import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.signal import detrend, find_peaks
import gsw
import os

# =========================================================
# ЧТЕНИЕ ДАННЫХ
# =========================================================
xlsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4.xlsx")
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
    # Нижняя граница схемы — глубина самого нижнего датчика (визуально на дне).
    min_depth, max_depth = 0, float(np.nanmax(md))
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.plot([0, 0], [min_depth, max_depth], color="black", lw=2)
    ax.hlines(0, -0.5, 0.5, color="navy", lw=2)
    ax.text(0.6, 0, "Уровень моря", va="center", ha="left", color="navy")
    ax.hlines(max_depth, -0.5, 0.5, color="saddlebrown", lw=3)
    ax.text(0.6, max_depth, "Дно", va="center", ha="left", color="saddlebrown")
    ax.scatter(np.zeros_like(md), md, s=100, c="red", zorder=5)
    for i, d in enumerate(md):
        label = f"{i+1}\n{d:.1f} м"
        if i == 0:
            label += "\n(закреплен на дне)"
        ax.text(0.1, d, label, va="center", ha="left", fontsize=9)
    ax.set_ylim(max_depth + 0.15, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("Глубина, м")
    ax.set_title("Схема термокосы №4 с датчиками")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
    if show:
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig01.png"), dpi=150)
        plt.close("all")
    return fig, ax


plot_scheme(depths)

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

plt.figure(figsize=(12, 6))
plt.contourf(TT, DD, temps_30s.T, 20, cmap="viridis")
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
# 5–6. ВЫБОР ТРЁХ ИЗОТЕРМ (верхний, средний, нижний слои)
# =========================================================
T_min = np.nanmin(temps_30s)
T_max = np.nanmax(temps_30s)
T_range = T_max - T_min
T_third = T_range / 3.0

print(f"\nДиапазон температур: {T_min:.1f} – {T_max:.1f} °C")
print(f"  Нижний слой:  {T_min:.1f} – {T_min + T_third:.1f} °C")
print(f"  Средний слой: {T_min + T_third:.1f} – {T_min + 2*T_third:.1f} °C")
print(f"  Верхний слой: {T_min + 2*T_third:.1f} – {T_max:.1f} °C")

iso_input = []
for i in range(3):
    iso_input.append(float(input(f"\nВведите изотерму {i+1} (°C): ")))

iso_sorted = sorted(iso_input, reverse=True)
iso_values = iso_sorted
iso_labels = ["верхний слой", "средний слой", "нижний слой"]

print(f"\nИзотермы (отсортированы по слоям):")
for tv, lb in zip(iso_values, iso_labels):
    print(f"  {tv}°C  ({lb})")

iso_depths = {}
for T_iso in iso_values:
    z_iso = np.full(len(time_30s), np.nan)
    for t in range(len(time_30s)):
        z_iso[t] = interp1d(temps_30s[t, :], median_depths,
                            bounds_error=False, fill_value=np.nan)(T_iso)
    iso_depths[T_iso] = z_iso


def plot_period_heatmap_with_isotherms(
    time_arr,
    depth_levels,
    temp_field,
    iso_depths_dict,
    iso_vals,
    iso_labs,
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
        levels = np.linspace(temp_min, temp_max, 20)
        cf = ax.contourf(TT_sel, DD_sel, temps_sel.T, levels=levels, cmap="viridis", extend="both")
    else:
        cf = ax.contourf(TT_sel, DD_sel, temps_sel.T, 20, cmap="viridis")
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
            label=f"{T_iso}°C ({iso_labs[idx]})",
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
cf = ax7.contourf(TT, DD, temps_30s.T, 20, cmap="viridis")
ax7.invert_yaxis()
plt.colorbar(cf, ax=ax7, label="Температура, °C")
label_positions = [0.25, 0.50, 0.75]
iso_field_colors = ["#FFFFFF", "#D0D0D0", "#A0A0A0"]
for idx, T_iso in enumerate(iso_values):
    z_iso = iso_depths[T_iso]
    ic = iso_field_colors[idx]
    ax7.plot(time_30s, z_iso, color=ic, lw=0.8, label=f"{T_iso}°C ({iso_labels[idx]})")
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
ax7.legend(loc="lower right", fontsize=9)
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
             label=f"{T_iso}°C ({iso_labels[idx]}), z̄ = {z_mean_seg:.1f} м")
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
        w(f"f_in = {fin_hz:.6f} ч⁻¹,  <N>_профиль = {N_mean_cph:.6f} ч⁻¹")
        w(sep)

        for block in spec_list:
            T_iso = block["T_iso"]
            lbl = block["label"]
            z_mean = block["z_mean"]
            N_iso = block["N_iso"]
            gm_slope_str = block["gm_slope_str"]
            N_pts = block["N_pts"]
            Fv = block["Fv"]
            amplitude = block["amplitude"]
            f_psd = block["f_psd"]
            Pxx = block["Pxx"]
            S_GM = block["S_GM"]

            w()
            w(f"--- Изотерма {T_iso} °C ({lbl}) ---")
            w(
                f"    z_mean = {z_mean:.3f} м,  N(z_mean) = {N_iso:.6f} ч⁻¹,  "
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
                        f"    Сравнение с Г–М в диапазоне f_in<f<N(z): "
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
        w("(участок непрерывной лучшей изотермы; волны по критериям h>=0.5 м, T>=3 мин)")
        w(f"Лучшая изотерма: {best_iso:.1f} °C")
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


# =========================================================
# 9–11. СПЕКТРЫ НА ОБЩЕМ УЧАСТКЕ (ДВА ГРАФИКА)
#   1) Амплитудные спектры — три изотермы на одной сетке
#   2) PSD — три изотермы + модель Гарретта–Манка на одной сетке
# =========================================================
fig_sp, (ax_amp, ax_psd) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
amp_colors = ["royalblue", "seagreen", "coral"]
psd_colors = ["darkviolet", "sienna", "darkgoldenrod"]

print(f"\nМодель Гарретта–Манка: S(f,z) = C_M·f_in·√(f²−f_in²) / (N(z)·f³)")
print(f"  Модель определена в диапазоне f_in < f < N(z)")
print(f"  f_in (инерционная частота) = {fin:.4f} ч⁻¹")

spec_console_data = []

for idx, T_iso in enumerate(iso_values):
    z_iso = iso_depths[T_iso]
    z_segment = z_iso[c_start:c_end]
    z_mean = np.nanmean(z_segment)

    signal = detrend(z_segment, type='linear')
    N_pts = len(signal)

    # --- Амплитудный спектр: F(ν) = ∫ f(t)·exp(−i2πνt) dt ---
    Feta = np.fft.fft(signal) / N_pts
    Fv = np.linspace(0, Fn, N_pts // 2 + 1)
    amplitude = 2 * np.abs(Feta[:len(Fv)])

    # --- Спектральная плотность мощности: Ŵ(ω) = (1/(N·fₐ)) |Σ x(k)·exp(−jωkT)|² ---
    f_hz = np.fft.rfftfreq(N_pts, d=dt)
    f_psd = f_hz * 3600.0
    fa = 1.0 / dt
    X = np.fft.rfft(signal)
    Pxx = ((1.0 / (N_pts * fa)) * (np.abs(X) ** 2)) / 3600.0

    # --- Модель Гарретта–Манка: S(f,z) = C_M·f_in·√(f²−f_in²) / (N(z)·f³) ---
    N_iso = np.interp(z_mean, median_depths, N_profile_cph)  # цикл/час
    S_GM = np.zeros_like(f_psd)
    mg = (f_psd > fin) & (f_psd < N_iso)
    S_GM[mg] = C_M * (fin * np.sqrt(f_psd[mg] ** 2 - fin ** 2)) / (N_iso * f_psd[mg] ** 3)

    gm_slope = np.nan
    mgp_fit = mg & (S_GM > 0) & np.isfinite(S_GM)
    if np.count_nonzero(mgp_fit) >= 3:
        lf = np.log10(f_psd[mgp_fit])
        ls = np.log10(S_GM[mgp_fit])
        gm_slope, _ = np.polyfit(lf, ls, 1)

    gm_slope_str = f"{gm_slope:.2f}" if np.isfinite(gm_slope) else "—"
    print(
        f"  {T_iso}°C: z̄ = {z_mean:.1f} м, N(z̄) = {N_iso:.4f} ч⁻¹, "
        f"наклон Г–М (log–log) = {gm_slope_str}"
    )

    spec_console_data.append(
        {
            "T_iso": T_iso,
            "label": iso_labels[idx],
            "z_mean": z_mean,
            "N_iso": N_iso,
            "gm_slope_str": gm_slope_str,
            "N_pts": N_pts,
            "Fv": Fv.copy(),
            "amplitude": amplitude.copy(),
            "f_psd": f_psd.copy(),
            "Pxx": Pxx.copy(),
            "S_GM": S_GM.copy(),
        }
    )

    lbl = f"{T_iso}°C ({iso_labels[idx]})"

    # --- 1) Амплитудный спектр ---
    mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
    if np.any(mfft):
        ax_amp.loglog(Fv[mfft], amplitude[mfft], color=amp_colors[idx], lw=1, label=lbl)

    # --- 2) PSD ---
    mpsd = (f_psd > 0) & np.isfinite(Pxx) & (Pxx > 0)
    if np.any(mpsd):
        ax_psd.loglog(f_psd[mpsd], Pxx[mpsd], color=psd_colors[idx], lw=1, label=f"PSD {lbl}")

    # --- Модель Г–М (чёрная, разные стили линий для различения) ---
    gm_styles = ["-", "--", "-."]
    mgp = (f_psd > 0) & np.isfinite(S_GM) & (S_GM > 0)
    if np.any(mgp):
        ax_psd.loglog(f_psd[mgp], S_GM[mgp], color="black", ls=gm_styles[idx], lw=1.5,
                      alpha=0.8, label=f"Г–М {T_iso}°C (N={N_iso:.2f})")

# --- Пунктир частоты Вяйсяля–Брента ---
N_mean = float(np.nanmean(N_profile_cph))  # цикл/час

_write_spectra_detail_txt(spec_console_data, fin, N_mean, seg_len, seg_hours, dt)

for ax in (ax_amp, ax_psd):
    ax.axvline(N_mean, color="black", ls="--", lw=1.3)

ax_amp.plot([], [], color="black", ls="--", lw=1.3,
            label=f"Частота Вяйсяля–Брента ({N_mean:.2f} цикл/час)")

ax_amp.set_ylabel("Амплитуда, м")
ax_amp.set_title("Амплитудные спектры изотерм (общий участок)")
ax_amp.grid(True, which="both", alpha=0.3)
ax_amp.legend(fontsize=8, loc="best")

ax_psd.plot([], [], color="black", ls="--", lw=1.3,
            label=f"Частота Вяйсяля–Брента ({N_mean:.2f} цикл/час)")
ax_psd.set_xlabel("Частота, цикл/час")
ax_psd.set_ylabel("PSD, м²·час")
ax_psd.set_title("Спектральная плотность мощности + модель Гарретта–Манка (общий участок)")
ax_psd.grid(True, which="both", alpha=0.3)
ax_psd.legend(fontsize=8, loc="best")

fig_sp.suptitle(f"Спектральный анализ на общем участке ({seg_len} точ., {seg_hours:.1f} ч)",
                y=1.01, fontsize=13)
fig_sp.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig08.png"), dpi=150)
plt.close("all")

# =========================================================
# 12. ЛУЧШАЯ ИЗОТЕРМА СРЕДИ ВСЕХ ИЗОТЕРМ
# =========================================================
def longest_continuous_segment(arr):
    isnan, segments, start = np.isnan(arr), [], None
    for i, val in enumerate(isnan):
        if not val and start is None:
            start = i
        elif val and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(arr)))
    if not segments:
        return None, None
    return segments[np.argmax([e - s for s, e in segments])]


def add_day_boundaries(ax, t_arr):
    day_starts = pd.date_range(t_arr[0].normalize(), t_arr[-1].normalize(), freq="D")
    for d in day_starts:
        ax.axvline(d, color="black", ls="--", lw=0.8, alpha=0.6)


all_iso_values = np.arange(np.floor(T_min), np.ceil(T_max) + 0.01, 0.5)
all_iso_depths = {}
for T_iso in all_iso_values:
    z_iso = np.full(len(time_30s), np.nan)
    for t in range(len(time_30s)):
        z_iso[t] = interp1d(
            temps_30s[t, :],
            median_depths,
            bounds_error=False,
            fill_value=np.nan,
        )(T_iso)
    all_iso_depths[T_iso] = z_iso

best_iso = None
best_len = 0
best_start, best_end = None, None

for T_iso in all_iso_values:
    z_iso = all_iso_depths[T_iso]
    s, e = longest_continuous_segment(z_iso)
    if s is None:
        continue
    length = e - s
    if length > best_len:
        best_len = length
        best_iso = T_iso
        best_start, best_end = s, e

if best_iso is None:
    raise RuntimeError("Нет непрерывных участков для всех изотерм!")

print(
    f"\nЛучшая изотерма среди всех: {best_iso:.1f}°C "
    f"({best_len} точек, {(best_len - 1) * dt / 3600:.1f} часов)"
)

z_best = all_iso_depths[best_iso]
z_segment = z_best[best_start:best_end]
t_segment = time_30s[best_start:best_end]
z_mean_best = np.nanmean(z_segment)

fig_best, ax_best = plt.subplots(figsize=(15, 5))
ax_best.plot(t_segment, z_segment, lw=0.8, color="teal", label=f"{best_iso:.1f}°C")
ax_best.axhline(z_mean_best, color="red", ls="--", lw=1, label=f"z̄ = {z_mean_best:.1f} м")
add_day_boundaries(ax_best, t_segment)
ax_best.plot([], [], color="black", ls="--", lw=1, label="Границы суток")
ax_best.invert_yaxis()
ax_best.set_ylabel("Глубина, м")
ax_best.set_xlabel("Время")
ax_best.set_title(f"Лучшая изотерма {best_iso:.1f}°C (самый длинный непрерывный участок)")
ax_best.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
ax_best.grid(True, alpha=0.3)
ax_best.legend(fontsize=9, loc="best")
fig_best.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig09.png"), dpi=150)
plt.close("all")

# =========================================================
# 13. АНАЛИЗ ВСЕЙ ЛУЧШЕЙ ИЗОТЕРМЫ (её непрерывного участка)
#     Статистика короткопериодных волн + PSD повторяемых изотерм
# =========================================================
t_win = t_segment
z_best_win = z_segment
# Смещаем относительно среднего по всему непрерывному участку.
z_best_shift = z_best_win - np.nanmean(z_best_win)

print(f"\nИнтервал анализа: {t_win[0].strftime('%d.%m.%Y %H:%M')} — {t_win[-1].strftime('%d.%m.%Y %H:%M')}")

def detect_waves(z_shifted, dt_seconds, min_period_min=3.0, min_height_m=0.5):
    """Поиск волн между соседними минимумами по тем же критериям, что для лучшей изотермы."""
    minima, _ = find_peaks(-z_shifted)
    waves = []
    for i in range(len(minima) - 1):
        i0, i1 = minima[i], minima[i + 1]
        period_min = (i1 - i0) * dt_seconds / 60.0
        if period_min < min_period_min:
            continue
        seg = z_shifted[i0:i1 + 1]
        imax = i0 + np.argmax(seg)
        zmax = z_shifted[imax]
        h_front = zmax - z_shifted[i0]
        h_rear = zmax - z_shifted[i1]
        h_wave = 0.5 * (h_front + h_rear)
        if h_wave >= min_height_m:
            waves.append((i0, i1, imax, h_wave, period_min))
    return waves


def _grouped_moments_stats(values, n_bins):
    """Статистика группированных данных через центры интервалов и частоты."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None

    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = int(np.sum(counts))
    if n == 0:
        return None

    fi = counts.astype(float)
    c = centers

    mean = np.sum(c * fi) / n
    var = np.sum(fi * (c - mean) ** 2) / n
    std = np.sqrt(var)
    mad = np.sum(fi * np.abs(c - mean)) / n

    m1 = np.sum(fi * (c ** 1)) / n
    m2 = np.sum(fi * (c ** 2)) / n
    m3 = np.sum(fi * (c ** 3)) / n
    m4 = np.sum(fi * (c ** 4)) / n

    mu2 = np.sum(fi * ((c - mean) ** 2)) / n
    mu3 = np.sum(fi * ((c - mean) ** 3)) / n
    mu4 = np.sum(fi * ((c - mean) ** 4)) / n

    if mu2 > 0:
        skew = mu3 / (mu2 ** 1.5)
        kurt = (mu4 / (mu2 ** 2)) - 3.0
    else:
        skew = np.nan
        kurt = np.nan

    return {
        "mean": mean,
        "mad": mad,
        "var": var,
        "std": std,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
        "mu2": mu2,
        "mu3": mu3,
        "mu4": mu4,
        "skew": skew,
        "kurt": kurt,
    }


def _print_grouped_stats_block(values, n_bins, name, unit):
    s = _grouped_moments_stats(values, n_bins=n_bins)
    if s is None:
        print(f"  {name}: недостаточно данных для группированной статистики.")
        return
    unit2 = f"{unit}²" if unit else ""
    print(f"  {name}:")
    print(f"    Выборочное среднее (математическое ожидание): {s['mean']:.4f} {unit}")
    print(f"    Среднее отклонение: {s['mad']:.4f} {unit}")
    print(f"    Выборочная дисперсия: {s['var']:.6f} {unit2}")
    print(f"    Среднеквадратичное отклонение: {s['std']:.4f} {unit}")
    print(
        "    Начальные моменты: "
        f"m1={s['m1']:.6f}, m2={s['m2']:.6f}, m3={s['m3']:.6f}, m4={s['m4']:.6f}"
    )
    print(
        "    Центральные моменты: "
        f"μ2={s['mu2']:.6f}, μ3={s['mu3']:.6f}, μ4={s['mu4']:.6f}"
    )
    print(f"    Коэффициент асимметрии: {s['skew']:.6f}")
    print(f"    Коэффициент эксцесса: {s['kurt']:.6f}")


def print_wave_statistics(H, Tm, title, n_bins):
    """Печать статистики группированных данных по формулам через (c_i, f_i)."""
    print(f"\nСводная статистика ({title}):")
    print(f"  Диапазон высот: {np.min(H):.2f}–{np.max(H):.2f} м")
    print(f"  Диапазон периодов: {np.min(Tm):.2f}–{np.max(Tm):.2f} мин")
    print(f"  Число интервалов (Стерджесс): k={n_bins}")
    _print_grouped_stats_block(H, n_bins=n_bins, name="Высота H", unit="м")
    _print_grouped_stats_block(Tm, n_bins=n_bins, name="Период T", unit="мин")


def plot_wave_histograms(H, Tm, title_suffix, out_file):
    fig_hist, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(13, 5))
    n = len(H)
    k = max(1, int(np.ceil(1 + np.log2(n))))

    ax_h.hist(H, bins=k, density=True, alpha=0.7, edgecolor="black", color="steelblue")
    ax_h.set_xlabel("Высота H, м")
    ax_h.set_ylabel("Плотность")
    ax_h.set_title(f"Гистограмма высот волн ({title_suffix})")
    ax_h.grid(True, alpha=0.3)

    ax_t.hist(Tm, bins=k, density=True, alpha=0.7, edgecolor="black", color="seagreen")
    ax_t.set_xlabel("Период T, мин")
    ax_t.set_ylabel("Плотность")
    ax_t.set_title(f"Гистограмма периодов волн ({title_suffix})")
    ax_t.grid(True, alpha=0.3)

    fig_hist.tight_layout()
    fig_hist.savefig(out_file, dpi=200)
    print(f"Гистограммы сохранены: {out_file}")
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig10.png"), dpi=150)
    plt.close("all")


# Волны на лучшей изотерме.
# Ищем все локальные минимумы, затем фильтруем по T >= 3 мин и h >= 0.5 м.
selected_waves = detect_waves(z_best_shift, dt_seconds=dt, min_period_min=3.0, min_height_m=0.5)

print(f"Найдено волн на лучшей изотерме (h >= 0.5 м, T >= 3 мин): {len(selected_waves)}")
rows = []
for n, (i0, i1, _imax, h_wave, period_min) in enumerate(selected_waves, start=1):
    rows.append({
        "№": n,
        "Начало": t_win[i0].strftime("%d.%m %H:%M:%S"),
        "Конец": t_win[i1].strftime("%d.%m %H:%M:%S"),
        "Высота H, м": h_wave,
        "Период T, мин": period_min,
    })

if len(rows) > 0:
    df_waves = pd.DataFrame(rows)
    print("\nТаблица волн на лучшей изотерме:")
    print(df_waves.to_string(index=False, justify="center", float_format=lambda x: f"{x:.2f}"))

    H = df_waves["Высота H, м"].to_numpy()
    Tm = df_waves["Период T, мин"].to_numpy()
    n_waves = len(df_waves)
    k_st = max(1, int(np.ceil(1 + np.log2(n_waves))))
    print(f"  Стерджесс: n={n_waves}, k={k_st}")
    print_wave_statistics(H, Tm, title="лучшая изотерма", n_bins=k_st)
    plot_wave_histograms(
        H,
        Tm,
        title_suffix="лучшая изотерма",
        out_file="best_iso_waves_hist_full_segment.png",
    )
else:
    print("Недостаточно волн на лучшей изотерме для статистики и гистограмм.")

# Повторяемость волн по всем изотермам 11–24 °C в том же участке.
iso_window_values = np.arange(11.0, 25.0, 1.0)
window_iso_series = {}
for T_iso in iso_window_values:
    window_iso_series[T_iso] = all_iso_depths.get(T_iso, np.full(len(time_30s), np.nan))[best_start:best_end]

iso_repeats = []
for T_iso in iso_window_values:
    z_iso_win = window_iso_series[T_iso]
    support = 0
    for i0, i1, _imax, _h_best, _per in selected_waves:
        if i1 >= len(z_iso_win):
            continue
        seg = z_iso_win[i0:i1 + 1]
        if np.any(np.isnan(seg)):
            continue
        imax_iso = i0 + np.argmax(seg)
        zmax_iso = z_iso_win[imax_iso]
        h_front_iso = zmax_iso - z_iso_win[i0]
        h_rear_iso = zmax_iso - z_iso_win[i1]
        h_iso = 0.5 * (h_front_iso + h_rear_iso)
        if h_iso >= 0.5:
            support += 1
    iso_repeats.append((T_iso, support))

iso_repeats.sort(key=lambda x: x[1], reverse=True)
top_k = 3
top_isos = [x[0] for x in iso_repeats[:top_k] if x[1] > 0]
print("\nИзотермы с наибольшей повторяемостью волн:")
for T_iso, cnt in iso_repeats[:top_k]:
    print(f"  {T_iso:.1f}°C: {cnt} волн")

# PSD + наклон (лог-лог) для выбранных изотерм.
if len(top_isos) > 0:
    repeat_counts = dict(iso_repeats)
    repeat_psd_blocks = []
    fig_psd, ax_psd = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for T_iso in top_isos:
        z_win = window_iso_series[T_iso]
        if len(z_win) < 8:
            continue
        if np.any(np.isnan(z_win)):
            valid = np.isfinite(z_win)
            if np.sum(valid) < 8:
                continue
            # Линейно заполняем пропуски, чтобы корректно оценить PSD.
            z_win = np.interp(np.arange(len(z_win)), np.where(valid)[0], z_win[valid])
        sig = detrend(z_win, type="linear")
        n_pts = len(sig)
        fa = 1.0 / dt
        f_hz = np.fft.rfftfreq(n_pts, d=dt)
        f_cph = f_hz * 3600.0
        X = np.fft.rfft(sig)
        Pxx = ((1.0 / (n_pts * fa)) * (np.abs(X) ** 2)) / 3600.0

        m = (f_cph > 0) & (Pxx > 0) & np.isfinite(Pxx)
        if np.sum(m) < 3:
            continue

        fv, pv = f_cph[m], Pxx[m]
        log_f = np.log10(fv)
        log_P = np.log10(pv)
        slope, intercept = np.polyfit(log_f, log_P, 1)
        fit = 10 ** (intercept + slope * log_f)

        repeat_psd_blocks.append(
            {
                "T_iso": float(T_iso),
                "repeat_count": int(repeat_counts.get(T_iso, 0)),
                "n_pts": int(n_pts),
                "f_cph": fv.copy(),
                "Pxx": pv.copy(),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )

        ax_psd.loglog(fv, pv, lw=1.2, label=f"PSD {T_iso:.1f}°C")
        plotted_any = True

        ax_psd.loglog(fv, fit, ls="--", lw=1.0, label=f"{T_iso:.1f}°C: наклон={slope:.2f}")
        print(f"  {T_iso:.1f}°C: наклон PSD (лог-лог) = {slope:.2f}")

    _write_top_repeated_isos_psd_txt(
        repeat_psd_blocks,
        dt_s=dt,
        best_iso=float(best_iso),
        t_start_str=t_win[0].strftime("%d.%m.%Y %H:%M"),
        t_end_str=t_win[-1].strftime("%d.%m.%Y %H:%M"),
        iso_repeats_sorted=iso_repeats,
        top_k=top_k,
    )

    ax_psd.set_xlabel("Частота, цикл/час")
    ax_psd.set_ylabel("PSD, м²·час")
    ax_psd.set_title("PSD для изотерм с максимальной повторяемостью волн")
    ax_psd.grid(True, which="both", alpha=0.3)
    if plotted_any:
        ax_psd.legend(fontsize=8, loc="best")
    fig_psd.tight_layout()
    out_psd = "top_repeated_isotherms_psd_full_segment.png"
    fig_psd.savefig(out_psd, dpi=200)
    print(f"График PSD сохранен: {out_psd}")
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig11.png"), dpi=150)
    plt.close("all")
else:
    print("Нет изотерм с повторяемыми волнами для построения PSD.")

# =========================================================
# 14. КОРОТКИЙ УЧАСТОК 16.06.2023 11:00–13:00 (лучшая изотерма)
# =========================================================
short_start = pd.to_datetime("2023-06-16 11:00")
short_end = pd.to_datetime("2023-06-16 13:00")
short_mask = (t_segment >= short_start) & (t_segment <= short_end)

# Графики короткого участка (после PSD) по диапазону введённых 3 изотерм:
# -1 час, целевое окно, +1 час. Изотермы только целые в пределах [min, max].
iso_min_int = int(np.ceil(min(iso_input)))
iso_max_int = int(np.floor(max(iso_input)))
if iso_min_int > iso_max_int:
    iso_min_int = int(np.floor(min(iso_input)))
    iso_max_int = int(np.ceil(max(iso_input)))
iso_vals_short = np.arange(iso_min_int, iso_max_int + 1, 1, dtype=float)
short_iso_labels = [f"изотерма {int(T_iso)}°C" for T_iso in iso_vals_short]

windows = [
    (short_start - pd.Timedelta(hours=1), short_start, "За час до выбранного интервала"),
    (short_start, short_end, "Выбранный интервал"),
    (short_end, short_end + pd.Timedelta(hours=1), "Через час после выбранного интервала"),
]

fig_short, axes_short = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 2, 1]},
)
for ax, (w0, w1, caption) in zip(axes_short, windows):
    try:
        plot_period_heatmap_with_isotherms(
            time_arr=time_30s,
            depth_levels=median_depths,
            temp_field=temps_30s,
            iso_depths_dict=all_iso_depths,
            iso_vals=iso_vals_short,
            iso_labs=short_iso_labels,
            start_time=w0,
            end_time=w1,
            line_width=0.7,
            legend_fontsize=7,
            temp_min=iso_min_int,
            temp_max=iso_max_int,
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"{caption}\n"
            f"{w0.strftime('%d.%m.%Y %H:%M')} — {w1.strftime('%d.%m.%Y %H:%M')}"
        )
    except ValueError:
        ax.text(
            0.5,
            0.5,
            "Нет данных\nв этом временном окне",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
        )
        ax.set_title(
            f"{caption}\n"
            f"{w0.strftime('%d.%m.%Y %H:%M')} — {w1.strftime('%d.%m.%Y %H:%M')}"
        )
        ax.set_xlabel("Время")
        ax.set_ylabel("Глубина, м")
fig_short.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig12.png"), dpi=150)
plt.close("all")

if np.any(short_mask):
    t_short = t_segment[short_mask]
    z_short = z_segment[short_mask]
    z_short_shift = z_short - np.nanmean(z_short)

    selected_waves_short = detect_waves(
        z_short_shift,
        dt_seconds=dt,
        min_period_min=3.0,
        min_height_m=0.5,
    )

    print(
        f"\nКороткий участок анализа: {t_short[0].strftime('%d.%m.%Y %H:%M')} — "
        f"{t_short[-1].strftime('%d.%m.%Y %H:%M')}"
    )
    print(
        "Найдено волн на коротком участке "
        f"(h >= 0.5 м, T >= 3 мин): {len(selected_waves_short)}"
    )

    rows_short = []
    for n, (i0, i1, _imax, h_wave, period_min) in enumerate(selected_waves_short, start=1):
        rows_short.append(
            {
                "№": n,
                "Начало": t_short[i0].strftime("%d.%m %H:%M:%S"),
                "Конец": t_short[i1].strftime("%d.%m %H:%M:%S"),
                "Высота H, м": h_wave,
                "Период T, мин": period_min,
            }
        )

    if len(rows_short) > 0:
        df_waves_short = pd.DataFrame(rows_short)
        print("\nТаблица волн на коротком участке:")
        print(df_waves_short.to_string(index=False, justify="center", float_format=lambda x: f"{x:.2f}"))
    else:
        print("На коротком участке волны по заданным критериям не обнаружены.")
else:
    print("\nНет данных лучшей изотермы в интервале 16.06.2023 11:00–13:00.")
