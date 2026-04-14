import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from scipy.signal import detrend
import gsw

# =========================================================
# ЧТЕНИЕ ДАННЫХ
# =========================================================
xlsx_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st4.xlsx"
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
    min_depth, max_depth = 0, np.nanmax(md) + 1
    fig, ax = plt.subplots(figsize=(4, 8))
    ax.plot([0, 0], [min_depth, max_depth], color="black", lw=2)
    ax.hlines(0, -0.5, 0.5, color="navy", lw=2)
    ax.text(0.6, 0, "Уровень моря", va="center", ha="left", color="navy")
    ax.hlines(max_depth, -0.5, 0.5, color="saddlebrown", lw=3)
    ax.text(0.6, max_depth, "Дно (прибл.)", va="center", ha="left", color="saddlebrown")
    ax.scatter(np.zeros_like(md), md, s=100, c="red", zorder=5)
    for i, d in enumerate(md):
        ax.text(0.1, d, f"{i+1}\n{d:.1f} м", va="center", ha="left", fontsize=9)
    ax.set_ylim(max_depth + 0.5, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("Глубина, м")
    ax.set_title("Схема термокосы №4 с датчиками")
    ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
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
plt.show()

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
plt.show()

# =========================================================
# 4. ПРОФИЛЬ ЧАСТОТЫ ВЯЙСЯЛЯ–БРЕНТА  N(z) = √( g/ρ₀(z) · dρ/dz )
# =========================================================
rho0 = np.nanmean(rho, axis=0)
drho_dz = np.gradient(rho0, median_depths)
N2 = (g / rho0) * drho_dz
N_profile = np.sqrt(np.clip(N2, 0, None))

N_max = np.nanmax(N_profile)
z_max = median_depths[np.nanargmax(N_profile)]

plt.figure(figsize=(5, 7))
plt.plot(N_profile, median_depths, color="darkcyan", marker="o", lw=1.5, markersize=5, label="N(z)")
plt.scatter(N_max, z_max, s=100, color="red", zorder=5,
            label=f"Nmax = {N_max:.3e} 1/с\nz = {z_max:.1f} м")
plt.gca().invert_yaxis()
plt.xlabel("N(z), 1/с")
plt.ylabel("Глубина, м")
plt.title("Профиль частоты Вяйсяля–Брента")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

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
plt.show()

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
plt.show()

# =========================================================
# ПАРАМЕТРЫ СПЕКТРАЛЬНОГО АНАЛИЗА
# =========================================================
Fn = (1 / dt) * 3600 / 2
Omega = 7.2921e-5
fin = (2 * Omega * np.sin(np.deg2rad(lat)) / (2 * np.pi)) * 3600
C_M = 204.0

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
    N_iso = (np.interp(z_mean, median_depths, N_profile) / (2 * np.pi)) * 3600
    S_GM = np.zeros_like(f_psd)
    mg = (f_psd > fin) & (f_psd < N_iso)
    S_GM[mg] = C_M * (fin * np.sqrt(f_psd[mg] ** 2 - fin ** 2)) / (N_iso * f_psd[mg] ** 3)

    print(f"  {T_iso}°C: z̄ = {z_mean:.1f} м, N(z̄) = {N_iso:.4f} ч⁻¹")

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
N_mean = (np.nanmean(N_profile) / (2 * np.pi)) * 3600

for ax in (ax_amp, ax_psd):
    ax.axvline(N_mean, color="black", ls="--", lw=1.3)

ax_amp.plot([], [], color="black", ls="--", lw=1.3,
            label=f"Частота Вяйсяля–Брента ({N_mean:.2f} ч⁻¹)")

ax_amp.set_ylabel("Амплитуда, м")
ax_amp.set_title("Амплитудные спектры изотерм (общий участок)")
ax_amp.grid(True, which="both", alpha=0.3)
ax_amp.legend(fontsize=8, loc="best")

ax_psd.plot([], [], color="black", ls="--", lw=1.3,
            label=f"Частота Вяйсяля–Брента ({N_mean:.2f} ч⁻¹)")
ax_psd.set_xlabel("Частота, 1/час")
ax_psd.set_ylabel("PSD, м²·час")
ax_psd.set_title("Спектральная плотность мощности + модель Гарретта–Манка (общий участок)")
ax_psd.grid(True, which="both", alpha=0.3)
ax_psd.legend(fontsize=8, loc="best")

fig_sp.suptitle(f"Спектральный анализ на общем участке ({seg_len} точ., {seg_hours:.1f} ч)",
                y=1.01, fontsize=13)
fig_sp.tight_layout()
plt.show()
