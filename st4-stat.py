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
N_profile = np.sqrt(np.clip(N2, 0, None))           # рад/с
N_profile_cph = N_profile * 3600.0 / (2.0 * np.pi)  # цикл/час

N_max_cph = np.nanmax(N_profile_cph)
z_max = median_depths[np.nanargmax(N_profile_cph)]
T_min_N = 60.0 / N_max_cph  # период в минутах

plt.figure(figsize=(5, 7))
plt.plot(N_profile_cph, median_depths, color="darkcyan", lw=1.5, label="N(z)")
plt.scatter(N_max_cph, z_max, s=40, color="red", zorder=5,
            label=f"$N_{{max}}$ = {N_max_cph:.1f} цикл/час\n"
                  f"(T = {T_min_N:.1f} мин, z = {z_max:.1f} м)")
plt.gca().invert_yaxis()
plt.xlabel("N(z), цикл/час")
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
    N_iso = np.interp(z_mean, median_depths, N_profile_cph)  # цикл/час
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

# =========================================================
# 12. ЛУЧШАЯ ИЗОТЕРМА (САМЫЙ ДЛИННЫЙ НЕПРЕРЫВНЫЙ УЧАСТОК)
# =========================================================
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import norm, rayleigh, kstest, shapiro


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


best_iso = None
best_len = 0
best_start, best_end = None, None

for T_iso in iso_values:
    z_iso = iso_depths[T_iso]
    s, e = longest_continuous_segment(z_iso)
    if s is not None:
        length = e - s
        print(f"Изотерма {T_iso}°C: непрерывный участок {length} точек "
              f"({(length - 1) * dt / 3600:.1f} ч)")
        if length > best_len:
            best_len = length
            best_iso = T_iso
            best_start, best_end = s, e

if best_iso is None:
    raise RuntimeError("Нет непрерывных участков!")

print(f"\nЛучшая изотерма: {best_iso}°C ({best_len} точек, "
      f"{(best_len - 1) * dt / 3600:.1f} часов)")

z_best = iso_depths[best_iso]
z_segment = z_best[best_start:best_end]
t_segment = time_30s[best_start:best_end]
z_mean_best = np.nanmean(z_segment)

# --- Профиль глубины лучшей изотермы ---
fig_bp, ax_bp = plt.subplots(figsize=(14, 5))
ax_bp.plot(t_segment, z_segment, lw=0.8, color="teal")
ax_bp.axhline(z_mean_best, color="red", ls="--", lw=1,
              label=f"z̄ = {z_mean_best:.1f} м")
ax_bp.invert_yaxis()
ax_bp.set_ylabel("Глубина, м")
ax_bp.set_xlabel("Время")
ax_bp.set_title(f"Глубина изотермы {best_iso}°C (лучшая, {best_len} точ.)")
ax_bp.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
ax_bp.grid(True, alpha=0.3)
ax_bp.legend()
fig_bp.tight_layout()
plt.show()

# =========================================================
# 13. ВЫДЕЛЕНИЕ ЦУГОВ КОРОТКОПЕРИОДНЫХ ВОЛН
#     Подход: полосовой фильтр → огибающая (Гильберт) →
#     цуги = участки где огибающая ≥ 0.5 м →
#     внутри цугов считаем отдельные волны по пикам
# =========================================================
from scipy.signal import hilbert as hilbert_transform

signal_best = detrend(z_segment, type='linear')

fs_hz = 1.0 / dt
nyq = fs_hz / 2.0

low_freq = 1.0 / (10 * 60)
high_freq = 1.0 / (1.5 * 60)
b_bp, a_bp = butter(3, [low_freq / nyq, high_freq / nyq], btype='band')
filtered = filtfilt(b_bp, a_bp, signal_best)

envelope = np.abs(hilbert_transform(filtered))

amp_threshold = 0.5
above = envelope >= amp_threshold
padded = np.concatenate(([False], above, [False]))
d_pad = np.diff(padded.astype(int))
train_starts = np.where(d_pad == 1)[0]
train_ends = np.where(d_pad == -1)[0]

min_train_pts = int(2 * 60 / dt)
trains = []
for s, e in zip(train_starts, train_ends):
    if (e - s) >= min_train_pts:
        trains.append((s, e))

print(f"\n{'=' * 60}")
print(f"ЦУГИ КОРОТКОПЕРИОДНЫХ ВОЛН (огибающая ≥ {amp_threshold} м)")
print(f"{'=' * 60}")
print(f"Обнаружено цугов: {len(trains)}")

if len(trains) > 0:
    for ti, (ts, te) in enumerate(trains):
        dur_min = (te - ts) * dt / 60.0
        amp_max = np.max(envelope[ts:te])
        print(f"  Цуг {ti+1}: {t_segment[ts].strftime('%H:%M')}–{t_segment[te-1].strftime('%H:%M')}, "
              f"длит. {dur_min:.1f} мин, макс. амплитуда {amp_max:.2f} м")

wave_heights = []
wave_periods_s = []
wave_train_id = []

for ti, (ts, te) in enumerate(trains):
    seg_filt = filtered[ts:te]
    pks, _ = find_peaks(seg_filt)
    trs, _ = find_peaks(-seg_filt)
    extrema = np.sort(np.concatenate([pks, trs]))
    for j in range(len(extrema) - 1):
        i1, i2 = extrema[j], extrema[j + 1]
        h = abs(seg_filt[i1] - seg_filt[i2])
        period_s = (i2 - i1) * dt
        if h >= 0.25:
            wave_heights.append(h)
            wave_periods_s.append(period_s)
            wave_train_id.append(ti)

wave_heights = np.array(wave_heights)
wave_periods_min = np.array(wave_periods_s) / 60.0

print(f"\nВолн внутри цугов (h ≥ 0.25 м): {len(wave_heights)}")

if len(wave_heights) >= 3:
    print(f"\nВысоты волн (м):")
    print(f"  среднее   = {np.mean(wave_heights):.3f}")
    print(f"  медиана   = {np.median(wave_heights):.3f}")
    print(f"  ст.откл.  = {np.std(wave_heights):.3f}")
    print(f"  мин       = {np.min(wave_heights):.3f}")
    print(f"  макс      = {np.max(wave_heights):.3f}")
    print(f"  дисперсия = {np.var(wave_heights):.4f}")

    print(f"\nПериоды волн (мин):")
    print(f"  среднее   = {np.mean(wave_periods_min):.2f}")
    print(f"  медиана   = {np.median(wave_periods_min):.2f}")
    print(f"  ст.откл.  = {np.std(wave_periods_min):.2f}")
    print(f"  мин       = {np.min(wave_periods_min):.2f}")
    print(f"  макс      = {np.max(wave_periods_min):.2f}")

    # --- Гистограммы ---
    fig_hist, (ax_hh, ax_hp) = plt.subplots(1, 2, figsize=(12, 5))
    ax_hh.hist(wave_heights, bins='auto', edgecolor='black', alpha=0.7, density=True)
    ax_hh.set_xlabel("Высота волны, м")
    ax_hh.set_ylabel("Плотность вероятности")
    ax_hh.set_title(f"Распределение высот (N={len(wave_heights)})")
    ax_hh.grid(True, alpha=0.3)

    ax_hp.hist(wave_periods_min, bins='auto', edgecolor='black', alpha=0.7, density=True)
    ax_hp.set_xlabel("Период волны, мин")
    ax_hp.set_ylabel("Плотность вероятности")
    ax_hp.set_title(f"Распределение периодов (N={len(wave_periods_min)})")
    ax_hp.grid(True, alpha=0.3)
    fig_hist.suptitle(f"Волны внутри цугов, изотерма {best_iso}°C")
    fig_hist.tight_layout()
    plt.show()

    # --- Подгонка распределений ---
    mu_h, std_h = norm.fit(wave_heights)
    ray_loc_h, ray_scale_h = rayleigh.fit(wave_heights)

    fig_fit, (ax_f1, ax_f2) = plt.subplots(1, 2, figsize=(13, 5))
    x_h = np.linspace(wave_heights.min() * 0.8, wave_heights.max() * 1.2, 200)
    ax_f1.hist(wave_heights, bins='auto', density=True, alpha=0.5, edgecolor='black',
               label='Данные')
    ax_f1.plot(x_h, norm.pdf(x_h, mu_h, std_h), 'r-', lw=2,
               label=f'Норм. (μ={mu_h:.2f}, σ={std_h:.2f})')
    ax_f1.plot(x_h, rayleigh.pdf(x_h, ray_loc_h, ray_scale_h), 'g-', lw=2,
               label=f'Рэлей (loc={ray_loc_h:.2f}, sc={ray_scale_h:.2f})')
    ax_f1.set_xlabel("Высота, м")
    ax_f1.set_ylabel("Плотность")
    ax_f1.set_title("Подгонка распределений высот")
    ax_f1.legend(fontsize=8)
    ax_f1.grid(True, alpha=0.3)

    mu_p, std_p = norm.fit(wave_periods_min)
    ray_loc_p, ray_scale_p = rayleigh.fit(wave_periods_min)
    x_p = np.linspace(wave_periods_min.min() * 0.8, wave_periods_min.max() * 1.2, 200)
    ax_f2.hist(wave_periods_min, bins='auto', density=True, alpha=0.5, edgecolor='black',
               label='Данные')
    ax_f2.plot(x_p, norm.pdf(x_p, mu_p, std_p), 'r-', lw=2,
               label=f'Норм. (μ={mu_p:.2f}, σ={std_p:.2f})')
    ax_f2.plot(x_p, rayleigh.pdf(x_p, ray_loc_p, ray_scale_p), 'g-', lw=2,
               label=f'Рэлей (loc={ray_loc_p:.2f}, sc={ray_scale_p:.2f})')
    ax_f2.set_xlabel("Период, мин")
    ax_f2.set_ylabel("Плотность")
    ax_f2.set_title("Подгонка распределений периодов")
    ax_f2.legend(fontsize=8)
    ax_f2.grid(True, alpha=0.3)
    fig_fit.tight_layout()
    plt.show()

    # --- Проверка гипотез ---
    print(f"\n{'=' * 60}")
    print("ПРОВЕРКА ГИПОТЕЗ О РАСПРЕДЕЛЕНИИ")
    print(f"{'=' * 60}")

    print("\nВысоты:")
    ks_stat, ks_p = kstest(wave_heights, 'norm', args=(mu_h, std_h))
    print(f"  KS (нормальное): D={ks_stat:.4f}, p={ks_p:.4f}"
          f"  {'не отвергается' if ks_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
    ks_stat, ks_p = kstest(wave_heights, 'rayleigh', args=(ray_loc_h, ray_scale_h))
    print(f"  KS (Рэлей):      D={ks_stat:.4f}, p={ks_p:.4f}"
          f"  {'не отвергается' if ks_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
    if 3 <= len(wave_heights) <= 5000:
        sw_stat, sw_p = shapiro(wave_heights)
        print(f"  Шапиро–Уилк:     W={sw_stat:.4f}, p={sw_p:.4f}"
              f"  {'не отвергается' if sw_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")

    print("\nПериоды:")
    ks_stat, ks_p = kstest(wave_periods_min, 'norm', args=(mu_p, std_p))
    print(f"  KS (нормальное): D={ks_stat:.4f}, p={ks_p:.4f}"
          f"  {'не отвергается' if ks_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
    ks_stat, ks_p = kstest(wave_periods_min, 'rayleigh', args=(ray_loc_p, ray_scale_p))
    print(f"  KS (Рэлей):      D={ks_stat:.4f}, p={ks_p:.4f}"
          f"  {'не отвергается' if ks_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
    if 3 <= len(wave_periods_min) <= 5000:
        sw_stat, sw_p = shapiro(wave_periods_min)
        print(f"  Шапиро–Уилк:     W={sw_stat:.4f}, p={sw_p:.4f}"
              f"  {'не отвергается' if sw_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
else:
    print("Недостаточно волн для статистики.")

# =========================================================
# 14. СПЕКТРЫ ДО И ПОСЛЕ НАИБОЛЕЕ МОЩНОГО ЦУГА
# =========================================================
if len(trains) > 0:
    train_amps = [np.max(envelope[s:e]) for s, e in trains]
    best_train_idx = np.argmax(train_amps)
    ci_start, ci_end = trains[best_train_idx]
    ci_dur_min = (ci_end - ci_start) * dt / 60.0

    print(f"\nСамый мощный цуг №{best_train_idx+1}: "
          f"{t_segment[ci_start].strftime('%H:%M')}–{t_segment[ci_end-1].strftime('%H:%M')}, "
          f"длит. {ci_dur_min:.1f} мин")

    window_pts = int(2 * 3600 / dt)

    before_end = ci_start
    before_start = max(0, before_end - window_pts)
    after_start = ci_end
    after_end = min(len(signal_best), after_start + window_pts)

    seg_before = signal_best[before_start:before_end]
    seg_after = signal_best[after_start:after_end]

    print(f"Участок ДО цуга:    {len(seg_before)} точек "
          f"({t_segment[before_start].strftime('%H:%M')}–{t_segment[before_end].strftime('%H:%M')})")
    print(f"Участок ПОСЛЕ цуга: {len(seg_after)} точек "
          f"({t_segment[after_start].strftime('%H:%M')}–"
          f"{t_segment[min(after_end-1, len(t_segment)-1)].strftime('%H:%M')})")

    # --- График: сигнал + огибающая + цуги ---
    fig_tseg, (ax_sig, ax_env) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax_sig.plot(t_segment, signal_best, lw=0.5, color="gray", label="Сигнал (detrend)")
    ax_sig.plot(t_segment, filtered, lw=0.7, color="teal", label="Фильтр 1.5–10 мин")
    ax_sig.set_ylabel("Отклонение глубины, м")
    ax_sig.set_title(f"Изотерма {best_iso}°C")
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(fontsize=8)

    ax_env.plot(t_segment, envelope, lw=0.8, color="darkblue", label="Огибающая (Гильберт)")
    ax_env.axhline(amp_threshold, color="red", ls="--", lw=1,
                   label=f"Порог = {amp_threshold} м")
    for ti, (ts, te) in enumerate(trains):
        lbl = "Цуги" if ti == 0 else None
        ax_env.axvspan(t_segment[ts], t_segment[te - 1], color="red", alpha=0.15, label=lbl)
    ax_env.axvspan(t_segment[before_start], t_segment[before_end],
                   color="blue", alpha=0.12, label="До цуга")
    ax_env.axvspan(t_segment[after_start], t_segment[min(after_end - 1, len(t_segment) - 1)],
                   color="green", alpha=0.12, label="После цуга")
    ax_env.set_ylabel("Амплитуда, м")
    ax_env.set_xlabel("Время")
    ax_env.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax_env.grid(True, alpha=0.3)
    ax_env.legend(fontsize=8)
    fig_tseg.tight_layout()
    plt.show()

    # --- Спектры до и после ---
    def compute_psd_slope(sig, dt_s):
        n = len(sig)
        if n < 64:
            return None, None, None, None, None
        sig_d = detrend(sig, type='linear')
        f_hz = np.fft.rfftfreq(n, d=dt_s)
        f_h = f_hz * 3600.0
        fa = 1.0 / dt_s
        X = np.fft.rfft(sig_d)
        Pxx = ((1.0 / (n * fa)) * (np.abs(X) ** 2)) / 3600.0
        mask = (f_h > 0) & (Pxx > 0) & np.isfinite(Pxx)
        if np.sum(mask) < 2:
            return f_h, Pxx, np.nan, np.nan, mask
        log_f = np.log10(f_h[mask])
        log_P = np.log10(Pxx[mask])
        slope, intercept = np.polyfit(log_f, log_P, 1)
        return f_h, Pxx, slope, intercept, mask

    r_before = compute_psd_slope(seg_before, dt)
    r_after = compute_psd_slope(seg_after, dt)

    if r_before[0] is not None and r_after[0] is not None:
        f_b, Pxx_b, slope_b, int_b, mask_b = r_before
        f_a, Pxx_a, slope_a, int_a, mask_a = r_after

        print(f"\nНаклон спектра ДО цуга:    {slope_b:.2f}")
        print(f"Наклон спектра ПОСЛЕ цуга: {slope_a:.2f}")

        fig_ba, (ax_ba1, ax_ba2) = plt.subplots(1, 2, figsize=(14, 6))

        mpsd_b = (f_b > 0) & (Pxx_b > 0) & np.isfinite(Pxx_b)
        mpsd_a = (f_a > 0) & (Pxx_a > 0) & np.isfinite(Pxx_a)

        if np.any(mpsd_b):
            ax_ba1.loglog(f_b[mpsd_b], Pxx_b[mpsd_b], "b", lw=1, label="PSD до цуга")
        if np.isfinite(slope_b):
            f_fit = f_b[mask_b]
            ax_ba1.loglog(f_fit, 10 ** (int_b + slope_b * np.log10(f_fit)),
                          "r--", lw=1.5, label=f"Наклон = {slope_b:.2f}")
        ax_ba1.set_xlabel("Частота, 1/час")
        ax_ba1.set_ylabel("PSD, м²·час")
        ax_ba1.set_title(f"ДО цуга ({len(seg_before)} точек)")
        ax_ba1.grid(True, which="both", alpha=0.3)
        ax_ba1.legend(fontsize=9)

        if np.any(mpsd_a):
            ax_ba2.loglog(f_a[mpsd_a], Pxx_a[mpsd_a], "darkgreen", lw=1, label="PSD после цуга")
        if np.isfinite(slope_a):
            f_fit = f_a[mask_a]
            ax_ba2.loglog(f_fit, 10 ** (int_a + slope_a * np.log10(f_fit)),
                          "r--", lw=1.5, label=f"Наклон = {slope_a:.2f}")
        ax_ba2.set_xlabel("Частота, 1/час")
        ax_ba2.set_ylabel("PSD, м²·час")
        ax_ba2.set_title(f"ПОСЛЕ цуга ({len(seg_after)} точек)")
        ax_ba2.grid(True, which="both", alpha=0.3)
        ax_ba2.legend(fontsize=9)

        fig_ba.suptitle(f"Спектры до и после цуга (изотерма {best_iso}°C)", fontsize=13)
        fig_ba.tight_layout()
        plt.show()
    else:
        print("Недостаточно данных для построения спектров до/после цуга.")
else:
    print("Цугов не обнаружено — спектры до/после не строятся.")
