import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from scipy.signal import detrend, find_peaks
from scipy.stats import norm, rayleigh, kstest, shapiro
import gsw

xlsx_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st4.xlsx"
sheet_dep, sheet_time, sheet_temp = "dep_n", "ss", "TV"
depths = pd.read_excel(xlsx_path, sheet_name=sheet_dep, header=None).values
temps = pd.read_excel(xlsx_path, sheet_name=sheet_temp, header=None).values
time = pd.to_datetime(pd.read_excel(xlsx_path, sheet_name=sheet_time, header=None).iloc[:, 0])
dfT, dfD = pd.DataFrame(temps, index=time), pd.DataFrame(depths, index=time)
grid_30s = dfT.resample("30s").mean()
temps_30s, depths_30s, time_30s = grid_30s.values, dfD.resample("30s").mean().values, grid_30s.index
median_depths = np.nanmedian(depths_30s, axis=0)

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

plot_scheme(depths_30s)
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

lat, lon, g = 44.5, 37.98, 9.81
p = gsw.p_from_z(-median_depths, lat)
SP = 18 * np.ones_like(temps_30s)
rho = np.zeros_like(temps_30s)
for i in range(len(median_depths)):
    SA = gsw.SA_from_SP(SP[:, i], p[i], lon, lat)
    CT = gsw.CT_from_t(SA, temps_30s[:, i], p[i])
    rho[:, i] = gsw.rho(SA, CT, p[i])
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

iso_values = [float(input(f"Введите изотерму {i+1} (°C): ")) for i in range(3)]
iso_depths = {}
for T_iso in iso_values:
    z_iso = []
    for t in range(len(time_30s)):
        z_iso.append(interp1d(temps_30s[t, :], median_depths, bounds_error=False, fill_value=np.nan)(T_iso))
    iso_depths[T_iso] = np.array(z_iso)

# --- Температурное поле с изотермами (белые подписанные линии) ---
fig_ti, ax_ti = plt.subplots(figsize=(12, 6))
cf = ax_ti.contourf(TT, DD, temps_30s.T, 20, cmap="viridis")
ax_ti.invert_yaxis()
plt.colorbar(cf, ax=ax_ti, label="Температура, °C")
for T_iso, z_iso in iso_depths.items():
    ax_ti.plot(time_30s, z_iso, color="white", lw=1.5)
    valid_idx = np.where(~np.isnan(z_iso))[0]
    if len(valid_idx) > 0:
        mid = valid_idx[len(valid_idx) // 2]
        ax_ti.text(time_30s[mid], z_iso[mid], f" {T_iso}°C", color="white",
                   fontsize=10, fontweight="bold", va="bottom")
ax_ti.set_ylabel("Глубина, м")
ax_ti.set_xlabel("Дата")
ax_ti.set_title("Временная изменчивость температуры с изотермами")
ax_ti.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
fig_ti.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for T_iso, z_iso in iso_depths.items():
    plt.plot(time_30s, z_iso, lw=1, label=f"{T_iso} °C")
plt.gca().invert_yaxis()
plt.legend()
plt.ylabel("Глубина, м")
plt.title("Колебания изотерм")
plt.grid()
plt.show()

rho0 = np.nanmean(rho, axis=0)
drho_dz = np.gradient(rho0, median_depths)
N_profile = np.sqrt((g / rho0) * drho_dz)
N_max, z_max = np.nanmax(N_profile), median_depths[np.argmax(N_profile)]
plt.figure(figsize=(5, 7))
plt.plot(N_profile, median_depths, label="N(z)")
plt.scatter(N_max, z_max, s=80, label=f"Nmax = {N_max:.3e} 1/с\nz = {z_max:.2f} м")
plt.gca().invert_yaxis()
plt.xlabel("N(z), 1/с")
plt.ylabel("Глубина, м")
plt.title("Профиль частоты Вяйсяля-Брента")
plt.grid()
plt.legend()
plt.show()

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


def _ascii_log_spectrum(title, series, x0, x1, y0, y1, vlines, w=78, h=14):
    if not (np.isfinite(x0) and np.isfinite(x1) and x0 > 0 and x1 > x0):
        print(f"{title}\n  (некорректный диапазон X)\n")
        return
    if not (np.isfinite(y0) and np.isfinite(y1) and y0 > 0 and y1 > y0):
        print(f"{title}\n  (некорректный диапазон Y)\n")
        return
    lx0, lx1 = np.log10(x0), np.log10(x1)
    ly0, ly1 = np.log10(y0), np.log10(y1)
    grid = [[" " for _ in range(w)] for _ in range(h)]

    def col_from_x(xv):
        if xv <= 0:
            return None
        c = int((np.log10(xv) - lx0) / (lx1 - lx0) * (w - 1))
        return max(0, min(w - 1, c))

    def row_from_y(yv):
        if yv <= 0:
            return None
        r = int((ly1 - np.log10(yv)) / (ly1 - ly0) * (h - 1))
        return max(0, min(h - 1, r))

    for s in series:
        fx, fy, ch = s["fx"], s["fy"], s["ch"]
        ok = (fx > 0) & (fy > 0) & np.isfinite(fx) & np.isfinite(fy)
        if not np.any(ok):
            continue
        fx, fy = fx[ok], fy[ok]
        o = np.argsort(fx)
        fx, fy = fx[o], fy[o]
        for j in range(w):
            lx = lx0 + (j / max(w - 1, 1)) * (lx1 - lx0)
            xt = 10.0**lx
            if xt < fx[0] or xt > fx[-1]:
                continue
            yp = float(np.interp(xt, fx, fy))
            if not np.isfinite(yp) or yp < y0 or yp > y1:
                continue
            r = row_from_y(yp)
            if r is None:
                continue
            grid[r][j] = ch if grid[r][j] == " " else "+"

    for xv, _ in vlines:
        if not np.isfinite(xv) or xv <= 0 or xv < x0 or xv > x1:
            continue
        c = col_from_x(xv)
        if c is None:
            continue
        for r in range(h):
            if grid[r][c] == " ":
                grid[r][c] = "|"

    print(title)
    for row in grid:
        print("".join(row))
    print("-" * w)
    print(f"X: {x0:.4g} … {x1:.4g} (1/ч)   Y: {y0:.3e} … {y1:.3e} (лог-сетка)")
    print()


def _print_spectra_console(T_iso, f_fft, amp_fft, f, Pxx, S_GM, mpsd, mgp, xlo, xhi, ylo, yhi, f17v, N_isov):
    bar = "=" * 78
    print(f"\n{bar}\nСпектры (консоль), изотерма {T_iso} °C\n{bar}")
    vl = [(f17v, "17.1 ч"), (N_isov, "N(z)")]
    if f_fft.size > 0:
        _ascii_log_spectrum("FFT — амплитуда, м:  * = кривая,  | = 17.1 ч и N(z)", [{"fx": f_fft, "fy": amp_fft, "ch": "*"}], xlo, xhi, ylo, yhi, vl)
    else:
        print("FFT — нет точек для консольного графика\n")
    ser = []
    if np.any(mpsd):
        ser.append({"fx": f[mpsd], "fy": Pxx[mpsd], "ch": "#"})
    if np.any(mgp):
        ser.append({"fx": f[mgp], "fy": S_GM[mgp], "ch": "="})
    if ser:
        _ascii_log_spectrum("PSD:  # = периодограмма,  = = Гарретт–Мунк,  | = опорные частоты", ser, xlo, xhi, ylo, yhi, vl)
    else:
        print("PSD — нет точек для консольного графика\n")
    print("Легенда: * FFT(м)   # PSD   = модель Г–М   | вертикали 17.1 ч и N(z)   + пересечение кривых\n")


dt, fs, Fn = 30, (1 / 30) * 3600, (1 / 30) * 3600 / 2
Omega = 7.2921e-5
fin = (2 * Omega * np.sin(np.deg2rad(lat)) / (2 * np.pi)) * 3600
C_M, f17 = 204.0, 1 / 17.1

# =========================================================
# ОБЩИЙ НЕПРЕРЫВНЫЙ ИНТЕРВАЛ ДЛЯ ТРЁХ ИЗОТЕРМ
# =========================================================
iso_arrays = [iso_depths[T] for T in iso_values]
c_start, c_end = common_continuous_interval(*iso_arrays)

if c_start is None:
    print("Нет общего непрерывного участка для трёх изотерм!")
else:
    seg_len = c_end - c_start
    seg_hours = (seg_len - 1) * dt / 3600.0
    print(f"\nОбщий непрерывный участок: индексы {c_start}–{c_end-1}, "
          f"длина {seg_len} точек ({seg_hours:.1f} часов)")

    fig_common, axes_common = plt.subplots(len(iso_values), 2,
                                           figsize=(14, 5 * len(iso_values)),
                                           squeeze=False)
    slopes_info = []

    for idx, T_iso in enumerate(iso_values):
        z_iso = iso_depths[T_iso]
        z_segment = z_iso[c_start:c_end]
        z_mean = np.nanmean(z_segment)

        # Убираем тренд (линейный), а не среднее
        signal = detrend(z_segment, type='linear')

        # --- FFT (алгоритм без изменений) ---
        N_pts = len(signal)
        Feta = np.fft.fft(signal) / N_pts
        Fv = np.linspace(0, Fn, N_pts // 2 + 1)
        amplitude = 2 * np.abs(Feta[:len(Fv)])

        # --- PSD (алгоритм без изменений) ---
        Npsd = len(signal)
        f_hz = np.fft.rfftfreq(Npsd, d=dt)
        f_psd = f_hz * 3600.0
        fa = 1.0 / dt
        X = np.fft.rfft(signal)
        Pxx = ((1.0 / (Npsd * fa)) * (np.abs(X) ** 2)) / 3600.0

        # --- Гарретт–Мунк ---
        N_iso = (np.interp(z_mean, median_depths, N_profile) / (2 * np.pi)) * 3600
        S_GM = np.zeros_like(f_psd)
        mg = (f_psd > fin) & (f_psd < N_iso)
        S_GM[mg] = C_M * (fin * np.sqrt(f_psd[mg] ** 2 - fin ** 2)) / (N_iso * f_psd[mg] ** 3)

        # --- Оценка наклона спектра (линейная регрессия в log-log) ---
        mask_slope = (f_psd > 0) & (Pxx > 0) & np.isfinite(f_psd) & np.isfinite(Pxx)
        if np.sum(mask_slope) >= 2:
            log_f = np.log10(f_psd[mask_slope])
            log_P = np.log10(Pxx[mask_slope])
            slope, intercept = np.polyfit(log_f, log_P, 1)
        else:
            slope, intercept = np.nan, np.nan
        slopes_info.append((T_iso, slope))
        print(f"  Изотерма {T_iso}°C: наклон спектра = {slope:.2f}")

        # --- FFT график ---
        ax_fft = axes_common[idx, 0]
        mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
        if np.any(mfft):
            ax_fft.loglog(Fv[mfft], amplitude[mfft], "m", lw=1, label="FFT (амплитуда)")
        ax_fft.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_fft.axvline(N_iso, color="black", ls="--", lw=1, label="N(z)")
        ax_fft.grid(True, which="both")
        ax_fft.set_ylabel("Амплитуда FFT, м")
        ax_fft.set_title(f"{T_iso}°C, FFT, z̄≈{z_mean:.1f} м")
        ax_fft.legend(fontsize=8, loc="best")

        # --- PSD график ---
        ax_psd = axes_common[idx, 1]
        mpsd_m = (f_psd > 0) & np.isfinite(Pxx) & (Pxx > 0)
        mgp = (f_psd > 0) & np.isfinite(S_GM) & (S_GM > 0)
        if np.any(mpsd_m):
            ax_psd.loglog(f_psd[mpsd_m], Pxx[mpsd_m], "k", lw=1, label="PSD (периодограмма)")
        if np.any(mgp):
            ax_psd.loglog(f_psd[mgp], S_GM[mgp], "b-.", lw=2, label="Модель Гарретта–Манка")
        if np.isfinite(slope):
            f_fit = f_psd[mask_slope]
            P_fit = 10 ** (intercept + slope * np.log10(f_fit))
            ax_psd.loglog(f_fit, P_fit, "r--", lw=1.5, label=f"Наклон = {slope:.2f}")
        ax_psd.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_psd.axvline(N_iso, color="black", ls="--", lw=1, label="N(z)")
        ax_psd.grid(True, which="both")
        ax_psd.set_xlabel("Частота, 1/час")
        ax_psd.set_ylabel("PSD, м²·час")
        ax_psd.set_title(f"{T_iso}°C, PSD + наклон, z̄≈{z_mean:.1f} м")
        ax_psd.legend(fontsize=8, loc="best")

    fig_common.suptitle(f"Спектры на общем участке ({seg_len} точ., {seg_hours:.1f} ч)",
                        y=1.02)
    fig_common.tight_layout()
    plt.show()

    print("\n--- Сводка наклонов спектров ---")
    for T_iso, sl in slopes_info:
        print(f"  {T_iso}°C: наклон = {sl:.2f}")

# =========================================================
# ЛУЧШАЯ ИЗОТЕРМА (САМЫЙ ДЛИННЫЙ НЕПРЕРЫВНЫЙ УЧАСТОК)
# =========================================================
best_iso = None
best_len = 0
best_start, best_end = None, None

for T_iso in iso_values:
    z_iso = iso_depths[T_iso]
    s, e = longest_continuous_segment(z_iso)
    if s is not None:
        length = e - s
        print(f"Изотерма {T_iso}°C: непрерывный участок {length} точек")
        if length > best_len:
            best_len = length
            best_iso = T_iso
            best_start, best_end = s, e

if best_iso is None:
    print("Нет непрерывных участков ни для одной изотермы!")
else:
    print(f"\nЛучшая изотерма: {best_iso}°C (длина {best_len} точек, "
          f"{(best_len - 1) * dt / 3600:.1f} часов)")

    z_best = iso_depths[best_iso]
    z_segment = z_best[best_start:best_end]
    z_mean = np.nanmean(z_segment)
    t_segment = time_30s[best_start:best_end]

    # --- Профиль по глубине ---
    fig_prof, ax_prof = plt.subplots(figsize=(14, 5))
    ax_prof.plot(t_segment, z_segment, lw=1, color="teal")
    ax_prof.axhline(z_mean, color="red", ls="--", lw=1,
                    label=f"Средняя глубина z̄ = {z_mean:.1f} м")
    ax_prof.invert_yaxis()
    ax_prof.set_ylabel("Глубина, м")
    ax_prof.set_xlabel("Время")
    ax_prof.set_title(f"Профиль глубины изотермы {best_iso}°C (лучшая)")
    ax_prof.grid(True, ls="--", alpha=0.4)
    ax_prof.legend()
    ax_prof.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    fig_prof.tight_layout()
    plt.show()

    # --- Спектры для лучшей изотермы (убираем тренд) ---
    signal_best = detrend(z_segment, type='linear')

    N_pts = len(signal_best)
    Feta = np.fft.fft(signal_best) / N_pts
    Fv = np.linspace(0, Fn, N_pts // 2 + 1)
    amplitude = 2 * np.abs(Feta[:len(Fv)])

    Npsd = len(signal_best)
    f_hz = np.fft.rfftfreq(Npsd, d=dt)
    f_psd = f_hz * 3600.0
    fa = 1.0 / dt
    X_best = np.fft.rfft(signal_best)
    Pxx_best = ((1.0 / (Npsd * fa)) * (np.abs(X_best) ** 2)) / 3600.0

    N_iso = (np.interp(z_mean, median_depths, N_profile) / (2 * np.pi)) * 3600
    S_GM = np.zeros_like(f_psd)
    mg = (f_psd > fin) & (f_psd < N_iso)
    S_GM[mg] = C_M * (fin * np.sqrt(f_psd[mg] ** 2 - fin ** 2)) / (N_iso * f_psd[mg] ** 3)

    fig_best_sp, (ax_fft, ax_psd) = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
    f_fft, amp_fft = Fv[mfft], amplitude[mfft]
    ax_fft.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
    ax_fft.axvline(N_iso, color="black", ls="--", lw=1, label="N(z) — частота Вяйсяля–Брента")
    if f_fft.size > 0:
        ax_fft.loglog(f_fft, amp_fft, "m", lw=1, label="FFT (амплитуда)")
    ax_fft.grid(True, which="both")
    ax_fft.set_ylabel("Амплитуда FFT, м")
    ax_fft.set_title(f"Лучшая изотерма {best_iso}°C, спектры, z̄≈{z_mean:.1f} м")
    ax_fft.legend(fontsize=9, loc="best")

    mpsd = (f_psd > 0) & np.isfinite(Pxx_best) & (Pxx_best > 0)
    mgp = (f_psd > 0) & np.isfinite(S_GM) & (S_GM > 0)
    ax_psd.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
    ax_psd.axvline(N_iso, color="black", ls="--", lw=1, label="N(z) — частота Вяйсяля–Брента")
    if np.any(mgp):
        ax_psd.loglog(f_psd[mgp], S_GM[mgp], "b-.", lw=2, label="Модель Гарретта–Манка")
    if np.any(mpsd):
        ax_psd.loglog(f_psd[mpsd], Pxx_best[mpsd], "k", lw=1, label="PSD по DFT (периодограмма)")
    ax_psd.grid(True, which="both")
    ax_psd.set_xlabel("Частота, 1/час")
    ax_psd.set_ylabel("PSD, м²·час")
    ax_psd.legend(fontsize=9, loc="best")

    _print_spectra_console(
        best_iso, f_fft, amp_fft, f_psd, Pxx_best, S_GM, mpsd, mgp,
        *ax_fft.get_xlim(), *ax_fft.get_ylim(), f17, N_iso,
    )

    fig_best_sp.suptitle(f"Спектры лучшей изотермы {best_iso}°C ({best_len} точ.)", y=1.02)
    fig_best_sp.tight_layout()
    plt.show()

    # =========================================================
    # ВЫДЕЛЕНИЕ КОРОТКОПЕРИОДНЫХ ВОЛН (ВЫСОТА > 0.5 М)
    # =========================================================
    peaks_idx, _ = find_peaks(signal_best, distance=2)

    wave_heights = []
    wave_periods_h = []

    for i in range(len(peaks_idx) - 1):
        p1 = peaks_idx[i]
        p2 = peaks_idx[i + 1]
        seg = signal_best[p1:p2 + 1]
        h = np.max(seg) - np.min(seg)
        period_s = (p2 - p1) * dt
        wave_heights.append(h)
        wave_periods_h.append(period_s / 3600.0)

    wave_heights = np.array(wave_heights)
    wave_periods_h = np.array(wave_periods_h)

    mask_05 = wave_heights > 0.5
    heights_05 = wave_heights[mask_05]
    periods_05 = wave_periods_h[mask_05]

    print(f"\n{'=' * 60}")
    print(f"СТАТИСТИКА КОРОТКОПЕРИОДНЫХ ВОЛН (h > 0.5 м)")
    print(f"{'=' * 60}")
    print(f"Всего выделено волн: {len(wave_heights)}")
    print(f"Волн с высотой > 0.5 м: {len(heights_05)}")

    if len(heights_05) > 0:
        print(f"\nВысота волн (м):")
        print(f"  среднее  = {np.mean(heights_05):.3f}")
        print(f"  медиана  = {np.median(heights_05):.3f}")
        print(f"  стд.откл = {np.std(heights_05):.3f}")
        print(f"  мин      = {np.min(heights_05):.3f}")
        print(f"  макс     = {np.max(heights_05):.3f}")

        print(f"\nПериод волн (часы):")
        print(f"  среднее  = {np.mean(periods_05):.4f}")
        print(f"  медиана  = {np.median(periods_05):.4f}")
        print(f"  стд.откл = {np.std(periods_05):.4f}")
        print(f"  мин      = {np.min(periods_05):.4f}")
        print(f"  макс     = {np.max(periods_05):.4f}")

        # --- Гистограммы ---
        fig_hist, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(12, 5))
        ax_h.hist(heights_05, bins='auto', edgecolor='black', alpha=0.7, density=True)
        ax_h.set_xlabel("Высота волны, м")
        ax_h.set_ylabel("Плотность вероятности")
        ax_h.set_title(f"Распределение высот волн (h > 0.5 м, N={len(heights_05)})")
        ax_h.grid(True, alpha=0.3)

        ax_p.hist(periods_05, bins='auto', edgecolor='black', alpha=0.7, density=True)
        ax_p.set_xlabel("Период волны, ч")
        ax_p.set_ylabel("Плотность вероятности")
        ax_p.set_title(f"Распределение периодов волн (h > 0.5 м, N={len(periods_05)})")
        ax_p.grid(True, alpha=0.3)
        fig_hist.tight_layout()
        plt.show()

        # --- Оценка распределений ---
        mu_h, std_h = norm.fit(heights_05)
        ray_loc_h, ray_scale_h = rayleigh.fit(heights_05)

        x_h = np.linspace(heights_05.min() * 0.9, heights_05.max() * 1.1, 200)

        fig_dist, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(13, 5))
        ax_d1.hist(heights_05, bins='auto', density=True, alpha=0.5, edgecolor='black',
                   label='Данные')
        ax_d1.plot(x_h, norm.pdf(x_h, mu_h, std_h), 'r-', lw=2,
                   label=f'Норм. (μ={mu_h:.2f}, σ={std_h:.2f})')
        ax_d1.plot(x_h, rayleigh.pdf(x_h, ray_loc_h, ray_scale_h), 'g-', lw=2,
                   label=f'Рэлей (loc={ray_loc_h:.2f}, sc={ray_scale_h:.2f})')
        ax_d1.set_xlabel("Высота волны, м")
        ax_d1.set_ylabel("Плотность")
        ax_d1.set_title("Оценка распределений высот")
        ax_d1.legend(fontsize=8)
        ax_d1.grid(True, alpha=0.3)

        mu_p, std_p = norm.fit(periods_05)
        ray_loc_p, ray_scale_p = rayleigh.fit(periods_05)

        x_p = np.linspace(periods_05.min() * 0.9, periods_05.max() * 1.1, 200)
        ax_d2.hist(periods_05, bins='auto', density=True, alpha=0.5, edgecolor='black',
                   label='Данные')
        ax_d2.plot(x_p, norm.pdf(x_p, mu_p, std_p), 'r-', lw=2,
                   label=f'Норм. (μ={mu_p:.2f}, σ={std_p:.2f})')
        ax_d2.plot(x_p, rayleigh.pdf(x_p, ray_loc_p, ray_scale_p), 'g-', lw=2,
                   label=f'Рэлей (loc={ray_loc_p:.2f}, sc={ray_scale_p:.2f})')
        ax_d2.set_xlabel("Период волны, ч")
        ax_d2.set_ylabel("Плотность")
        ax_d2.set_title("Оценка распределений периодов")
        ax_d2.legend(fontsize=8)
        ax_d2.grid(True, alpha=0.3)
        fig_dist.tight_layout()
        plt.show()

        # --- Проверка гипотез о принадлежности распределению ---
        print(f"\n{'=' * 60}")
        print("ПРОВЕРКА ГИПОТЕЗ О ПРИНАДЛЕЖНОСТИ РАСПРЕДЕЛЕНИЮ")
        print(f"{'=' * 60}")

        print("\nВысоты волн:")
        ks_n_stat, ks_n_p = kstest(heights_05, 'norm', args=(mu_h, std_h))
        print(f"  KS-тест (нормальное):  D={ks_n_stat:.4f}, p={ks_n_p:.4f}"
              f"  {'не отвергается' if ks_n_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
        ks_r_stat, ks_r_p = kstest(heights_05, 'rayleigh', args=(ray_loc_h, ray_scale_h))
        print(f"  KS-тест (Рэлей):      D={ks_r_stat:.4f}, p={ks_r_p:.4f}"
              f"  {'не отвергается' if ks_r_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
        if 3 <= len(heights_05) <= 5000:
            sw_stat, sw_p = shapiro(heights_05)
            print(f"  Шапиро–Уилк (норм.):  W={sw_stat:.4f}, p={sw_p:.4f}"
                  f"  {'не отвергается' if sw_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")

        print("\nПериоды волн:")
        ks_n_stat_p, ks_n_p_p = kstest(periods_05, 'norm', args=(mu_p, std_p))
        print(f"  KS-тест (нормальное):  D={ks_n_stat_p:.4f}, p={ks_n_p_p:.4f}"
              f"  {'не отвергается' if ks_n_p_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
        ks_r_stat_p, ks_r_p_p = kstest(periods_05, 'rayleigh', args=(ray_loc_p, ray_scale_p))
        print(f"  KS-тест (Рэлей):      D={ks_r_stat_p:.4f}, p={ks_r_p_p:.4f}"
              f"  {'не отвергается' if ks_r_p_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
        if 3 <= len(periods_05) <= 5000:
            sw_stat_p, sw_p_p = shapiro(periods_05)
            print(f"  Шапиро–Уилк (норм.):  W={sw_stat_p:.4f}, p={sw_p_p:.4f}"
                  f"  {'не отвергается' if sw_p_p > 0.05 else 'ОТВЕРГАЕТСЯ'} (α=0.05)")
    else:
        print("Волн с высотой > 0.5 м не обнаружено.")
