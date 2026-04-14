import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import detrend, butter, filtfilt, find_peaks
import gsw

# Путь к данным: аргумент командной строки или переменная окружения ST4_XLSX
_default_xlsx = r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st4.xlsx"
xlsx_path = os.environ.get("ST4_XLSX") or (sys.argv[1] if len(sys.argv) > 1 else _default_xlsx)
sheet_dep, sheet_time, sheet_temp = "dep_n", "ss", "TV"
depths = pd.read_excel(xlsx_path, sheet_name=sheet_dep, header=None).values
temps = pd.read_excel(xlsx_path, sheet_name=sheet_temp, header=None).values
time = pd.to_datetime(pd.read_excel(xlsx_path, sheet_name=sheet_time, header=None).iloc[:, 0])
dfT, dfD = pd.DataFrame(temps, index=time), pd.DataFrame(depths, index=time)
grid_30s = dfT.resample("30s").mean()
temps_30s, depths_30s, time_30s = grid_30s.values, dfD.resample("30s").mean().values, grid_30s.index
median_depths = np.nanmedian(depths_30s, axis=0)

lat, lon, g = 44.5, 37.98, 9.81
p = gsw.p_from_z(-median_depths, lat)
SP = 18 * np.ones_like(temps_30s)
rho = np.zeros_like(temps_30s)
for i in range(len(median_depths)):
    SA = gsw.SA_from_SP(SP[:, i], p[i], lon, lat)
    CT = gsw.CT_from_t(SA, temps_30s[:, i], p[i])
    rho[:, i] = gsw.rho(SA, CT, p[i])

rho0 = np.nanmean(rho, axis=0)
drho_dz = np.gradient(rho0, median_depths)
N_profile = np.sqrt((g / rho0) * drho_dz)

dt, fs, Fn = 30, (1 / 30) * 3600, (1 / 30) * 3600 / 2
Omega = 7.2921e-5
fin = (2 * Omega * np.sin(np.deg2rad(lat)) / (2 * np.pi)) * 3600
C_M, f17 = 204.0, 1 / 17.1


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


def longest_common_finite_run(*arrays):
    """Максимальный отрезок индексов, где все ряды конечны (не NaN)."""
    n = len(arrays[0])
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        if len(a) != n:
            raise ValueError("Длины рядов для общего интервала должны совпадать")
        mask &= np.isfinite(a)
    segments, start = [], None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, n))
    if not segments:
        return None, None
    return segments[np.argmax([e - s for s, e in segments])]


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


def estimate_loglog_slope(f_hz, y, f_lo_hz=None, f_hi_hz=None, min_pts=5):
    """Наклон β в приближении y ∝ f^β (по log10–log10). f в Гц, y > 0."""
    f = np.asarray(f_hz, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = (f > 0) & (yy > 0) & np.isfinite(f) & np.isfinite(yy)
    f, yy = f[ok], yy[ok]
    if f.size < min_pts:
        return np.nan, np.nan, np.nan, None
    if f_lo_hz is None:
        f_lo_hz = np.nanpercentile(f, 15)
    if f_hi_hz is None:
        f_hi_hz = np.nanpercentile(f, 85)
    band = (f >= f_lo_hz) & (f <= f_hi_hz)
    if np.sum(band) < min_pts:
        band = ok
    lf = np.log10(f[band])
    ly = np.log10(yy[band])
    msk = np.isfinite(lf) & np.isfinite(ly)
    lf, ly = lf[msk], ly[msk]
    if lf.size < min_pts:
        return np.nan, np.nan, np.nan, None
    slope, intercept, r, p, _ = stats.linregress(lf, ly)
    return slope, r**2, p, (lf, ly)


def build_spectra_for_segment(z_segment, z_mean_for_N, T_iso_label, time_seg, show_plots=True):
    """
    z_segment — ряд глубины изотермы; линейный детренд (не только среднее).
    """
    if len(z_segment) < 512:
        print(f"{T_iso_label} °C — слишком короткий сегмент ({len(z_segment)} точ.)")
        return None
    signal = detrend(z_segment, type="linear")
    z_mean = float(np.nanmean(z_segment))

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.plot(time_seg, z_segment, lw=1)
    ax2.axhline(z_mean, color="red", ls="--", lw=1, label=f"z̄ = {z_mean:.1f} м")
    ax2.invert_yaxis()
    ax2.set_ylabel("Глубина, м")
    ax2.set_title(f"Участок: линейный детренд для спектра ({len(z_segment)} точ.)")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="best")
    dfmt = mdates.DateFormatter("%d.%m\n%H:%M")
    t0, t1 = time_seg[0].to_pydatetime(), time_seg[-1].to_pydatetime()
    dh = (t1 - t0).total_seconds() / 3600.0
    dloc = mdates.HourLocator(interval=2) if dh <= 12 else mdates.HourLocator(interval=6) if dh <= 48 else mdates.DayLocator(interval=1)
    ax2.xaxis.set_major_locator(dloc)
    ax2.xaxis.set_major_formatter(dfmt)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    seg_h = ((len(z_segment) - 1) * dt / 3600.0) if len(z_segment) >= 2 else 0.0
    n17 = seg_h / 17.1 if 17.1 > 0 else np.nan
    n17i = int(np.floor(n17)) if np.isfinite(n17) else None
    fig.suptitle(f"Сегмент для изотермы {T_iso_label} °C", y=1.02)
    fig.text(0.5, 0.01, f"Длительность: {seg_h:.2f} ч. Период 17.1 ч: {n17:.2f} раз (целых: {n17i}). Детренд: линейный.", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    Npsd = len(signal)
    f_hz = np.fft.rfftfreq(Npsd, d=dt)
    f = f_hz * 3600.0
    fa = 1.0 / dt
    X = np.fft.rfft(signal)
    Pxx = ((1.0 / (Npsd * fa)) * (np.abs(X) ** 2)) / 3600.0
    N_iso = (np.interp(z_mean_for_N, median_depths, N_profile) / (2 * np.pi)) * 3600
    S_GM = np.zeros_like(f)
    mg = (f > fin) & (f < N_iso)
    S_GM[mg] = C_M * (fin * np.sqrt(f[mg] ** 2 - fin ** 2)) / (N_iso * f[mg] ** 3)
    N = len(signal)
    Feta = np.fft.fft(signal) / N
    Fv = np.linspace(0, Fn, N // 2 + 1)
    amplitude = 2 * np.abs(Feta[: len(Fv)])

    # Наклоны: f в Гц для FFT амплитуды; для PSD тот же масштаб
    f_hz_pos = f_hz[f_hz > 0]
    P_pos = Pxx[f_hz > 0]
    beta_psd, r2_psd, p_psd, _ = estimate_loglog_slope(f_hz_pos, P_pos, min_pts=5)
    Fv_pos = Fv[Fv > 0]
    amp_pos = amplitude[Fv > 0]
    beta_fft, r2_fft, p_fft, _ = estimate_loglog_slope(Fv_pos, amp_pos, min_pts=5)
    print(f"  Наклон спектра PSD (log–log): β ≈ {beta_psd:.3f}, R² ≈ {r2_psd:.3f}, p ≈ {p_psd:.3g}")
    print(f"  Наклон FFT амплитуды (log–log): β ≈ {beta_fft:.3f}, R² ≈ {r2_fft:.3f}, p ≈ {p_fft:.3g}")

    try:
        fig_sp, (ax_fft, ax_psd) = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
        mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
        f_fft, amp_fft = Fv[mfft], amplitude[mfft]
        ax_fft.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_fft.axvline(N_iso, color="black", ls="--", lw=1, label="N(z)")
        if f_fft.size > 0:
            ax_fft.loglog(f_fft, amp_fft, "m", lw=1, label="FFT (амплитуда)")
        else:
            ax_fft.text(0.5, 0.5, "Нет данных для FFT", transform=ax_fft.transAxes, ha="center", va="center")
        ax_fft.grid(True, which="both")
        ax_fft.set_ylabel("Амплитуда FFT, м")
        ax_fft.set_title(f"{T_iso_label} °C, спектры, z̄≈{z_mean:.1f} м (детренд линейный)")
        ax_fft.legend(fontsize=9, loc="best")
        mpsd = (f > 0) & np.isfinite(Pxx) & (Pxx > 0)
        mgp = (f > 0) & np.isfinite(S_GM) & (S_GM > 0)
        ax_psd.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_psd.axvline(N_iso, color="black", ls="--", lw=1, label="N(z)")
        if np.any(mgp):
            ax_psd.loglog(f[mgp], S_GM[mgp], "b-.", lw=2, label="Модель Гарретта–Манка")
        if np.any(mpsd):
            ax_psd.loglog(f[mpsd], Pxx[mpsd], "k", lw=1, label="PSD по DFT")
        if not np.any(mpsd) and not np.any(mgp):
            ax_psd.text(0.5, 0.5, "Нет данных для PSD", transform=ax_psd.transAxes, ha="center", va="center")
        ax_psd.grid(True, which="both")
        ax_psd.set_xlabel("Частота, 1/час")
        ax_psd.set_ylabel("PSD, м²·час (для оси 1/час)")
        ax_psd.legend(fontsize=9, loc="best")
        xmin = max(fin, np.nanmin(f[f > 0])) if np.any(f > 0) else fin
        xmax = max(np.nanmax(f), np.nanmax(Fv))
        if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
            xleft = xmin * 0.52
            if f17 > 0 and xleft >= f17 * 0.96:
                xleft = f17 * 0.50
            if xleft > 0 and xleft < xmax:
                ax_fft.set_xlim([xleft, xmax])
                ax_psd.set_xlim([xleft, xmax])
        yc = []
        if f_fft.size > 0:
            yv = amp_fft[np.isfinite(amp_fft) & (amp_fft > 0)]
            if yv.size > 0:
                yc.append(yv)
        if np.any(mpsd):
            yc.append(Pxx[mpsd])
        if np.any(mgp):
            yc.append(S_GM[mgp])
        if yc:
            ya = np.concatenate([np.ravel(v) for v in yc if v is not None and len(v) > 0])
            ya = ya[(ya > 0) & np.isfinite(ya)]
            if ya.size > 0:
                ymin, ymax = np.nanmin(ya), np.nanmax(ya)
                ax_fft.set_ylim([ymin * 0.65, ymax * 1.45])
                ax_psd.set_ylim([ymin * 0.65, ymax * 1.45])
        try:
            ax_fft.text(f17, ax_fft.get_ylim()[1] * 0.7, "17.1 ч", rotation=90, color="gray", va="bottom", ha="right")
        except Exception:
            pass
        _print_spectra_console(
            T_iso_label, f_fft, amp_fft, f, Pxx, S_GM, mpsd, mgp,
            *ax_fft.get_xlim(), *ax_fft.get_ylim(), f17, N_iso,
        )
        fig_sp.suptitle(f"Изотерма {T_iso_label} °C, N={len(z_segment)} точ.", y=1.02)
        fig_sp.tight_layout()
        if show_plots:
            plt.show()
        else:
            plt.close(fig_sp)
    except Exception as e:
        fig_err, ax_err = plt.subplots(1, 1, figsize=(9, 4))
        ax_err.axis("off")
        ax_err.text(0.02, 0.9, f"Ошибка спектров {T_iso_label} °C:\n{type(e).__name__}: {e}", ha="left", va="top", fontsize=11)
        plt.show()

    return {
        "z_mean": z_mean,
        "N_iso": N_iso,
        "beta_psd": beta_psd,
        "beta_fft": beta_fft,
        "f": f,
        "Pxx": Pxx,
        "Fv": Fv,
        "amplitude": amplitude,
    }


def distribution_hypothesis_tests(x, name="выборка"):
    """Проверки нормальности и согласия с несколькими распределениями."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        print(f"{name}: слишком мало точек для тестов ({x.size})")
        return
    print(f"\n--- Распределение: {name}, n = {x.size} ---")
    sh = stats.shapiro(x)
    print(f"Шапиро–Уилк: W = {sh.statistic:.4f}, p = {sh.pvalue:.4g}")
    # Сравнение с подогнанными распределениями (KS после оценки параметров)
    results = []
    # Нормальное
    loc, scale = np.mean(x), np.std(x, ddof=1)
    if scale > 0:
        ks_n = stats.kstest(x, lambda v: stats.norm.cdf(v, loc, scale))
        results.append(("нормальное", ks_n.statistic, ks_n.pvalue))
    # Экспоненциальное (x должен быть > 0)
    if np.all(x > 0):
        scale_e = np.mean(x)
        ks_e = stats.kstest(x, lambda v: stats.expon.cdf(v, scale=scale_e))
        results.append(("экспоненциальное", ks_e.statistic, ks_e.pvalue))
    # Логнормальное
    if np.all(x > 0):
        s_ln = np.log(x)
        ks_ln = stats.kstest(np.log(x), lambda v: stats.norm.cdf(v, np.mean(s_ln), np.std(s_ln, ddof=1)))
        results.append(("логнормальное (ln x)", ks_ln.statistic, ks_ln.pvalue))
    for label, D, pv in results:
        print(f"Kolmogorov–Smirnov vs {label}: D = {D:.4f}, p ≈ {pv:.4g}")
    print("(Малый p при KS: отклонение гипотезы о виде распределения с оценёнными параметрами.)")


def analyze_short_waves_best(z_seg, T_best, min_amplitude_m=0.5, highpass_hours=6.0):
    """
    Короткопериодные колебания: высокочастотная составляющая (ВЧ выше периода highpass_hours),
    выделение экстремумов с |отклонение| > min_amplitude_m от нуля после ВЧ-фильтра.
    """
    z = np.asarray(z_seg, dtype=float)
    n = z.size
    if n < 64:
        print("Сегмент слишком короткий для анализа коротких волн.")
        return
    y = detrend(z, type="linear")
    nyq = 0.5 * (1.0 / dt)
    fcut = 1.0 / (highpass_hours * 3600.0)
    Wn = min(0.99 * nyq, max(fcut / nyq, 0.001))
    b, a = butter(4, Wn, btype="high")
    y_hp = filtfilt(b, a, y)
    # Пики вверх и вниз
    pos_peaks, pp = find_peaks(y_hp, height=min_amplitude_m)
    neg_peaks, np_ = find_peaks(-y_hp, height=min_amplitude_m)
    heights = np.concatenate([y_hp[pos_peaks], -y_hp[neg_peaks]])
    all_idx = np.sort(np.concatenate([pos_peaks, neg_peaks]))
    if all_idx.size < 2:
        print("Нет выделенных экстремумов с амплитудой > {:.2f} м.".format(min_amplitude_m))
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax[0].plot(np.arange(n) * dt / 3600.0, y_hp, "k", lw=0.8)
        ax[0].set_ylabel("ВЧ глубина, м")
        ax[0].set_title(f"Изотерма {T_best} °C: ВЧ-сигнал (>{highpass_hours:g} ч)")
        ax[1].hist(heights, bins=min(20, max(5, len(heights))), color="steelblue", edgecolor="k", alpha=0.85)
        ax[1].set_xlabel("Амплитуда, м")
        plt.tight_layout()
        plt.show()
        return
    periods_s = np.diff(all_idx) * dt
    periods_h = periods_s / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    t_h = np.arange(n) * dt / 3600.0
    axes[0].plot(t_h, y_hp, "k", lw=0.7, label="ВЧ")
    axes[0].scatter(all_idx * dt / 3600.0, y_hp[all_idx], c="red", s=25, zorder=5, label="экстремумы")
    axes[0].axhline(min_amplitude_m, color="gray", ls="--", lw=0.8)
    axes[0].axhline(-min_amplitude_m, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("м")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Короткопериодные волны (|A| > {min_amplitude_m} м), изотерма {T_best} °C")
    axes[1].hist(periods_h, bins=min(30, max(5, len(periods_h))), color="coral", edgecolor="k", alpha=0.85)
    axes[1].set_xlabel("Период (между экстремумами), ч")
    axes[1].set_ylabel("частота")
    axes[2].hist(np.abs(heights), bins=min(25, max(5, len(heights))), color="seagreen", edgecolor="k", alpha=0.85)
    axes[2].set_xlabel("|Амплитуда|, м")
    axes[2].set_ylabel("частота")
    plt.tight_layout()
    plt.show()

    print(f"\nКороткие волны: число экстремумов = {all_idx.size}, парных интервалов = {periods_h.size}")
    if periods_h.size:
        print(f"  Период: медиана = {np.median(periods_h):.4g} ч, mean = {np.mean(periods_h):.4g} ч, std = {np.std(periods_h):.4g} ч")
    if heights.size:
        print(f"  Амплитуда: медиана |A| = {np.median(np.abs(heights)):.4g} м, max |A| = {np.max(np.abs(heights)):.4g} м")

    distribution_hypothesis_tests(periods_h, "периоды (ч)")
    distribution_hypothesis_tests(np.abs(heights), "|амплитуда| (м)")


# --- основной сценарий ---
plot_scheme(depths_30s)

TT, DD = np.meshgrid(time_30s, median_depths)
fig_t, ax_t = plt.subplots(figsize=(12, 6))
cf_t = ax_t.contourf(TT, DD, temps_30s.T, 20, cmap="viridis")
ax_t.invert_yaxis()
fig_t.colorbar(cf_t, ax=ax_t, label="Температура, °C")
ax_t.set_ylabel("Глубина, м")
ax_t.set_xlabel("Дата")
ax_t.set_title("Временная изменчивость температуры")
ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
fig_t.tight_layout()

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
# Нижний / средний / верхний слой: по возрастанию температуры (холоднее глубже)
iso_values_sorted = sorted(iso_values)
T_low, T_mid, T_high = iso_values_sorted
print(f"Порядок слоёв (по T): нижний {T_low}, средний {T_mid}, верхний {T_high} °C")

iso_depths = {}
for T_iso in iso_values_sorted:
    z_iso = []
    for t in range(len(time_30s)):
        z_iso.append(interp1d(temps_30s[t, :], median_depths, bounds_error=False, fill_value=np.nan)(T_iso))
    iso_depths[T_iso] = np.array(z_iso)

z_lo = iso_depths[T_low]
z_mi = iso_depths[T_mid]
z_hi = iso_depths[T_high]
com_start, com_end = longest_common_finite_run(z_lo, z_mi, z_hi)
if com_start is None:
    raise SystemExit("Нет общего участка, где все три изотермы определены непрерывно.")

# Подписанные белые линии на поле t–z
colors_iso = {"label": "white"}
for T_iso, z_iso in iso_depths.items():
    ax_t.plot(time_30s, z_iso, color="white", lw=1.8, zorder=10)
    # подпись у середины ряда
    mid = len(time_30s) // 2
    zmid = z_iso[mid]
    if np.isfinite(zmid):
        ax_t.text(
            time_30s[mid], zmid, f" {T_iso:g} °C",
            color="white", fontsize=9, fontweight="bold",
            va="center", ha="left",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
            zorder=11,
        )
ax_t.axvspan(time_30s[com_start], time_30s[com_end - 1], color="cyan", alpha=0.15, zorder=2, label="общий интервал")
ax_t.legend(loc="upper right", fontsize=8)
fig_t.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for T_iso, z_iso in iso_depths.items():
    plt.plot(time_30s, z_iso, lw=1, label=f"{T_iso} °C")
plt.axvspan(time_30s[com_start], time_30s[com_end - 1], color="orange", alpha=0.2, label="общий участок")
plt.gca().invert_yaxis()
plt.legend()
plt.ylabel("Глубина, м")
plt.title("Колебания изотерм")
plt.grid()
plt.show()

print(
    f"\nОбщий непрерывный интервал (все три изотермы): индексы [{com_start}, {com_end}), "
    f"длина {com_end - com_start} точ. (~{(com_end - com_start - 1) * dt / 3600:.2f} ч)"
)

# Спектры на общем участке (алгоритм тот же; сигнал — линейный детренд)
print("\n=== Спектры на ОБЩЕМ участке (детренд линейный) ===")
for T_iso in iso_values_sorted:
    z_full = iso_depths[T_iso]
    z_seg = z_full[com_start:com_end]
    if len(z_seg) < 512:
        print(f"{T_iso} °C — общий сегмент короче 512 точек, спектр пропущен.")
        continue
    z_mean_N = float(np.nanmean(z_seg))
    build_spectra_for_segment(z_seg, z_mean_N, T_iso, time_30s[com_start:com_end], show_plots=True)

# Лучшая изотерма — самый длинный индивидуальный непрерывный участок
lengths = {}
for T_iso, z_iso in iso_depths.items():
    s, e = longest_continuous_segment(z_iso)
    lengths[T_iso] = (e - s) if s is not None else 0
T_best = max(lengths, key=lengths.get)
s_b, e_b = longest_continuous_segment(iso_depths[T_best])
print(f"\n«Лучшая» изотерма (максимальная длина непрерывного участка): {T_best} °C, длина = {lengths[T_best]} точ.")

plt.figure(figsize=(12, 5))
plt.plot(time_30s, iso_depths[T_best], lw=1, color="navy")
if s_b is not None:
    plt.axvspan(time_30s[s_b], time_30s[e_b - 1], color="lime", alpha=0.2, label="лучший сегмент")
plt.gca().invert_yaxis()
plt.ylabel("Глубина, м")
plt.title(f"Изотерма {T_best} °C — выбранный сегмент")
plt.legend()
plt.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.show()

if s_b is None:
    raise SystemExit("Нет непрерывного участка для лучшей изотермы.")

z_best_seg = iso_depths[T_best][s_b:e_b]
time_best = time_30s[s_b:e_b]

# Профиль T(z) — средний по времени на участке лучшей изотермы
T_prof = np.nanmean(temps_30s[s_b:e_b, :], axis=0)
fig_p, ax_p = plt.subplots(figsize=(5, 7))
ax_p.plot(T_prof, median_depths, "b-o", ms=4, lw=1)
ax_p.invert_yaxis()
ax_p.set_xlabel("Температура, °C")
ax_p.set_ylabel("Глубина, м")
ax_p.set_title(f"Профиль T(z), среднее по времени (изотерма {T_best} °C, сегмент)")
ax_p.grid(True, ls="--", alpha=0.4)
fig_p.tight_layout()
plt.show()

print(f"\n=== Спектры для лучшей изотермы {T_best} °C (её максимальный сегмент) ===")
build_spectra_for_segment(
    z_best_seg, float(np.nanmean(z_best_seg)), T_best, time_best, show_plots=True,
)

print(f"\n=== Короткопериодные волны (|A| > 0.5 м после ВЧ-фильтра), изотерма {T_best} °C ===")
analyze_short_waves_best(z_best_seg, T_best, min_amplitude_m=0.5, highpass_hours=6.0)

plt.figure(figsize=(5, 7))
plt.plot(N_profile, median_depths, label="N(z)")
plt.gca().invert_yaxis()
plt.xlabel("N(z), 1/с")
plt.ylabel("Глубина, м")
plt.title("Профиль частоты Вяйсяля-Брента")
plt.grid()
plt.legend()
plt.show()
