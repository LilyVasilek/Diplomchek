import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
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

for T_iso, z_iso in iso_depths.items():
    start, end = longest_continuous_segment(z_iso)
    if start is None:
        print(f"{T_iso} °C — нет непрерывных данных")
        continue
    z_segment = z_iso[start:end]
    if len(z_segment) < 512:
        print(f"{T_iso} °C — слишком короткий сегмент")
        continue
    signal = z_segment - np.nanmean(z_segment)
    z_mean = np.nanmean(z_segment)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(time_30s, z_iso, lw=1)
    ax1.axvspan(time_30s[start], time_30s[end - 1], color="orange", alpha=0.25)
    ax1.invert_yaxis()
    ax1.set_ylabel("Глубина, м")
    ax1.set_title(f"{T_iso} °C: z_iso")
    ax1.grid(True, ls="--", alpha=0.4)
    ax2.plot(time_30s[start:end], z_segment, lw=1)
    ax2.axhline(z_mean, color="red", ls="--", lw=1, label=f"Средняя глубина участка z̄ = {z_mean:.1f} м")
    ax2.invert_yaxis()
    ax2.set_ylabel("Глубина, м")
    ax2.set_title(f"Непрерывный участок ({end - start} точ.)")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="best")
    ax2.set_xlabel("")
    dfmt = mdates.DateFormatter("%d.%m\n%H:%M")
    t0, t1 = time_30s[start].to_pydatetime(), time_30s[end - 1].to_pydatetime()
    dh = (t1 - t0).total_seconds() / 3600.0
    dloc = mdates.HourLocator(interval=2) if dh <= 12 else mdates.HourLocator(interval=6) if dh <= 48 else mdates.DayLocator(interval=1)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(dloc)
        ax.xaxis.set_major_formatter(dfmt)
    ax1.tick_params(axis="x", labelbottom=False)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    seg_h = ((end - start - 1) * dt / 3600.0) if (end - start) >= 2 else 0.0
    n17 = seg_h / 17.1 if 17.1 > 0 else np.nan
    n17i = int(np.floor(n17)) if np.isfinite(n17) else None
    fig.suptitle(f"Выбор участка для изотермы {T_iso} °C", y=1.02)
    fig.text(0.5, 0.01, f"Сегмент длительность: {seg_h:.2f} ч. Период 17.1 ч укладывается: {n17:.2f} раза (целых: {n17i}).", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()

    Npsd = len(signal)
    f_hz = np.fft.rfftfreq(Npsd, d=dt)
    f = f_hz * 3600.0
    fa = 1.0 / dt
    X = np.fft.rfft(signal)
    Pxx = ((1.0 / (Npsd * fa)) * (np.abs(X) ** 2)) / 3600.0
    N_iso = (np.interp(z_mean, median_depths, N_profile) / (2 * np.pi)) * 3600
    S_GM = np.zeros_like(f)
    mg = (f > fin) & (f < N_iso)
    S_GM[mg] = C_M * (fin * np.sqrt(f[mg] ** 2 - fin ** 2)) / (N_iso * f[mg] ** 3)
    N = len(z_segment)
    Feta = np.fft.fft(z_segment) / N
    Fv = np.linspace(0, Fn, N // 2 + 1)
    amplitude = 2 * np.abs(Feta[: len(Fv)])

    try:
        fig_sp, (ax_fft, ax_psd) = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
        mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
        f_fft, amp_fft = Fv[mfft], amplitude[mfft]
        ax_fft.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_fft.axvline(N_iso, color="black", ls="--", lw=1, label="N(z) — частота Вяйсяля–Брента")
        if f_fft.size > 0:
            ax_fft.loglog(f_fft, amp_fft, "m", lw=1, label="FFT (амплитуда)")
        else:
            ax_fft.text(0.5, 0.5, "Нет данных для FFT\n(проверить NaN/сегмент)", transform=ax_fft.transAxes, ha="center", va="center")
        ax_fft.grid(True, which="both")
        ax_fft.set_ylabel("Амплитуда FFT, м")
        ax_fft.set_title(f"{T_iso} °C, спектры, z̄≈{z_mean:.1f} м")
        ax_fft.legend(fontsize=9, loc="best")
        mpsd = (f > 0) & np.isfinite(Pxx) & (Pxx > 0)
        mgp = (f > 0) & np.isfinite(S_GM) & (S_GM > 0)
        ax_psd.axvline(f17, color="gray", ls="--", lw=1, label="17.1 ч")
        ax_psd.axvline(N_iso, color="black", ls="--", lw=1, label="N(z) — частота Вяйсяля–Брента")
        if np.any(mgp):
            ax_psd.loglog(f[mgp], S_GM[mgp], "b-.", lw=2, label="Модель Гарретта–Манка")
        if np.any(mpsd):
            ax_psd.loglog(f[mpsd], Pxx[mpsd], "k", lw=1, label="PSD по DFT (периодограмма)")
        if not np.any(mpsd) and not np.any(mgp):
            ax_psd.text(0.5, 0.5, "Нет данных для PSD\n(проверить NaN/диапазон f)", transform=ax_psd.transAxes, ha="center", va="center")
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
            T_iso, f_fft, amp_fft, f, Pxx, S_GM, mpsd, mgp,
            *ax_fft.get_xlim(), *ax_fft.get_ylim(), f17, N_iso,
        )
        fig_sp.suptitle(f"Выбранный участок: {T_iso} °C (длина сегмента = {end - start} точ.)", y=1.02)
        fig_sp.tight_layout()
        plt.show()
    except Exception as e:
        fig_err, ax_err = plt.subplots(1, 1, figsize=(9, 4))
        ax_err.axis("off")
        ax_err.text(0.02, 0.9, f"Ошибка при построении спектров для изотермы {T_iso} °C:\n{type(e).__name__}: {e}", ha="left", va="top", fontsize=11)
        ax_err.text(0.02, 0.5, "Пожалуйста, скопируйте текст ошибки.", ha="left", va="top", fontsize=10)
        plt.show()
