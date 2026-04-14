import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from scipy.io import savemat
from scipy.signal import find_peaks

# ----------- Функция как в MATLAB: longestwithoutNanSequence -----------
def longestwithoutNanSequence(v):
    
    v = np.array(v)

    is_valid = ~np.isnan(v)
    B = np.concatenate(([0], is_valid.astype(int), [0]))
    diff_B = np.diff(B)

    starts = np.where(diff_B == 1)[0]
    ends   = np.where(diff_B == -1)[0]

    lengths = ends - starts
    if len(lengths) == 0:
        return 0, None, None

    ind = np.argmax(lengths)
    maxLength = lengths[ind]
    st1 = starts[ind]
    st2 = ends[ind] - 1
    return maxLength, st1, st2

# ---------------------- Параметры ----------------------
xlsx_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st1.xlsx"
output_dir = r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st1-results"

# Очищаем папку перед запуском
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

sheet_dep = "dep1"
sheet_time = "ss1"
sheet_temp = "temp1"

# ---------------------- Чтение данных ----------------------
dep_df = pd.read_excel(xlsx_path, sheet_name=sheet_dep, header=None)
time_df = pd.read_excel(xlsx_path, sheet_name=sheet_time, header=None)
temp_df = pd.read_excel(xlsx_path, sheet_name=sheet_temp, header=None)

depths = np.round(dep_df.values.astype(float), 1)
temps = np.round(temp_df.values.astype(float), 1)
times_1d = pd.to_datetime(time_df.iloc[:, 0])
time_strings = times_1d.apply(lambda x: x.strftime('%d-%b-%Y %H:%M:%S'))

# ---------------------- Изотермы ----------------------
tmin, tmax = np.nanmin(temps), np.nanmax(temps)
tmin_iso = int(np.ceil(tmin))
tmax_iso = int(np.floor(tmax))
isotherms = np.arange(tmin_iso, tmax_iso+1, 1)

# ---------------------- Интерполяция глубин изотерм ----------------------
depths_isotherms = np.full((len(times_1d), len(isotherms)), np.nan)
for ti in range(len(times_1d)):
    t_profile = temps[ti, :]
    d_profile = depths[ti, :]
    valid = ~np.isnan(t_profile) & ~np.isnan(d_profile)
    if np.sum(valid) < 2:
        continue
    t_sorted = t_profile[valid][np.argsort(t_profile[valid])]
    d_sorted = d_profile[valid][np.argsort(d_profile[valid])]
    for ii, iso in enumerate(isotherms):
        if iso < t_sorted[0] or iso > t_sorted[-1]:
            depths_isotherms[ti, ii] = np.nan
        else:
            depths_isotherms[ti, ii] = np.interp(iso, t_sorted, d_sorted)

# ---------------------- Визуализация термокосы ----------------------
def plot_scheme(depths, outpath=None, show=True):
    median_depths = np.nanmedian(depths, axis=0)
    min_depth = 0
    max_depth = np.nanmax(median_depths) + 1
    fig, ax = plt.subplots(figsize=(4,8))
    ax.plot([0,0],[min_depth,max_depth], color='black', linewidth=2)
    ax.hlines(0, -0.5, 0.5, color='navy', linewidth=2)
    ax.text(0.6, 0, 'Уровень моря', va='center', ha='left', color='navy')
    ax.hlines(max_depth, -0.5, 0.5, color='saddlebrown', linewidth=3)
    ax.text(0.6, max_depth, 'Дно (прибл.)', va='center', ha='left', color='saddlebrown')
    ax.scatter(np.zeros_like(median_depths), median_depths, s=100, c='red', zorder=5)
    for i,d in enumerate(median_depths):
        ax.text(0.1,d,f'{i+1}\n{d:.1f} м', va='center', ha='left', fontsize=9)
    ax.set_ylim(max_depth+0.5, -0.5)
    ax.set_xlim(-1,1)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks([])
    ax.set_ylabel('Глубина, м')
    ax.set_title('Схема теромокосы №1 с датчиками')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Сохранено: {outpath}")
    if show:
        plt.show()
    return fig, ax

# ---------------------- Вертикальный разрез температуры ----------------------
def plot_temp_field(times_1d, depths, temps, outpath=None, show=True):
    subset_step = max(1, len(times_1d)//1000)
    time_subset = times_1d[::subset_step]
    temp_subset = temps[::subset_step, :]
    TT, DD = np.meshgrid(time_subset, np.nanmedian(depths, axis=0))
    ZZ = temp_subset.T
    fig, ax = plt.subplots(figsize=(12, 8))
    contour = ax.contourf(TT, DD, ZZ, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, label='Температура, °C')
    ax.set_ylabel('Глубина, м')
    ax.set_xlabel('Время')
    ax.set_title('Временная изменчивость температуры st1')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%b'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {outpath}")
    if show:
        plt.show()

# ---------------------- Графики изотерм ----------------------
def plot_isotherms_all(times_1d, depths_iso, isotherms, depths, outpath=None, show=True):
    fig, ax = plt.subplots(figsize=(14,6))
    colors = cm.viridis(np.linspace(0,1,len(isotherms)))
    for ii, iso in enumerate(isotherms):
        ax.plot(times_1d, depths_iso[:, ii], color=colors[ii], linewidth=1.25, label=f'{iso}°C')
    min_depth = np.nanmin(depths)
    max_depth = np.nanmax(depths)
    ax.set_ylim(min_depth, max_depth)
    ticks = np.linspace(min_depth, max_depth, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{d:.1f}" for d in ticks[::-1]])
    ax.set_ylabel('Глубина, м')
    ax.set_xlabel('Время')
    ax.set_title('График изотерм на st1')
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y %H:%M:%S'))
    fig.autofmt_xdate()
    ax.legend(ncol=2, fontsize='small')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200)
        print(f"Сохранено: {outpath}")
    if show:
        plt.show()
    plt.close(fig)
    return colors

def plot_isotherms_separately(times_1d, depths_iso, isotherms, depths, colors=None, outdir=None, show=True):
    if colors is None:
        colors = cm.viridis(np.linspace(0,1,len(isotherms)))
    for ii, iso in enumerate(isotherms):
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(times_1d, depths_iso[:, ii], color=colors[ii], linewidth=1.0, label=f'{iso}°C')
        median_depths = np.nanmedian(depths, axis=0)
        for d in median_depths:
            ax.axhline(d, color='black', linestyle='--', linewidth=1.5)
        min_depth = np.nanmin(depths)
        max_depth = np.nanmax(depths)
        ax.set_ylim(min_depth, max_depth)
        ticks = np.linspace(min_depth, max_depth, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{d:.1f}" for d in ticks[::-1]])
        ax.set_ylabel('Глубина, м')
        ax.set_xlabel('Время')
        ax.set_title(f'Глубинный профиль изотермы {iso}°C на st1')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y %H:%M:%S'))
        fig.autofmt_xdate()
        fig.tight_layout()
        if outdir:
            fig.savefig(os.path.join(outdir, f'isotherm_{iso}.png'), dpi=200)
            print(f"Сохранено: {os.path.join(outdir, f'isotherm_{iso}.png')}")
        if show:
            plt.show()
        plt.close(fig)

# ---------------------- Сохранение в .mat ----------------------
def save_data_to_mat(time_strings, depths_isotherms, isotherms, output_dir):
    data_dict = {'times': time_strings, 'isopikny': depths_isotherms, 'isotherms': isotherms}
    mat_file_path = os.path.join(output_dir, 'isotherms_data.mat')
    savemat(mat_file_path, data_dict)
    print(f"Данные сохранены в {mat_file_path}")

# ---------------------- Вызов функций ----------------------
plot_scheme(depths, outpath=os.path.join(output_dir, 'scheme.png'))
plot_temp_field(times_1d, depths, temps, outpath=os.path.join(output_dir, 'temp_field.png'))
colors = plot_isotherms_all(times_1d, depths_isotherms, isotherms, depths,
                            outpath=os.path.join(output_dir, "isotherms_all.png"))
plot_isotherms_separately(times_1d, depths_isotherms, isotherms, depths,
                          colors=colors, outdir=output_dir)
save_data_to_mat(time_strings, depths_isotherms, isotherms, output_dir)
# ===========================================
# СПЕКТР ДЛЯ ВЫБРАННОЙ ИЗОТЕРМЫ + ПОИСК ПИКОВ
# ===========================================

# Показываем список всех доступных изотерм
print("\nДоступные изотермы (°C):")
print(", ".join(str(t) for t in isotherms))

# Пользователь вводит температуру
chosen_temp = float(input("\nВведите температуру изотермы (°C), для которой построить спектр: "))

# Проверяем существование такой изотермы
if chosen_temp not in isotherms:
    raise ValueError(f"Изотермы {chosen_temp}°C нет в данных!")

# Находим индекс столбца
col = np.where(isotherms == chosen_temp)[0][0]

print(f"\nПостроение спектра для изотермы {chosen_temp}°C на st1")

# Берём временной ряд глубины
v = depths_isotherms[:, col]

# Находим самый длинный непрерывный участок без NaN
maxLen, st1, st2 = longestwithoutNanSequence(v)

if st1 is None:
    print("Нет непрерывного участка без NaN → спектр построить нельзя.")
else:
    print(f"Непрерывный участок без NaN: длина {maxLen}, индексы {st1}–{st2}")

    # === Подготовка данных ===
    v_segment = v[st1:st2+1].astype(float)
    dt = 10 / 3600   # шаг 10 секунд в часах
    Fs = 1 / dt
    Fn = Fs / 2
    N = len(v_segment)

    # === FFT ===
    Feta = np.fft.fft(v_segment) / N
    freqs = np.linspace(0, Fn, N//2 + 1)
    spectrum = 2 * np.abs(Feta[:N//2 + 1])

    # === Автоматический поиск пиков ===
    peaks, props = find_peaks(spectrum, height=np.mean(spectrum)*3)

    # === Таблица пиков ===
    peak_freqs = freqs[peaks]
    peak_amps = spectrum[peaks]
    peak_periods = 1 / peak_freqs

    peak_table = pd.DataFrame({
        "freq (1/h)": peak_freqs,
        "period (h)": peak_periods,
        "amplitude": peak_amps
    })

    print("\nНайденные пики:")
    print(peak_table)

    # === График спектра ===
    plt.figure(figsize=(12,7))
    plt.loglog(freqs, spectrum, 'm', linewidth=1.3, label='Спектр')

    # === Важные контрольные частоты ===
    f_in = 1 / 17
    f_VB = 1 / (2/60)

    for ff, color, label in [
        (1/24, 'b', '24h (суточный цикл)'),
        (1/12, 'k', '12h (полусуточный цикл)'),
        (f_in, 'g', '17h (внутр. волны)'),
        (f_VB, 'r', '2 min (вибрации V-B)')
    ]:
        plt.axvline(ff, color=color, linestyle='--', linewidth=1.4, label=label)

    plt.xlabel("Частота, 1/час")
    plt.ylabel("Амплитуда, м")
    plt.title(f"Спектр изотермы {chosen_temp}°C на st1")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()




