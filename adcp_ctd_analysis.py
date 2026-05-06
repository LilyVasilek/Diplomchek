import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import griddata
import pywt
import gsw
import os

# =========================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def julian_to_datetime(jd):
    return datetime.fromordinal(int(jd)) + timedelta(days=jd % 1) - timedelta(days=366)


def fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')


def compute_speed_direction(Ve, Vn):
    speed = np.sqrt(Ve**2 + Vn**2)
    direction = np.degrees(np.arctan2(Ve, Vn)) % 360
    return speed, direction


def stats_table(Ve_col, Vn_col, label, dt_minutes=1.0):
    """Compute statistical parameters for one depth level."""
    u = Ve_col.dropna().values
    v = Vn_col.dropna().values
    speed = np.sqrt(u**2 + v**2)

    u_pos_pct = 100.0 * np.sum(u > 0) / len(u) if len(u) > 0 else np.nan
    u_neg_pct = 100.0 * np.sum(u < 0) / len(u) if len(u) > 0 else np.nan

    # N_rev: number of sign changes of u component
    signs = np.sign(u[u != 0])
    n_rev = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0

    return {
        'Горизонт': label,
        'U ср, м/с': round(float(np.mean(u)), 4),
        'U макс, м/с': round(float(np.max(u)), 4),
        'U мин, м/с': round(float(np.min(u)), 4),
        '|U std|, м/с': round(float(np.std(u)), 4),
        'V ср, м/с': round(float(np.mean(v)), 4),
        'V макс, м/с': round(float(np.max(v)), 4),
        'V мин, м/с': round(float(np.min(v)), 4),
        '|V std|, м/с': round(float(np.std(v)), 4),
        'Скор. ср, м/с': round(float(np.mean(speed)), 4),
        'Скор. макс, м/с': round(float(np.max(speed)), 4),
        'U pos, %': round(u_pos_pct, 1),
        'U neg, %': round(u_neg_pct, 1),
        'N rev': n_rev,
    }


# =========================================================
# 2. ПУТИ К ФАЙЛАМ
# =========================================================
# Папка с данными: рядом со скриптом (или укажите абсолютный путь)
base_path = os.path.dirname(os.path.abspath(__file__))

adcp_files = [
    os.path.join(base_path, "ADCP_1.txt"),
    os.path.join(base_path, "ADCP_2.txt"),
    os.path.join(base_path, "ADCP_3.txt"),
]

ctd_file = os.path.join(base_path, "CTD.txt")

# =========================================================
# 3. ЧТЕНИЕ ADCP
# =========================================================
adcp_list = []

for file in adcp_files:
    df = pd.read_csv(file, sep='\t', encoding='cp1251')

    for col in ['Ve', 'Vn', 'Vz', 'Depth']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Ve', 'Vn', 'Depth'])
    df['datetime'] = df['iTimeDbl'].apply(julian_to_datetime)

    adcp_list.append(df)

adcp_raw = pd.concat(adcp_list).sort_values(['datetime', 'Depth']).reset_index(drop=True)

# =========================================================
# 4. ФОРМИРОВАНИЕ СЕТКИ ГЛУБИНА × ВРЕМЯ
# =========================================================
# Уникальные глубины (бины) и временные моменты
depth_bins = np.sort(adcp_raw['Depth'].unique())
time_index = adcp_raw['datetime'].unique()
time_index.sort()

# Разворачиваем в wide-формат: строки = время, столбцы = глубина
adcp_Ve = adcp_raw.pivot_table(index='datetime', columns='Depth', values='Ve', aggfunc='mean')
adcp_Vn = adcp_raw.pivot_table(index='datetime', columns='Depth', values='Vn', aggfunc='mean')

# Преобразуем индекс в DatetimeIndex (нужно для resample)
adcp_Ve.index = pd.DatetimeIndex(adcp_Ve.index)
adcp_Vn.index = pd.DatetimeIndex(adcp_Vn.index)

# Интерполируем пропуски по времени (вдоль строк) для каждой глубины
adcp_Ve = adcp_Ve.interpolate(axis=0, limit=5)
adcp_Vn = adcp_Vn.interpolate(axis=0, limit=5)

# Скорость и направление
Ve_grid = adcp_Ve.values          # shape: (ntime, ndepth)
Vn_grid = adcp_Vn.values
speed_grid, dir_grid = compute_speed_direction(Ve_grid, Vn_grid)

times = adcp_Ve.index.to_pydatetime()
depths = adcp_Ve.columns.values

# Шаг дискретизации (в секундах)
dt_sec = np.median(np.diff([t.timestamp() for t in times]))
dt_min = dt_sec / 60.0
fs = 1.0 / dt_sec  # Гц

print(f"Временной шаг ADCP: {dt_sec:.1f} с ({dt_min:.2f} мин)")
print(f"Количество горизонтов: {len(depths)}, глубины: {depths[0]:.1f}–{depths[-1]:.1f} м")

# =========================================================
# 5. ЧТЕНИЕ CTD
# =========================================================
ctd = pd.read_csv(ctd_file, sep=',', encoding='cp1251')

ctd['datetime'] = pd.to_datetime(ctd['DATE'] + ' ' + ctd['TIME'], errors='coerce')

ctd = ctd.dropna(subset=[
    'Temp',
    'Sal(psu)',
    'Pressure(psia)',
    'Depth(m)',
    'GPSLatitude',
    'GPSLongitude',
    'datetime'
])

# =========================================================
# 6. ЧАСТОТА ВЯЙСЯЛЯ–БРЕНТА (CTD)
# =========================================================
ctd = ctd.sort_values('Depth(m)').reset_index(drop=True)
ctd = ctd.loc[ctd['Depth(m)'].diff().fillna(1) != 0]

p = ctd['Depth(m)'].values  # приближение давления

SA = gsw.SA_from_SP(
    ctd['Sal(psu)'].values,
    p,
    ctd['GPSLongitude'].values,
    ctd['GPSLatitude'].values
)

CT = gsw.CT_from_t(SA, ctd['Temp'].values, p)
rho = gsw.rho(SA, CT, p)

drho_dz = np.gradient(rho, ctd['Depth(m)'].values)

g = 9.81
rho0 = 1025

N2 = -g / rho0 * drho_dz
N = np.sqrt(np.clip(N2, 0, None))

fig, ax = plt.subplots(figsize=(5, 7))
ax.plot(N, ctd['Depth(m)'], linewidth=1.5)
ax.invert_yaxis()
ax.grid(True)
ax.set_xlabel('Частота Вяйсяля–Брента N, рад/с')
ax.set_ylabel('Глубина, м')
ax.set_title('Вертикальный профиль частоты\nВяйсяля–Брента по данным CTD')
plt.tight_layout()
plt.savefig('fig_01_brunt_vaisala.png', dpi=150)
plt.close()

# =========================================================
# 7. ГРАФИКИ ГЛУБИНА–ВРЕМЯ (U, V, СКОРОСТЬ, НАПРАВЛЕНИЕ)
# =========================================================
T_mpl = mdates.date2num(times)
D_mesh, T_mesh = np.meshgrid(depths, T_mpl)

def make_depth_time_plot(data, depths, times_mpl, title, cbar_label, cmap,
                         fname, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    pcm = ax.pcolormesh(
        T_mesh, D_mesh, data,
        shading='nearest', cmap=cmap,
        vmin=vmin, vmax=vmax
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.colorbar(pcm, ax=ax, label=cbar_label)
    ax.set_xlabel('Дата/время')
    ax.set_ylabel('Глубина, м')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# Симметричные пределы для компонент скорости
u_lim = np.nanpercentile(np.abs(Ve_grid), 98)
v_lim = np.nanpercentile(np.abs(Vn_grid), 98)

make_depth_time_plot(
    Ve_grid, depths, T_mpl,
    'Зональная компонента скорости (U), м/с',
    'U, м/с', 'RdBu_r',
    'fig_02_depth_time_U.png',
    vmin=-u_lim, vmax=u_lim
)

make_depth_time_plot(
    Vn_grid, depths, T_mpl,
    'Меридиональная компонента скорости (V), м/с',
    'V, м/с', 'RdBu_r',
    'fig_03_depth_time_V.png',
    vmin=-v_lim, vmax=v_lim
)

make_depth_time_plot(
    speed_grid, depths, T_mpl,
    'Скорость течения, м/с',
    'Скорость, м/с', 'viridis',
    'fig_04_depth_time_speed.png',
    vmin=0
)

make_depth_time_plot(
    dir_grid, depths, T_mpl,
    'Направление течения, °',
    'Направление, °', 'hsv',
    'fig_05_depth_time_dir.png',
    vmin=0, vmax=360
)

# =========================================================
# 8. ВРЕМЕННЫЕ РЯДЫ U И V — НЕСКОЛЬКО ГОРИЗОНТОВ
# =========================================================
# Выбираем несколько горизонтов равномерно
n_sel = min(6, len(depths))
sel_idx = np.linspace(0, len(depths) - 1, n_sel, dtype=int)
sel_depths = depths[sel_idx]

colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_sel))

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, (idx, d) in enumerate(zip(sel_idx, sel_depths)):
    u_ts = adcp_Ve.iloc[:, idx].values
    v_ts = adcp_Vn.iloc[:, idx].values
    lbl = f'{d:.1f} м'
    axes[0].plot(times, u_ts, color=colors[i], linewidth=0.9, label=lbl)
    axes[1].plot(times, v_ts, color=colors[i], linewidth=0.9, label=lbl)

for ax, ylabel, title in zip(axes,
                              ['U (зональная), м/с', 'V (меридиональная), м/с'],
                              ['Зональная компонента скорости (U)', 'Меридиональная компонента скорости (V)']):
    ax.grid(True, linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    fmt_time_axis(ax)

axes[1].set_xlabel('Дата/время')
plt.tight_layout()
plt.savefig('fig_06_timeseries_UV.png', dpi=150)
plt.close()

# =========================================================
# 9. УСРЕДНЕНИЕ ПО 30 МИН ДЛЯ СТАТИСТИКИ
# =========================================================
adcp_Ve_30 = adcp_Ve.resample('30min').mean()
adcp_Vn_30 = adcp_Vn.resample('30min').mean()

# =========================================================
# 10. СТАТИСТИЧЕСКИЕ ПАРАМЕТРЫ ТЕЧЕНИЙ
# =========================================================
stats_rows = []

for d in depths:
    row = stats_table(adcp_Ve_30[d], adcp_Vn_30[d], f'{d:.1f} м')
    stats_rows.append(row)

# Среднее по вертикали
Ve_mean_col = adcp_Ve_30.mean(axis=1)
Vn_mean_col = adcp_Vn_30.mean(axis=1)
row_mean = stats_table(Ve_mean_col, Vn_mean_col, 'Среднее по вертикали')
stats_rows.append(row_mean)

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv('adcp_statistics.csv', index=False, encoding='utf-8-sig')
print("\nСтатистические параметры течений:")
print(stats_df.to_string(index=False))

# Визуализация таблицы
fig, ax = plt.subplots(figsize=(18, max(4, len(stats_df) * 0.4 + 1.5)))
ax.axis('off')
col_labels = stats_df.columns.tolist()
tbl = ax.table(
    cellText=stats_df.values,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(len(col_labels))))
ax.set_title('Статистические параметры течений (усреднение 30 мин)', pad=10, fontsize=11)
plt.tight_layout()
plt.savefig('fig_07_statistics_table.png', dpi=150)
plt.close()

# =========================================================
# 11. ГИСТОГРАММЫ СКОРОСТИ И НАПРАВЛЕНИЯ
# =========================================================
# Выбираем горизонты + среднее
hist_depths_idx = list(sel_idx) + [-1]   # -1 → среднее по вертикали
hist_labels = [f'{depths[i]:.1f} м' for i in sel_idx] + ['Среднее по вертикали']

fig_speed, axes_sp = plt.subplots(
    2, len(hist_depths_idx), figsize=(4 * len(hist_depths_idx), 8)
)

for col_i, (d_idx, lbl) in enumerate(zip(hist_depths_idx, hist_labels)):
    if d_idx == -1:
        u_vals = adcp_Ve_30.mean(axis=1).dropna().values
        v_vals = adcp_Vn_30.mean(axis=1).dropna().values
    else:
        u_vals = adcp_Ve_30.iloc[:, d_idx].dropna().values
        v_vals = adcp_Vn_30.iloc[:, d_idx].dropna().values

    speed_vals, dir_vals = compute_speed_direction(u_vals, v_vals)

    ax_sp = axes_sp[0, col_i]
    ax_dir = axes_sp[1, col_i]

    ax_sp.hist(speed_vals, bins=30, color='steelblue', edgecolor='white', linewidth=0.3)
    ax_sp.set_title(lbl, fontsize=9)
    ax_sp.set_xlabel('Скорость, м/с')
    ax_sp.set_ylabel('Частота')
    ax_sp.grid(True, linewidth=0.4)

    ax_dir.hist(dir_vals, bins=36, range=(0, 360), color='darkorange',
                edgecolor='white', linewidth=0.3)
    ax_dir.set_xlabel('Направление, °')
    ax_dir.set_ylabel('Частота')
    ax_dir.set_xticks([0, 90, 180, 270, 360])
    ax_dir.grid(True, linewidth=0.4)

axes_sp[0, 0].set_title(hist_labels[0] + '\n(скорость)', fontsize=9)
axes_sp[1, 0].set_title(hist_labels[0] + '\n(направление)', fontsize=9)
fig_speed.suptitle('Гистограммы скорости (верхний ряд) и направления течений (нижний ряд)',
                   fontsize=11)
plt.tight_layout()
plt.savefig('fig_08_histograms.png', dpi=150)
plt.close()

# Розы ветра / направлений (полярные диаграммы)
fig_rose, axes_rose = plt.subplots(
    1, len(hist_depths_idx),
    subplot_kw={'projection': 'polar'},
    figsize=(4 * len(hist_depths_idx), 5)
)

bin_edges = np.linspace(0, 2 * np.pi, 37)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
width = bin_edges[1] - bin_edges[0]

for col_i, (d_idx, lbl) in enumerate(zip(hist_depths_idx, hist_labels)):
    if d_idx == -1:
        u_vals = adcp_Ve_30.mean(axis=1).dropna().values
        v_vals = adcp_Vn_30.mean(axis=1).dropna().values
    else:
        u_vals = adcp_Ve_30.iloc[:, d_idx].dropna().values
        v_vals = adcp_Vn_30.iloc[:, d_idx].dropna().values

    speed_vals, dir_vals = compute_speed_direction(u_vals, v_vals)
    dir_rad = np.radians(dir_vals)
    counts, _ = np.histogram(dir_rad, bins=bin_edges)
    ax_r = axes_rose[col_i]
    ax_r.bar(bin_centers, counts, width=width, color='teal', alpha=0.8)
    ax_r.set_theta_zero_location('N')
    ax_r.set_theta_direction(-1)
    ax_r.set_title(lbl, va='bottom', fontsize=9)

fig_rose.suptitle('Роза направлений течений', fontsize=11)
plt.tight_layout()
plt.savefig('fig_09_rose_direction.png', dpi=150)
plt.close()

# =========================================================
# 12. СПЕКТРЫ МОЩНОСТИ — ЗОНАЛЬНАЯ КОМПОНЕНТА
# =========================================================
# Проверяем дискретность: берём данные на выбранных горизонтах
# Эксперимент с различными окнами nperseg

nperseg_variants = {
    '256 (узкое)': 256,
    '512': 512,
    '1024': 1024,
    '2048 (широкое)': 2048,
}

fig_spec, axes_spec = plt.subplots(
    1, len(sel_idx), figsize=(4 * len(sel_idx), 5), sharey=False
)
if len(sel_idx) == 1:
    axes_spec = [axes_spec]

for col_i, (idx, d) in enumerate(zip(sel_idx, sel_depths)):
    u_ts = adcp_Ve.iloc[:, idx].dropna().values
    u_ts = u_ts - np.mean(u_ts)

    ax = axes_spec[col_i]
    for lbl, nperseg_val in nperseg_variants.items():
        nperseg_actual = min(nperseg_val, len(u_ts) // 2)
        if nperseg_actual < 32:
            continue
        f_w, Pxx = welch(
            u_ts,
            fs=fs,
            window='hann',
            nperseg=nperseg_actual,
            noverlap=nperseg_actual // 2,
            detrend='linear'
        )
        mask = f_w > 0
        T_min = 1.0 / f_w[mask] / 60.0
        ax.loglog(T_min, Pxx[mask], linewidth=1.0, label=lbl)

    ax.invert_xaxis()
    ax.grid(True, which='both', linewidth=0.4)
    ax.set_xlabel('Период, мин')
    ax.set_ylabel('СПМ, (м/с)²/Гц') if col_i == 0 else None
    ax.set_title(f'{d:.1f} м')
    ax.legend(fontsize=7)

fig_spec.suptitle(
    f'Спектр мощности зональной компоненты U\n'
    f'(шаг: {dt_min:.2f} мин, окно Хэннинга, перекрытие 50%)',
    fontsize=10
)
plt.tight_layout()
plt.savefig('fig_10_power_spectrum_U.png', dpi=150)
plt.close()

# То же для меридиональной компоненты
fig_spec_v, axes_spec_v = plt.subplots(
    1, len(sel_idx), figsize=(4 * len(sel_idx), 5), sharey=False
)
if len(sel_idx) == 1:
    axes_spec_v = [axes_spec_v]

for col_i, (idx, d) in enumerate(zip(sel_idx, sel_depths)):
    v_ts = adcp_Vn.iloc[:, idx].dropna().values
    v_ts = v_ts - np.mean(v_ts)

    ax = axes_spec_v[col_i]
    for lbl, nperseg_val in nperseg_variants.items():
        nperseg_actual = min(nperseg_val, len(v_ts) // 2)
        if nperseg_actual < 32:
            continue
        f_w, Pxx = welch(
            v_ts,
            fs=fs,
            window='hann',
            nperseg=nperseg_actual,
            noverlap=nperseg_actual // 2,
            detrend='linear'
        )
        mask = f_w > 0
        T_min = 1.0 / f_w[mask] / 60.0
        ax.loglog(T_min, Pxx[mask], linewidth=1.0, label=lbl)

    ax.invert_xaxis()
    ax.grid(True, which='both', linewidth=0.4)
    ax.set_xlabel('Период, мин')
    ax.set_ylabel('СПМ, (м/с)²/Гц') if col_i == 0 else None
    ax.set_title(f'{d:.1f} м')
    ax.legend(fontsize=7)

fig_spec_v.suptitle(
    f'Спектр мощности меридиональной компоненты V\n'
    f'(шаг: {dt_min:.2f} мин, окно Хэннинга, перекрытие 50%)',
    fontsize=10
)
plt.tight_layout()
plt.savefig('fig_11_power_spectrum_V.png', dpi=150)
plt.close()

# =========================================================
# 13. ПОЛОСОВАЯ ФИЛЬТРАЦИЯ (5–15 МИН)
# =========================================================
low  = 1.0 / (15 * 60)
high = 1.0 / ( 5 * 60)

b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')

Ve_filt = np.full_like(Ve_grid, np.nan)
Vn_filt = np.full_like(Vn_grid, np.nan)

for col_i in range(Ve_grid.shape[1]):
    u_col = Ve_grid[:, col_i]
    v_col = Vn_grid[:, col_i]

    valid_u = np.isfinite(u_col)
    valid_v = np.isfinite(v_col)

    if valid_u.sum() > 4 * (len(b) - 1):
        u_tmp = np.where(valid_u, u_col, 0.0)
        Ve_filt[:, col_i] = np.where(valid_u, filtfilt(b, a, u_tmp), np.nan)

    if valid_v.sum() > 4 * (len(b) - 1):
        v_tmp = np.where(valid_v, v_col, 0.0)
        Vn_filt[:, col_i] = np.where(valid_v, filtfilt(b, a, v_tmp), np.nan)

u_f_lim = np.nanpercentile(np.abs(Ve_filt), 98)
v_f_lim = np.nanpercentile(np.abs(Vn_filt), 98)

make_depth_time_plot(
    Ve_filt, depths, T_mpl,
    'Отфильтрованная зональная компонента (5–15 мин), м/с',
    'U filtered, м/с', 'RdBu_r',
    'fig_12_depth_time_U_filtered.png',
    vmin=-u_f_lim, vmax=u_f_lim
)
make_depth_time_plot(
    Vn_filt, depths, T_mpl,
    'Отфильтрованная меридиональная компонента (5–15 мин), м/с',
    'V filtered, м/с', 'RdBu_r',
    'fig_13_depth_time_V_filtered.png',
    vmin=-v_f_lim, vmax=v_f_lim
)

# Временные ряды полосовых компонент на выбранных горизонтах
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i, (idx, d) in enumerate(zip(sel_idx, sel_depths)):
    lbl = f'{d:.1f} м'
    axes[0].plot(times, Ve_filt[:, idx], color=colors[i], linewidth=0.9, label=lbl)
    axes[1].plot(times, Vn_filt[:, idx], color=colors[i], linewidth=0.9, label=lbl)

for ax, ylabel, title in zip(
    axes,
    ['U (5–15 мин), м/с', 'V (5–15 мин), м/с'],
    ['Зональная компонента (полосовой фильтр 5–15 мин)',
     'Меридиональная компонента (полосовой фильтр 5–15 мин)']
):
    ax.grid(True, linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    fmt_time_axis(ax)

axes[1].set_xlabel('Дата/время')
plt.tight_layout()
plt.savefig('fig_14_timeseries_UV_filtered.png', dpi=150)
plt.close()

# =========================================================
# 14. ВЕЙВЛЕТ-АНАЛИЗ (ПЕРВЫЙ ГОРИЗОНТ)
# =========================================================
ref_idx = sel_idx[len(sel_idx) // 2]  # средний горизонт
ref_depth = depths[ref_idx]
step = max(1, int(round(60 / dt_sec)))  # прореживаем до ~1 мин

Ve_w = Ve_filt[::step, ref_idx]
dt_w = dt_sec * step

periods = np.linspace(5, 15, 40)
scales = periods * 60.0 / (2.0 * np.pi)

coeffs, _ = pywt.cwt(
    np.nan_to_num(Ve_w), scales, 'morl', sampling_period=dt_w
)

t_w = times[::step]

fig, ax = plt.subplots(figsize=(14, 5))
T_wv, P_wv = np.meshgrid(mdates.date2num(t_w), periods)
pcm = ax.pcolormesh(T_wv, P_wv, np.abs(coeffs), shading='nearest', cmap='jet')
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m\n%H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.colorbar(pcm, ax=ax, label='Амплитуда')
ax.set_xlabel('Дата/время')
ax.set_ylabel('Период, мин')
ax.set_title(f'Вейвлет-анализ (5–15 мин), горизонт {ref_depth:.1f} м')
plt.tight_layout()
plt.savefig('fig_15_wavelet.png', dpi=150)
plt.close()

# =========================================================
# 15. ОЦЕНКА НАПРАВЛЕНИЯ РАСПРОСТРАНЕНИЯ ВОЛН (МЕТОД ХИЛБЕРТА)
# =========================================================
from scipy.signal import hilbert as sp_hilbert

Ve_h_col = Ve_filt[:, ref_idx]
Vn_h_col = Vn_filt[:, ref_idx]

valid = np.isfinite(Ve_h_col) & np.isfinite(Vn_h_col)
Ve_h_clean = np.where(valid, Ve_h_col, 0.0)
Vn_h_clean = np.where(valid, Vn_h_col, 0.0)

analytic_Ve = sp_hilbert(Ve_h_clean)
analytic_Vn = sp_hilbert(Vn_h_clean)

phase_diff = np.angle(analytic_Vn) - np.angle(analytic_Ve)
direction_hilbert = np.degrees(np.arctan2(
    np.sin(phase_diff),
    np.cos(phase_diff)
))
direction_hilbert[~valid] = np.nan

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(times, direction_hilbert, linewidth=0.8, color='purple')
ax.grid(True, linewidth=0.5)
fmt_time_axis(ax)
ax.set_xlabel('Дата/время')
ax.set_ylabel('Фазовый угол, °')
ax.set_title(f'Оценка направления коротких волн (метод Гильберта), горизонт {ref_depth:.1f} м')
plt.tight_layout()
plt.savefig('fig_16_wave_direction_hilbert.png', dpi=150)
plt.close()

# =========================================================
# 16. ПРОГРЕССИВНЫЕ ВЕКТОРНЫЕ ДИАГРАММЫ (PVD)
# =========================================================
# PVD: кумулятивная интеграция (U,V) по времени — условный перенос водной массы.
# Реализация как в Khimchenko et al. (JMSE, 2022):
#   X(t) = ∑ U(t_i) * Δt
#   Y(t) = ∑ V(t_i) * Δt

fig_pvd, axes_pvd = plt.subplots(
    2, (len(sel_idx) + 1) // 2,
    figsize=(5 * ((len(sel_idx) + 1) // 2), 10)
)
axes_pvd_flat = np.array(axes_pvd).flatten()

for i, (idx, d) in enumerate(zip(sel_idx, sel_depths)):
    u_pvd = adcp_Ve.iloc[:, idx].values
    v_pvd = adcp_Vn.iloc[:, idx].values

    # Убираем NaN линейной интерполяцией перед интегрированием
    u_series = pd.Series(u_pvd).interpolate(limit=10).fillna(0).values
    v_series = pd.Series(v_pvd).interpolate(limit=10).fillna(0).values

    X = np.cumsum(u_series) * dt_sec / 1000.0  # км
    Y = np.cumsum(v_series) * dt_sec / 1000.0

    # Цвет по времени
    npts = len(X)
    seg_colors = plt.cm.plasma(np.linspace(0, 1, npts - 1))

    ax_pvd = axes_pvd_flat[i]
    for j in range(npts - 1):
        ax_pvd.plot(X[j:j+2], Y[j:j+2], color=seg_colors[j], linewidth=0.7)

    ax_pvd.plot(X[0], Y[0], 'go', markersize=5, label='Начало')
    ax_pvd.plot(X[-1], Y[-1], 'rs', markersize=5, label='Конец')
    ax_pvd.set_aspect('equal', 'box')
    ax_pvd.grid(True, linewidth=0.4)
    ax_pvd.set_xlabel('X (зональный перенос), км')
    ax_pvd.set_ylabel('Y (меридиональный перенос), км')
    ax_pvd.set_title(f'PVD, горизонт {d:.1f} м')
    ax_pvd.legend(fontsize=7)

# Скрыть лишние subplot-ы
for j in range(len(sel_idx), len(axes_pvd_flat)):
    axes_pvd_flat[j].set_visible(False)

# Цветовая полоска как шкала времени
sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=0, vmax=(npts - 1) * dt_sec / 3600))
sm.set_array([])
fig_pvd.colorbar(sm, ax=axes_pvd_flat[:len(sel_idx)], label='Время от начала, ч',
                 fraction=0.02, pad=0.04)
fig_pvd.suptitle('Прогрессивные векторные диаграммы течений', fontsize=12)
plt.tight_layout()
plt.savefig('fig_17_progressive_vector.png', dpi=150)
plt.close()

print("\nВсе рисунки сохранены:")
figs = [
    'fig_01_brunt_vaisala.png',
    'fig_02_depth_time_U.png',
    'fig_03_depth_time_V.png',
    'fig_04_depth_time_speed.png',
    'fig_05_depth_time_dir.png',
    'fig_06_timeseries_UV.png',
    'fig_07_statistics_table.png',
    'fig_08_histograms.png',
    'fig_09_rose_direction.png',
    'fig_10_power_spectrum_U.png',
    'fig_11_power_spectrum_V.png',
    'fig_12_depth_time_U_filtered.png',
    'fig_13_depth_time_V_filtered.png',
    'fig_14_timeseries_UV_filtered.png',
    'fig_15_wavelet.png',
    'fig_16_wave_direction_hilbert.png',
    'fig_17_progressive_vector.png',
    'adcp_statistics.csv',
]
for f in figs:
    print(f'  {f}')
