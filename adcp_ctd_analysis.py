import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gsw
import os


# =========================================================
# 1. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# =========================================================
def julian_to_datetime(jd):
    return datetime.fromordinal(int(jd)) + timedelta(days=jd % 1) - timedelta(days=366)


def n_u_reversals(u):
    """Число смен знака компоненты u между соседними отсчётами (усреднёнными)."""
    u = np.asarray(u, dtype=float)
    if len(u) < 2:
        return 0
    return int(np.sum(u[:-1] * u[1:] < 0))


def current_stats_table(u, v, speed):
    """Статистика течений по ряду (уже с нужным усреднением)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.asarray(speed, dtype=float)
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(speed)
    u, v, speed = u[mask], v[mask], speed[mask]
    if len(u) == 0:
        return {}
    u_pos = np.sum(u > 0) / len(u) * 100.0
    u_neg = np.sum(u < 0) / len(u) * 100.0
    return {
        'mean_U': np.nanmean(u),
        'mean_V': np.nanmean(v),
        'mean_speed': np.nanmean(speed),
        'max_speed': np.nanmax(speed),
        'min_speed': np.nanmin(speed),
        'std_speed': np.nanstd(speed),
        'U_pos_pct': u_pos,
        'U_neg_pct': u_neg,
        'N_rev': n_u_reversals(u),
    }


def pick_horizons(depths, n=3):
    """Несколько характерных горизонтов: у поверхности, середина, у дна."""
    d = np.sort(np.unique(np.asarray(depths, dtype=float)))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return []
    if len(d) <= n:
        return list(d)
    idx = np.linspace(0, len(d) - 1, n, dtype=int)
    return list(np.unique(d[idx]))


def nearest_depth(actual_depths, target):
    """Ближайший доступный горизонт к заданной глубине."""
    d = np.asarray(actual_depths, dtype=float)
    if len(d) == 0:
        return None
    return float(d[np.argmin(np.abs(d - target))])


def hovmoller_pcolormesh(ax, df, var, title, cbar_label, cmap='RdBu_r'):
    """Карта «глубина — время» для скалярного поля."""
    pt = df.pivot_table(index='Depth', columns='datetime', values=var, aggfunc='mean')
    if pt.empty:
        ax.set_title(title + ' (нет данных)')
        return
    depths = np.asarray(pt.index, dtype=float)
    times = pd.to_datetime(pt.columns)
    Z = pt.values.astype(float)
    tnum = mdates.date2num(times.to_pydatetime())
    if len(tnum) > 1:
        dt = np.median(np.diff(tnum))
        t_edges = np.concatenate([[tnum[0] - dt / 2], (tnum[:-1] + tnum[1:]) / 2, [tnum[-1] + dt / 2]])
    else:
        dt = 1.0 / 24.0
        t_edges = np.array([tnum[0] - dt / 2, tnum[0] + dt / 2])
    if len(depths) > 1:
        dz = np.median(np.diff(np.sort(depths)))
        z_edges = np.concatenate([[depths.min() - dz / 2], (depths[:-1] + depths[1:]) / 2, [depths.max() + dz / 2]])
    else:
        dz = 1.0
        z_edges = np.array([depths[0] - dz / 2, depths[0] + dz / 2])
    pcm = ax.pcolormesh(t_edges, z_edges, Z, shading='flat', cmap=cmap)
    ax.invert_yaxis()
    ax.set_ylabel('Глубина, м')
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax, label=cbar_label)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')


def progressive_vector_diagram(ax, sub, title):
    """
    Прогрессивная векторная диаграмма: траектория смещения (интеграл U·dt, V·dt).
    Цвет точек — порядок времени (как в классических PVD для приливных течений).
    """
    sub = sub.sort_values('datetime').dropna(subset=['U', 'V'])
    if len(sub) < 2:
        ax.text(0.5, 0.5, 'Недостаточно точек', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    t = pd.to_datetime(sub['datetime']).values
    u = sub['U'].to_numpy(dtype=float)
    v = sub['V'].to_numpy(dtype=float)
    t64 = t.astype('datetime64[ns]').astype(np.int64) / 1e9
    dt = np.zeros(len(t64), dtype=float)
    if len(t64) > 1:
        dt[1:] = np.diff(t64)
        dt[0] = float(np.median(dt[1:]))
    else:
        dt[0] = 1800.0
    x = np.cumsum(u * dt)
    y = np.cumsum(v * dt)
    sc = ax.scatter(x, y, c=np.arange(len(x)), cmap='viridis', s=12, zorder=2)
    ax.plot(x, y, color='gray', lw=0.6, alpha=0.7, zorder=1)
    ax.plot(x[0], y[0], 'go', ms=7, label='начало', zorder=4)
    ax.plot(x[-1], y[-1], 'rs', ms=7, label='конец', zorder=4)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('Смещение на восток, м (∫U dt)')
    ax.set_ylabel('Смещение на север, м (∫V dt)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    plt.colorbar(sc, ax=ax, label='Индекс времени')


# =========================================================
# 2. ПУТИ К ФАЙЛАМ
# =========================================================
base_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\adcp_ctd"

adcp_files = [
    os.path.join(base_path, "230610_0000-20_2359.txt"),
    os.path.join(base_path, "230601_0000-10_2359.txt"),
    os.path.join(base_path, "230621_0000-30_2359.txt")
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

    df = df.dropna(subset=['Ve', 'Vn'])
    df['datetime'] = df['iTimeDbl'].apply(julian_to_datetime)

    adcp_list.append(df)

adcp = pd.concat(adcp_list).sort_values('datetime').reset_index(drop=True)

# =========================================================
# 4. ЧТЕНИЕ CTD
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
# 5. ЧАСТОТА ВЯЙСЯЛЯ–БРЕНТА (CTD)
# =========================================================
ctd = ctd.sort_values('Depth(m)').reset_index(drop=True)
ctd = ctd.loc[ctd['Depth(m)'].diff().fillna(1) != 0]

# psia → dbar
p = ctd['Depth(m)'].values  # грубое, но устойчивое приближение

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

plt.figure(figsize=(5, 7))
plt.plot(N, ctd['Depth(m)'], linewidth=1.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xlabel('Частота Вяйсяля–Брента N, рад/с')
plt.ylabel('Глубина, м')
plt.title('Вертикальный профиль частоты Вяйсяля–Брента по данным CTD')
plt.show()
plt.close('all')

# =========================================================
# 6. ADCP: U, V, скорость, направление; усреднение 30 мин
# =========================================================
adcp = adcp.copy()
adcp['U'] = adcp['Ve'].astype(float)
adcp['V'] = adcp['Vn'].astype(float)
adcp['speed'] = np.hypot(adcp['U'], adcp['V'])
# Направление течения (куда направлен вектор), от севера по часовой, градусы
adcp['direction'] = (np.degrees(np.arctan2(adcp['U'], adcp['V'])) + 360) % 360

adcp_z = adcp.dropna(subset=['Depth'])
adcp_30 = (
    adcp_z.groupby([pd.Grouper(key='datetime', freq='30min'), 'Depth'])[
        ['U', 'V', 'speed', 'direction']
    ]
    .mean()
    .reset_index()
)

# «Среднее по вертикали»: векторное усреднение U, V по горизонтам на каждый момент
adcp_30_mean = (
    adcp_30.groupby('datetime', as_index=False)
    .agg(U=('U', 'mean'), V=('V', 'mean'))
)
adcp_30_mean['speed'] = np.hypot(adcp_30_mean['U'], adcp_30_mean['V'])
adcp_30_mean['direction'] = (
    np.degrees(np.arctan2(adcp_30_mean['U'], adcp_30_mean['V'])) + 360
) % 360
adcp_30_mean['Depth'] = np.nan

depths_avail = np.sort(adcp_30['Depth'].dropna().unique())
horizons = pick_horizons(depths_avail, n=3)
# привязка к фактическим бинам ADCP
horizons = [nearest_depth(depths_avail, h) for h in horizons]
horizons = list(dict.fromkeys([h for h in horizons if h is not None]))

# =========================================================
# 7. Временная изменчивость: а — скорости; б — направления
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for d in horizons:
    w = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    if w.empty:
        continue
    ax1.plot(w['datetime'], w['speed'], lw=0.9, label=f'|V|, {d:g} м')
    ax2.plot(w['datetime'], w['direction'], lw=0.9, label=f'{d:g} м')
wm = adcp_30_mean.sort_values('datetime')
ax1.plot(wm['datetime'], wm['speed'], color='k', lw=1.2, ls='--', label='Среднее по вертикали')
ax2.plot(wm['datetime'], wm['direction'], color='k', lw=1.2, ls='--', label='Среднее по вертикали')
ax1.set_ylabel('Скорость |V|, м/с')
ax1.set_title('а) Временная изменчивость скорости течения (ADCP, шаг 30 мин)')
ax1.grid(True, alpha=0.4)
ax1.legend(loc='upper left', fontsize=8, ncol=2)
ax2.set_ylabel('Направление, ° (от севера по ч. стр.)')
ax2.set_title('б) Временная изменчивость направления течения')
ax2.set_ylim(0, 360)
ax2.grid(True, alpha=0.4)
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')
plt.xlabel('Время')
plt.tight_layout()
plt.show()
plt.close('all')

# =========================================================
# 8. Распределение по глубине и времени (U, V, |V|, направление)
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
hovmoller_pcolormesh(axes[0, 0], adcp_30, 'U', 'U (на восток), м/с', 'м/с')
hovmoller_pcolormesh(axes[0, 1], adcp_30, 'V', 'V (на север), м/с', 'м/с')
hovmoller_pcolormesh(axes[1, 0], adcp_30, 'speed', 'Модуль скорости |V|, м/с', 'м/с', cmap='viridis')
# Направление: циклическая шкала
hovmoller_pcolormesh(axes[1, 1], adcp_30, 'direction', 'Направление течения, °', '°', cmap='twilight')
plt.tight_layout()
plt.show()
plt.close('all')

# =========================================================
# 9. Гистограммы скорости и направления (горизонты + среднее по вертикали)
# =========================================================
n_hist = max(1, len(horizons) + 1)
fig_s, axs_s = plt.subplots(1, n_hist, figsize=(3.2 * n_hist, 3.8), squeeze=False)
fig_d, axs_d = plt.subplots(
    1, n_hist, figsize=(3.2 * n_hist, 3.8), subplot_kw={'projection': 'polar'}, squeeze=False
)
axs_s, axs_d = axs_s[0], axs_d[0]

wm = adcp_30_mean.dropna(subset=['speed', 'direction'])

for j, d in enumerate(horizons):
    w = adcp_30[np.isclose(adcp_30['Depth'], d)]
    spd = w['speed'].dropna().values
    direc = np.radians(w['direction'].dropna().values)
    axs_s[j].hist(spd, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    axs_s[j].set_title(f'|V|, {d:g} м')
    axs_s[j].set_xlabel('|V|, м/с')
    axs_s[j].set_ylabel('N')
    if len(direc):
        axs_d[j].hist(direc, bins=24, color='coral', edgecolor='white', alpha=0.85)
    axs_d[j].set_theta_zero_location('N')
    axs_d[j].set_theta_direction(-1)
    axs_d[j].set_title(f'напр., {d:g} м', y=1.12)

j_mean = len(horizons)
axs_s[j_mean].hist(wm['speed'].values, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axs_s[j_mean].set_title('Среднее по вертикали, |V|')
axs_s[j_mean].set_xlabel('|V|, м/с')
if len(wm):
    axs_d[j_mean].hist(np.radians(wm['direction'].values), bins=24, color='coral', edgecolor='white', alpha=0.85)
axs_d[j_mean].set_theta_zero_location('N')
axs_d[j_mean].set_theta_direction(-1)
axs_d[j_mean].set_title('Среднее по вертикали', y=1.12)

fig_s.suptitle('Гистограммы модуля скорости |V| (шаг 30 мин)', y=1.02)
fig_s.tight_layout()
plt.show()
plt.close(fig_s)

fig_d.suptitle('Гистограммы направления (полярные, 0° = север, по ч. стр.), шаг 30 мин', y=1.14)
fig_d.tight_layout()
plt.show()
plt.close(fig_d)

# Гистограммы U и V на тех же горизонтах
fig_uv, ax_uv = plt.subplots(2, n_hist, figsize=(3.2 * n_hist, 5.5), squeeze=False)
for j, d in enumerate(horizons):
    w = adcp_30[np.isclose(adcp_30['Depth'], d)]
    ax_uv[0, j].hist(w['U'].dropna(), bins=30, color='seagreen', edgecolor='white', alpha=0.85)
    ax_uv[0, j].set_title(f'U, {d:g} м')
    ax_uv[0, j].set_xlabel('U, м/с')
    ax_uv[1, j].hist(w['V'].dropna(), bins=30, color='darkorange', edgecolor='white', alpha=0.85)
    ax_uv[1, j].set_title(f'V, {d:g} м')
    ax_uv[1, j].set_xlabel('V, м/с')
ax_uv[0, j_mean].hist(wm['U'].values, bins=30, color='seagreen', edgecolor='white', alpha=0.85)
ax_uv[0, j_mean].set_title('Среднее по вертикали, U')
ax_uv[0, j_mean].set_xlabel('U, м/с')
ax_uv[1, j_mean].hist(wm['V'].values, bins=30, color='darkorange', edgecolor='white', alpha=0.85)
ax_uv[1, j_mean].set_title('Среднее по вертикали, V')
ax_uv[1, j_mean].set_xlabel('V, м/с')
fig_uv.suptitle('Гистограммы компонент U и V (шаг 30 мин)', y=1.01)
fig_uv.tight_layout()
plt.show()
plt.close(fig_uv)

# =========================================================
# 10. Статистические параметры течений (30-мин ряды)
# =========================================================
rows = []
for d in horizons:
    w = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    st = current_stats_table(w['U'].values, w['V'].values, w['speed'].values)
    st['Глубина, м'] = d
    rows.append(st)

st_mean = current_stats_table(
    adcp_30_mean['U'].values, adcp_30_mean['V'].values, adcp_30_mean['speed'].values
)
st_mean['Глубина, м'] = 'среднее (верт.)'
rows.append(st_mean)

stats_df = pd.DataFrame(rows)
col_order = [
    'Глубина, м',
    'mean_U',
    'mean_V',
    'mean_speed',
    'max_speed',
    'min_speed',
    'std_speed',
    'U_pos_pct',
    'U_neg_pct',
    'N_rev',
]
stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
rename_ru = {
    'mean_U': 'U ср, м/с',
    'mean_V': 'V ср, м/с',
    'mean_speed': '|V| ср, м/с',
    'max_speed': '|V| max, м/с',
    'min_speed': '|V| min, м/с',
    'std_speed': 'std |V|, м/с',
    'U_pos_pct': 'U pos, %',
    'U_neg_pct': 'U neg, %',
    'N_rev': 'N rev',
}
stats_show = stats_df.rename(columns=rename_ru)
print('\n=== Статистика течений (усреднение 30 мин) ===')
print(stats_show.to_string(index=False))
plt.figure(figsize=(12, 0.45 * max(3, len(stats_show) + 2)))
plt.axis('off')
cell_text = []
for _, row in stats_show.iterrows():
    cells = []
    for col, v in zip(stats_show.columns, row):
        if isinstance(v, str):
            cells.append(v)
        elif isinstance(v, (float, np.floating)) and pd.notna(v):
            cells.append(f'{v:.4g}')
        else:
            cells.append('')
    cell_text.append(cells)
tbl = plt.table(
    cellText=cell_text,
    colLabels=stats_show.columns.tolist(),
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.4)
plt.title('Статистические параметры течений (30 мин): среднее U,V,|V|; max/min/std |V|; '
          'U pos/neg %; N rev — число смен знака U')
plt.show()
plt.close('all')

# =========================================================
# 11. Прогрессивные векторные диаграммы (PVD)
# =========================================================
n_pvd = len(horizons) + 1
ncols = min(3, n_pvd)
nrows = int(np.ceil(n_pvd / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
axes = np.atleast_2d(axes)
idx = 0
for d in horizons:
    r, c = divmod(idx, ncols)
    sub = adcp_30[np.isclose(adcp_30['Depth'], d)]
    progressive_vector_diagram(axes[r, c], sub, f'PVD, глубина {d:g} м (30 мин)')
    idx += 1
r, c = divmod(idx, ncols)
sub_mean = adcp_30_mean.copy()
sub_mean['Depth'] = np.nan
progressive_vector_diagram(axes[r, c], sub_mean, 'PVD, среднее по вертикали (30 мин)')
idx += 1
for k in range(idx, nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].axis('off')
plt.suptitle('Прогрессивные векторные диаграммы течений (интеграл U·dt, V·dt; см. Khimchenko et al., JMSE 2024)')
plt.tight_layout()
plt.show()
plt.close('all')
