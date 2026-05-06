import os
import glob
import re

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import gsw
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d


# =========================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def julian_to_datetime(jd):
    return datetime.fromordinal(int(jd)) + timedelta(days=jd % 1) - timedelta(days=366)


def n_u_reversals(u):
    """Число смен знака компоненты U между соседними отсчётами."""
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u) & (u != 0)]
    if len(u) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(u)) != 0))


def current_stats_table(u, v, speed):
    """Статистические параметры течений по ряду."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.asarray(speed, dtype=float)
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(speed)
    u, v, speed = u[mask], v[mask], speed[mask]
    if len(u) == 0:
        return {}
    return {
        'mean_U':    float(np.mean(u)),
        'mean_V':    float(np.mean(v)),
        'mean_speed':float(np.mean(speed)),
        'max_speed': float(np.max(speed)),
        'min_speed': float(np.min(speed)),
        'std_speed': float(np.std(speed)),
        'U_pos_pct': float(np.sum(u > 0) / len(u) * 100),
        'U_neg_pct': float(np.sum(u < 0) / len(u) * 100),
        'N_rev':     n_u_reversals(u),
    }


def pick_horizons(depths, n=4):
    """Несколько характерных горизонтов: приповерхностный, средние, придонный."""
    d = np.sort(np.unique(np.asarray(depths, dtype=float)))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return []
    if len(d) <= n:
        return list(d)
    idx = np.linspace(0, len(d) - 1, n, dtype=int)
    return list(np.unique(d[idx]))


def nearest_depth(actual_depths, target):
    d = np.asarray(actual_depths, dtype=float)
    if len(d) == 0:
        return None
    return float(d[np.argmin(np.abs(d - target))])


def fmt_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def hovmoller_pcolormesh(ax, df, var, title, cbar_label, cmap='RdBu_r',
                         vmin=None, vmax=None):
    """Сплошная карта «время–глубина» через imshow с билинейной интерполяцией."""
    pt = df.pivot_table(index='Depth', columns='datetime', values=var, aggfunc='mean')
    if pt.empty:
        ax.set_title(title + ' (нет данных)')
        return
    pt = pt.sort_index().sort_index(axis=1)
    pt = pt.interpolate(axis=1, limit_direction='both').interpolate(axis=0, limit_direction='both')
    pt = pt.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)

    depths = np.asarray(pt.index, dtype=float)
    times  = pd.to_datetime(pt.columns)
    Z = gaussian_filter1d(pt.values.astype(float), sigma=0.6, axis=0, mode='nearest')

    if len(depths) > 2:
        fine = np.linspace(depths.min(), depths.max(), len(depths) * 8)
        Z_fine = np.array([np.interp(fine, depths, Z[:, j]) for j in range(Z.shape[1])]).T
        depths, Z = fine, Z_fine

    tnum = mdates.date2num(times.to_pydatetime())
    # extent: [left, right, bottom, top] в координатах осей
    # origin='upper' → строка 0 Z соответствует depths[0] (мелко) и рисуется сверху
    extent = [tnum[0], tnum[-1], depths.max(), depths.min()]

    im = ax.imshow(
        Z,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        interpolation='bilinear',
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    # После imshow с указанным extent ось Y: сверху depths.min(), снизу depths.max()
    # Это уже правильно для океанографии (глубина растёт вниз). invert_yaxis не нужен.
    ax.set_xlabel('Время')
    ax.set_ylabel('Глубина, м')
    ax.set_title(title)
    fmt_time_axis(ax)
    plt.colorbar(im, ax=ax, label=cbar_label)


def progressive_vector_diagram(ax, sub, title):
    """
    Прогрессивная векторная диаграмма:
    X(t) = ΣU·Δt,  Y(t) = ΣV·Δt  (метод Khimchenko et al., JMSE 2022).
    """
    sub = sub.sort_values('datetime').dropna(subset=['U', 'V'])
    if len(sub) < 2:
        ax.text(0.5, 0.5, 'Недостаточно данных',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    t64  = pd.to_datetime(sub['datetime']).values.astype('datetime64[ns]').astype(np.int64) / 1e9
    u    = sub['U'].to_numpy(dtype=float)
    v    = sub['V'].to_numpy(dtype=float)
    dt   = np.empty_like(t64)
    dt[1:] = np.diff(t64)
    dt[0]  = float(np.median(dt[1:]))
    x = np.cumsum(u * dt)
    y = np.cumsum(v * dt)
    ax.plot(x, y, color='tab:blue', lw=1.0, alpha=0.9)
    ax.plot(x[0],  y[0],  'go', ms=6, zorder=4, label='Начало')
    ax.plot(x[-1], y[-1], 'rs', ms=6, zorder=4, label='Конец')
    t_s = pd.to_datetime(sub['datetime'].iloc[0]).strftime('%d.%m %H:%M')
    t_e = pd.to_datetime(sub['datetime'].iloc[-1]).strftime('%d.%m %H:%M')
    ax.text(0.02, 0.98, f'Старт: {t_s}\nКонец: {t_e}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('ΔX, м', fontsize=9)
    ax.set_ylabel('ΔY, м', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(title)


# =========================================================
# 2. ЗАГРУЗКА ДАННЫХ
# =========================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_adcp(base_dir=_SCRIPT_DIR):
    """Читает ADCP_1/2/3.txt из папки скрипта."""
    patterns = sorted(glob.glob(os.path.join(base_dir, "ADCP*.txt")))
    if not patterns:
        raise FileNotFoundError(
            f"Файлы ADCP*.txt не найдены в {base_dir}. "
            "Положите ADCP_1.txt, ADCP_2.txt, ADCP_3.txt рядом со скриптом."
        )
    frames = []
    for path in patterns:
        try:
            df = pd.read_csv(path, sep='\t', encoding='cp1251')
        except Exception:
            df = pd.read_csv(path, sep=r'\s+', encoding='cp1251')
        df["__src"] = path
        frames.append(df)
    adcp_df = pd.concat(frames, ignore_index=True)

    required = {'Ve', 'Vn', 'Depth'}
    missing = required - set(adcp_df.columns)
    if missing:
        raise ValueError(f"В ADCP-файле отсутствуют столбцы: {sorted(missing)}")

    if 'datetime' in adcp_df.columns:
        adcp_df['datetime'] = pd.to_datetime(adcp_df['datetime'], errors='coerce')
    elif 'iTimeDbl' in adcp_df.columns:
        adcp_df['iTimeDbl'] = pd.to_numeric(adcp_df['iTimeDbl'], errors='coerce')

        def _parse_start(filepath):
            m = re.search(r'(\d{6})_(\d{4})', os.path.basename(filepath))
            if not m:
                return None
            d6, t4 = m.groups()
            try:
                return pd.Timestamp(year=2000+int(d6[:2]), month=int(d6[2:4]),
                                    day=int(d6[4:6]), hour=int(t4[:2]), minute=int(t4[2:]))
            except ValueError:
                return None

        adcp_df['datetime'] = pd.NaT
        for src, idx in adcp_df.groupby('__src').groups.items():
            vals = adcp_df.loc[idx, 'iTimeDbl'].to_numpy(dtype=float)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            start = _parse_start(src)
            vmin, vmax = float(np.nanmin(vals[valid])), float(np.nanmax(vals[valid]))
            if start is not None and 0.0 <= vmin and vmax <= 366.0:
                dts = start + pd.to_timedelta(vals, unit='D')
            else:
                dts = pd.to_datetime(vals, unit='D', origin='1899-12-30', errors='coerce')
            adcp_df.loc[idx, 'datetime'] = dts
    elif 'JulianDay' in adcp_df.columns:
        adcp_df['datetime'] = adcp_df['JulianDay'].apply(julian_to_datetime)
    else:
        raise ValueError("Нет столбца времени ('datetime', 'iTimeDbl', 'JulianDay').")

    adcp_df['Depth'] = np.abs(pd.to_numeric(adcp_df['Depth'], errors='coerce'))
    adcp_df['Ve'] = pd.to_numeric(adcp_df['Ve'], errors='coerce')
    adcp_df['Vn'] = pd.to_numeric(adcp_df['Vn'], errors='coerce')
    adcp_df = adcp_df.drop(columns=['__src'], errors='ignore')
    adcp_df = adcp_df.dropna(subset=['datetime', 'Ve', 'Vn', 'Depth'])
    return adcp_df.sort_values('datetime').reset_index(drop=True)


def load_ctd(base_dir=_SCRIPT_DIR):
    """Читает CTD.txt из папки скрипта."""
    candidates = [
        os.path.join(base_dir, 'CTD.txt'),
        os.path.join(base_dir, 'ctd.csv'),
        os.path.join(base_dir, 'CTD.csv'),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        for enc in ('utf-8', 'utf-8-sig', 'cp1251', 'latin1'):
            for sep in ('\t', ',', ';', None):
                try:
                    kw = dict(encoding=enc)
                    if sep is None:
                        kw['sep'] = None
                        kw['engine'] = 'python'
                    else:
                        kw['sep'] = sep
                    df = pd.read_csv(path, **kw)
                    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
                    rn = {}
                    if 'Sal(psu)' in df.columns and 'Sal' not in df.columns:
                        rn['Sal(psu)'] = 'Sal'
                    if 'Temperature' in df.columns and 'Temp' not in df.columns:
                        rn['Temperature'] = 'Temp'
                    if 'Depth' in df.columns and 'Depth(m)' not in df.columns:
                        rn['Depth'] = 'Depth(m)'
                    if rn:
                        df = df.rename(columns=rn)
                    required = {'Depth(m)', 'Temp', 'Sal'}
                    if required.issubset(df.columns):
                        return df
                except Exception:
                    pass
    raise FileNotFoundError(
        f"Файл CTD не найден или не содержит нужных столбцов (Depth(m), Temp, Sal) в {base_dir}"
    )


# =========================================================
# 3. ЧТЕНИЕ
# =========================================================
adcp = load_adcp()
ctd  = load_ctd()

# =========================================================
# 4. ПРОФИЛЬ ЧАСТОТЫ ВЯЙСЯЛЯ–БРЕНТА (CTD)
# =========================================================
lat, lon, g = 44.5, 37.98, 9.81
ctd = ctd.sort_values('Depth(m)').dropna(subset=['Depth(m)', 'Temp', 'Sal']).reset_index(drop=True)
depth_ctd = np.abs(ctd['Depth(m)'].to_numpy(dtype=float))
p_dbar = gsw.p_from_z(-depth_ctd, lat)
SA  = gsw.SA_from_SP(ctd['Sal'].to_numpy(dtype=float), p_dbar, lon, lat)
CT  = gsw.CT_from_t(SA, ctd['Temp'].to_numpy(dtype=float), p_dbar)
rho = gsw.rho(SA, CT, p_dbar)

rho_df = (pd.DataFrame({'d': depth_ctd, 'rho': rho})
            .dropna().groupby('d', as_index=False).mean().sort_values('d'))
d_prof   = rho_df['d'].to_numpy(dtype=float)
rho_prof = rho_df['rho'].to_numpy(dtype=float)
N2  = -g / np.where(rho_prof > 0, rho_prof, np.nan) * np.gradient(rho_prof, d_prof)
N   = np.sqrt(np.clip(N2, 0, None))
N_cph = N * 3600.0 / (2.0 * np.pi)

fig, ax = plt.subplots(figsize=(5, 7))
ax.plot(N_cph, d_prof, linewidth=1.5)
ax.invert_yaxis()
ax.grid(True)
ax.set_xlabel('Частота Вяйсяля–Брента N, цикл/час')
ax.set_ylabel('Глубина, м')
ax.set_title('Профиль частоты Вяйсяля–Брента N(z)')
plt.tight_layout()
plt.savefig('fig_01_brunt_vaisala.png', dpi=150)
plt.close()

# =========================================================
# 5. ПОДГОТОВКА ADCP-ДАННЫХ
# =========================================================
adcp['U']         = adcp['Ve'].astype(float)
adcp['V']         = adcp['Vn'].astype(float)
adcp['speed']     = np.hypot(adcp['U'], adcp['V'])
adcp['direction'] = (np.degrees(np.arctan2(adcp['U'], adcp['V'])) + 360) % 360

# Реальный шаг дискретизации
times_uniq = np.sort(adcp['datetime'].unique())
if len(times_uniq) > 1:
    dt_sec = float(np.median(np.diff(
        times_uniq.astype('datetime64[ns]').astype(np.int64)
    )) / 1e9)
else:
    dt_sec = 60.0
dt_min = dt_sec / 60.0
fs = 1.0 / dt_sec
print(f"Шаг дискретизации ADCP: {dt_sec:.1f} с ({dt_min:.2f} мин),  fs = {fs:.4f} Гц")

# 30-минутное усреднение
adcp_30 = (
    adcp.groupby([pd.Grouper(key='datetime', freq='30min'), 'Depth'])
    [['U', 'V', 'speed', 'direction']].mean().reset_index()
)

adcp_30_mean = (
    adcp_30.groupby('datetime', as_index=False)
    .agg(U=('U', 'mean'), V=('V', 'mean'))
)
adcp_30_mean['speed']     = np.hypot(adcp_30_mean['U'], adcp_30_mean['V'])
adcp_30_mean['direction'] = (np.degrees(np.arctan2(adcp_30_mean['U'], adcp_30_mean['V'])) + 360) % 360
adcp_30_mean['Depth']     = np.nan

depths_avail = np.sort(adcp_30['Depth'].dropna().unique())
horizons = [nearest_depth(depths_avail, h) for h in pick_horizons(depths_avail, n=4)]
horizons = list(dict.fromkeys([h for h in horizons if h is not None]))

# =========================================================
# 6. КАРТЫ ГЛУБИНА–ВРЕМЯ (pcolormesh, сплошные)
# =========================================================
fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
hovmoller_pcolormesh(ax_s, adcp_30, 'speed',
                     'Скорость |V| (время–глубина, усреднение 30 мин)',
                     'м/с', cmap='jet', vmin=0)
hovmoller_pcolormesh(ax_d, adcp_30, 'direction',
                     'Направление течения (время–глубина, усреднение 30 мин)',
                     '°', cmap='hsv', vmin=0, vmax=360)
plt.tight_layout()
plt.savefig('fig_02_depth_time_speed_dir.png', dpi=150)
plt.close()

u_lim = float(np.nanpercentile(np.abs(adcp_30['U'].dropna()), 98))
v_lim = float(np.nanpercentile(np.abs(adcp_30['V'].dropna()), 98))

fig, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
hovmoller_pcolormesh(ax_u, adcp_30, 'U',
                     'Зональная компонента U (восток +, запад −)',
                     'м/с', cmap='RdBu_r', vmin=-u_lim, vmax=u_lim)
hovmoller_pcolormesh(ax_v, adcp_30, 'V',
                     'Меридиональная компонента V (север +, юг −)',
                     'м/с', cmap='RdBu_r', vmin=-v_lim, vmax=v_lim)
plt.tight_layout()
plt.savefig('fig_03_depth_time_UV.png', dpi=150)
plt.close()

# =========================================================
# 7. ВРЕМЕННЫЕ РЯДЫ U, V НА ОТДЕЛЬНЫХ ГОРИЗОНТАХ
# =========================================================
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(horizons)))

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i, d in enumerate(horizons):
    sub = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    lbl = f'{d:.1f} м'
    axes[0].plot(sub['datetime'], sub['U'], color=colors[i], lw=0.9, label=lbl)
    axes[1].plot(sub['datetime'], sub['V'], color=colors[i], lw=0.9, label=lbl)
for ax, ylabel, title in zip(
    axes,
    ['U (зональная), м/с', 'V (меридиональная), м/с'],
    ['Зональная компонента скорости U', 'Меридиональная компонента скорости V']
):
    ax.grid(True, lw=0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=len(horizons), fontsize=8)
    fmt_time_axis(ax)
axes[1].set_xlabel('Дата / Время')
plt.tight_layout()
plt.savefig('fig_04_timeseries_UV.png', dpi=150)
plt.close()

# =========================================================
# 8. СТАТИСТИЧЕСКИЕ ПАРАМЕТРЫ ТЕЧЕНИЙ (30 мин)
# =========================================================
rows = []
for d in horizons:
    w  = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    st = current_stats_table(w['U'].values, w['V'].values, w['speed'].values)
    st['Глубина, м'] = f'{d:.1f}'
    rows.append(st)
st_m = current_stats_table(adcp_30_mean['U'].values,
                            adcp_30_mean['V'].values,
                            adcp_30_mean['speed'].values)
st_m['Глубина, м'] = 'среднее (верт.)'
rows.append(st_m)

stats_df = pd.DataFrame(rows)
stats_df['N_rev'] = pd.to_numeric(stats_df.get('N_rev', 0), errors='coerce').fillna(0).astype(int)
col_order = ['Глубина, м', 'mean_U', 'mean_V', 'mean_speed', 'max_speed',
             'min_speed', 'std_speed', 'U_pos_pct', 'U_neg_pct', 'N_rev']
stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
rename_ru = {
    'mean_U': 'U ср, м/с', 'mean_V': 'V ср, м/с',
    'mean_speed': '|V| ср, м/с', 'max_speed': '|V| max, м/с',
    'min_speed': '|V| min, м/с', 'std_speed': 'std |V|, м/с',
    'U_pos_pct': 'U pos, %', 'U_neg_pct': 'U neg, %', 'N_rev': 'N rev',
}
stats_show = stats_df.rename(columns=rename_ru)
print('\n=== Статистика течений (усреднение 30 мин) ===')
print(stats_show.to_string(index=False))
stats_show.to_csv('adcp_statistics.csv', index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(16, max(3, len(stats_show) * 0.5 + 1.5)))
ax.axis('off')
tbl = ax.table(cellText=stats_show.values, colLabels=stats_show.columns,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(stats_show.columns))))
ax.set_title('Статистические параметры течений (усреднение 30 мин)', fontsize=11, pad=10)
plt.tight_layout()
plt.savefig('fig_05_statistics_table.png', dpi=150)
plt.close()

# =========================================================
# 9. ГИСТОГРАММЫ СКОРОСТИ И НАПРАВЛЕНИЯ
# =========================================================
hist_items = [(d, f'{d:.1f} м') for d in horizons] + [(None, 'Среднее (верт.)')]
ncols = len(hist_items)

fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

for ci, (d, lbl) in enumerate(hist_items):
    if d is None:
        u_h = adcp_30_mean['U'].dropna().values
        v_h = adcp_30_mean['V'].dropna().values
    else:
        sub = adcp_30[np.isclose(adcp_30['Depth'], d)]
        u_h = sub['U'].dropna().values
        v_h = sub['V'].dropna().values
    spd = np.hypot(u_h, v_h)
    dr  = (np.degrees(np.arctan2(u_h, v_h)) + 360) % 360

    axes[0, ci].hist(spd, bins=30, color='steelblue', edgecolor='white', lw=0.3)
    axes[0, ci].set_title(lbl, fontsize=9)
    axes[0, ci].set_xlabel('Скорость, м/с')
    axes[0, ci].set_ylabel('Частота')
    axes[0, ci].grid(True, lw=0.3)

    axes[1, ci].hist(dr, bins=36, range=(0, 360), color='darkorange', edgecolor='white', lw=0.3)
    axes[1, ci].set_xlabel('Направление, °')
    axes[1, ci].set_ylabel('Частота')
    axes[1, ci].set_xticks([0, 90, 180, 270, 360])
    axes[1, ci].grid(True, lw=0.3)

fig.suptitle('Гистограммы: скорость (верх) и направление течений (низ)', fontsize=11)
plt.tight_layout()
plt.savefig('fig_06_histograms.png', dpi=150)
plt.close()

# Роза направлений
fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5),
                          subplot_kw={'projection': 'polar'})
bin_e = np.linspace(0, 2 * np.pi, 37)
bin_c = 0.5 * (bin_e[:-1] + bin_e[1:])
bw    = bin_e[1] - bin_e[0]
for ci, (d, lbl) in enumerate(hist_items):
    if d is None:
        u_h = adcp_30_mean['U'].dropna().values
        v_h = adcp_30_mean['V'].dropna().values
    else:
        sub = adcp_30[np.isclose(adcp_30['Depth'], d)]
        u_h = sub['U'].dropna().values
        v_h = sub['V'].dropna().values
    dr_rad = np.radians((np.degrees(np.arctan2(u_h, v_h)) + 360) % 360)
    cnt, _ = np.histogram(dr_rad, bins=bin_e)
    axes[ci].bar(bin_c, cnt, width=bw, color='teal', alpha=0.85)
    axes[ci].set_theta_zero_location('N')
    axes[ci].set_theta_direction(-1)
    axes[ci].set_title(lbl, va='bottom', fontsize=9)
fig.suptitle('Роза направлений течений', fontsize=11)
plt.tight_layout()
plt.savefig('fig_07_rose_direction.png', dpi=150)
plt.close()

# =========================================================
# 10. СПЕКТР МОЩНОСТИ (WELCH): U и V
# =========================================================
# Спектральный анализ проводится на исходных (не усреднённых) данных.
# Дискретность dt_sec определена выше.
nperseg_variants = {'256 (узкое)': 256, '512': 512, '1024': 1024, '2048 (широкое)': 2048}

for comp, col, fname_base in [('U (зональная)', 'U', 'U'), ('V (меридиональная)', 'V', 'V')]:
    fig, axes_s = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 5), sharey=False)
    if len(horizons) == 1:
        axes_s = [axes_s]
    for ci, d in enumerate(horizons):
        sub = adcp[np.isclose(adcp['Depth'], d)].sort_values('datetime')
        ts  = sub[col].dropna().values
        ts  -= np.mean(ts)
        ax  = axes_s[ci]
        for lbl, nperseg in nperseg_variants.items():
            n = min(nperseg, len(ts) // 2)
            if n < 32:
                continue
            f_w, Pxx = welch(ts, fs=fs, window='hann', nperseg=n,
                             noverlap=n // 2, detrend='linear')
            mask = f_w > 0
            T_min = 1.0 / f_w[mask] / 60.0
            ax.loglog(T_min, Pxx[mask], lw=1.0, label=lbl)
        ax.invert_xaxis()
        ax.grid(True, which='both', lw=0.4)
        ax.set_xlabel('Период, мин')
        if ci == 0:
            ax.set_ylabel('СПМ, (м/с)²/Гц')
        ax.set_title(f'{d:.1f} м')
        ax.legend(fontsize=7)
    fig.suptitle(
        f'Спектр мощности компоненты {comp}\n'
        f'(шаг: {dt_min:.2f} мин, окно Хэннинга, перекрытие 50%)',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f'fig_08_spectrum_{fname_base}.png', dpi=150)
    plt.close()

# =========================================================
# 11. ПРОГРЕССИВНЫЕ ВЕКТОРНЫЕ ДИАГРАММЫ (PVD)
# =========================================================
pvd_items = [(d, f'Глубина {d:.1f} м (30 мин)') for d in horizons] + \
            [(None, 'Среднее по вертикали (30 мин)')]
ncols_pvd = 2
nrows_pvd = (len(pvd_items) + 1) // 2

fig, axes = plt.subplots(nrows_pvd, ncols_pvd,
                          figsize=(9, 5 * nrows_pvd), squeeze=False)
axes_flat = axes.flatten()

for i, (d, title) in enumerate(pvd_items):
    if d is None:
        sub = adcp_30_mean.copy()
    else:
        sub = adcp_30[np.isclose(adcp_30['Depth'], d)].copy()
    progressive_vector_diagram(axes_flat[i], sub, title)

for k in range(len(pvd_items), len(axes_flat)):
    axes_flat[k].axis('off')

fig.suptitle('Прогрессивные векторные диаграммы течений\n(X = ΣU·Δt, Y = ΣV·Δt)',
             fontsize=11)
plt.tight_layout()
plt.savefig('fig_09_progressive_vector.png', dpi=150)
plt.close()

# =========================================================
# ИТОГ
# =========================================================
saved = [
    'fig_01_brunt_vaisala.png',
    'fig_02_depth_time_speed_dir.png',
    'fig_03_depth_time_UV.png',
    'fig_04_timeseries_UV.png',
    'fig_05_statistics_table.png',
    'fig_06_histograms.png',
    'fig_07_rose_direction.png',
    'fig_08_spectrum_U.png',
    'fig_08_spectrum_V.png',
    'fig_09_progressive_vector.png',
    'adcp_statistics.csv',
]
print('\nСохранены файлы:')
for f in saved:
    print(f'  {f}')
