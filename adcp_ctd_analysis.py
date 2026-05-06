import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import gsw
import os
import glob
import re
from scipy.ndimage import gaussian_filter1d


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
    pt = pt.sort_index().sort_index(axis=1)
    pt = pt.ffill(axis=1).bfill(axis=1)
    pt = pt.interpolate(axis=0, limit_direction='both')
    pt = pt.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)
    depths = np.asarray(pt.index, dtype=float)
    times = pd.to_datetime(pt.columns)
    Z = pt.values.astype(float)
    # Более мелкая вертикальная фильтрация: сохраняем тонкие структуры.
    Z = gaussian_filter1d(Z, sigma=0.45, axis=0, mode='nearest')
    if len(depths) > 2:
        fine_depths = np.linspace(depths.min(), depths.max(), len(depths) * 8)
        Z_fine = np.empty((len(fine_depths), Z.shape[1]), dtype=float)
        for j in range(Z.shape[1]):
            Z_fine[:, j] = np.interp(fine_depths, depths, Z[:, j])
        depths = fine_depths
        Z = Z_fine
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
    pcm = ax.imshow(
        Z,
        aspect='auto',
        origin='upper',
        cmap=cmap,
        interpolation='bilinear',
        extent=[tnum[0], tnum[-1], depths.max(), depths.min()],
    )
    ax.invert_yaxis()
    ax.set_xlabel('Время')
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
    ax.plot(x, y, color='tab:blue', lw=1.0, alpha=0.9, zorder=2)
    ax.plot(x[0], y[0], 'o', color='black', ms=5, zorder=4)
    ax.plot(x[-1], y[-1], 'o', color='black', ms=5, zorder=4)
    t_start = pd.to_datetime(sub['datetime'].iloc[0]).strftime('%d.%m %H:%M')
    t_end = pd.to_datetime(sub['datetime'].iloc[-1]).strftime('%d.%m %H:%M')
    ax.text(
        0.02, 0.98,
        f'Старт: {t_start}\nКонец: {t_end}',
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85)
    )
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('ΔX, м', fontsize=9, labelpad=4)
    ax.set_ylabel('ΔY, м', fontsize=9, labelpad=4)
    ax.set_title(title)
    # Текстовые подписи начала/конца добавлены прямо на график.


# =========================================================
# 2. ПУТИ К ФАЙЛАМ
# =========================================================
base_path = r"C:\Документы\ДИПЛОМ\Химченко_данные\adcp_ctd"

adcp_file = os.path.join(base_path, "adcp.csv")
ctd_file = os.path.join(base_path, "ctd.csv")


def load_adcp(path, base_dir):
    def parse_start_datetime_from_filename(filepath):
        name = os.path.basename(filepath)
        m = re.search(r"(\d{6})_(\d{4})", name)
        if not m:
            return None
        d6, t4 = m.groups()
        yy = int(d6[:2])
        mm = int(d6[2:4])
        dd = int(d6[4:6])
        hh = int(t4[:2])
        mi = int(t4[2:4])
        year = 2000 + yy
        try:
            return pd.Timestamp(year=year, month=mm, day=dd, hour=hh, minute=mi)
        except ValueError:
            return None

    adcp_frames = []
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["__source_file"] = path
        adcp_frames.append(df)
    else:
        # Поддержка набора файлов ADCP_1/2/3.txt (таб-разделитель).
        patterns = [
            os.path.join(base_path, "ADCP*.txt"),
            os.path.join(base_dir, "ADCP*.txt"),
        ]
        adcp_txt_files = []
        for p in patterns:
            adcp_txt_files.extend(glob.glob(p))
        adcp_txt_files = sorted(set(adcp_txt_files))
        if not adcp_txt_files:
            raise FileNotFoundError(
                f"Файл ADCP не найден: {path}. Также не найдены ADCP*.txt в {base_path} и {base_dir}"
            )
        for txt in adcp_txt_files:
            df = pd.read_csv(txt, sep=r"\s+")
            df["__source_file"] = txt
            adcp_frames.append(df)
    adcp_df = pd.concat(adcp_frames, ignore_index=True)
    required = {"Ve", "Vn"}
    missing = required - set(adcp_df.columns)
    if missing:
        raise ValueError(f"В ADCP-файле отсутствуют столбцы: {sorted(missing)}")
    if "Depth" not in adcp_df.columns:
        raise ValueError("В ADCP-файле отсутствует столбец 'Depth'.")
    if "datetime" in adcp_df.columns:
        adcp_df["datetime"] = pd.to_datetime(adcp_df["datetime"])
    elif "iTimeDbl" in adcp_df.columns:
        adcp_df["iTimeDbl"] = pd.to_numeric(adcp_df["iTimeDbl"], errors="coerce")
        adcp_df["datetime"] = pd.NaT
        for src_file, idx in adcp_df.groupby("__source_file").groups.items():
            vals = adcp_df.loc[idx, "iTimeDbl"].to_numpy(dtype=float)
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            start_dt = parse_start_datetime_from_filename(src_file)
            vmin = float(np.nanmin(vals[valid]))
            vmax = float(np.nanmax(vals[valid]))
            if start_dt is not None and 0.0 <= vmin and vmax <= 366.0:
                dts = start_dt + pd.to_timedelta(vals, unit="D")
            else:
                # По умолчанию считаем, что это Excel serial date (дни от 1899-12-30).
                dts = pd.to_datetime(vals, unit="D", origin="1899-12-30", errors="coerce")
            adcp_df.loc[idx, "datetime"] = dts
    elif "JulianDay" in adcp_df.columns:
        adcp_df["datetime"] = adcp_df["JulianDay"].apply(julian_to_datetime)
    else:
        raise ValueError("В ADCP-файле нет столбца времени ('datetime', 'iTimeDbl' или 'JulianDay').")
    adcp_df["Depth"] = np.abs(adcp_df["Depth"].astype(float))
    adcp_df = adcp_df.drop(columns=["__source_file"], errors="ignore")
    return adcp_df.sort_values("datetime").reset_index(drop=True)


def load_ctd(path):
    def read_csv_with_fallbacks(csv_path):
        encodings = ("utf-8", "utf-8-sig", "cp1251", "latin1")
        seps = (",", ";", "\t", None)
        last_error = None
        for enc in encodings:
            for sep in seps:
                try:
                    if sep is None:
                        df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
                    else:
                        df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError) as e:
                    last_error = e
        raise UnicodeDecodeError(
            getattr(last_error, "encoding", "unknown"),
            getattr(last_error, "object", b""),
            getattr(last_error, "start", 0),
            getattr(last_error, "end", 0),
            f"Не удалось прочитать {csv_path} ни в одной из кодировок: {encodings}",
        )

    def normalize_columns(ctd_df):
        rename_map = {}
        if "Sal(psu)" in ctd_df.columns and "Sal" not in ctd_df.columns:
            rename_map["Sal(psu)"] = "Sal"
        if "Temperature" in ctd_df.columns and "Temp" not in ctd_df.columns:
            rename_map["Temperature"] = "Temp"
        if "Depth" in ctd_df.columns and "Depth(m)" not in ctd_df.columns:
            rename_map["Depth"] = "Depth(m)"
        if rename_map:
            ctd_df = ctd_df.rename(columns=rename_map)
        return ctd_df

    sources = []
    if os.path.exists(path):
        sources.append(path)
    local_ctd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CTD.txt")
    if os.path.exists(local_ctd) and local_ctd not in sources:
        sources.append(local_ctd)
    if not sources:
        raise FileNotFoundError(f"Файл CTD не найден: {path} и {local_ctd}")

    required = {"Depth(m)", "Temp", "Sal"}
    checked_columns = []
    for src in sources:
        ctd_df = normalize_columns(read_csv_with_fallbacks(src))
        missing = required - set(ctd_df.columns)
        if not missing:
            return ctd_df.copy()
        checked_columns.append((src, list(ctd_df.columns)))
    raise ValueError(f"В CTD-файле отсутствуют столбцы: {sorted(required)}. Проверены: {checked_columns}")


adcp = load_adcp(adcp_file, os.path.dirname(os.path.abspath(__file__)))
ctd = load_ctd(ctd_file)

# CTD: расчет N(z) через TEOS-10
lat, lon, g = 44.5, 37.98, 9.81
ctd = ctd.sort_values("Depth(m)").dropna(subset=["Depth(m)", "Temp", "Sal"]).reset_index(drop=True)
depth_ctd = np.abs(ctd["Depth(m)"].to_numpy(dtype=float))
p_dbar = gsw.p_from_z(-depth_ctd, lat)
SA = gsw.SA_from_SP(
    ctd["Sal"].to_numpy(dtype=float),
    p_dbar,
    lon,
    lat,
)
CT = gsw.CT_from_t(SA, ctd["Temp"].to_numpy(dtype=float), p_dbar)
rho = gsw.rho(SA, CT, p_dbar)
# Для устойчивого d(rho)/dz усредняем профиль по уникальным глубинам.
rho_prof_df = pd.DataFrame({"Depth(m)": depth_ctd, "rho": rho}).dropna()
rho_prof_df = rho_prof_df.groupby("Depth(m)", as_index=False).mean().sort_values("Depth(m)")
if len(rho_prof_df) < 2:
    raise ValueError("Недостаточно уникальных глубин CTD для расчета частоты Вяйсяля–Брента.")
depth_profile = rho_prof_df["Depth(m)"].to_numpy(dtype=float)
rho_profile = rho_prof_df["rho"].to_numpy(dtype=float)
rho0 = np.where(rho_profile <= 0, np.nan, rho_profile)
drho_dz = np.gradient(rho_profile, depth_profile)
N2 = -g / rho0 * drho_dz
N = np.sqrt(np.clip(N2, 0, None))
N_cph = N * 3600.0 / (2.0 * np.pi)

plt.figure(figsize=(5, 7))
plt.plot(N_cph, depth_profile, linewidth=1.5)
plt.gca().invert_yaxis()
plt.grid(True)
plt.xlabel('Частота Вяйсяля–Брента N, цикл/час')
plt.ylabel('Глубина, м')
plt.title('Профиль частоты Вяйсяля–Брента N(z)')
plt.show()
plt.close('all')

# =========================================================
# 6. ADCP: U, V, скорость, направление; усреднение 30 мин
# =========================================================
adcp = adcp.copy()
adcp['datetime'] = pd.to_datetime(adcp['datetime'], errors='coerce')
adcp = adcp.dropna(subset=['datetime'])
adcp['U'] = adcp['Ve'].astype(float)
adcp['V'] = adcp['Vn'].astype(float)
adcp['speed'] = np.hypot(adcp['U'], adcp['V'])
# Направление течения (куда направлен вектор), от севера по часовой, градусы
adcp['direction'] = (np.degrees(np.arctan2(adcp['U'], adcp['V'])) + 360) % 360

low = 1 / (15*60)
high = 1 / (5*60)
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
# 7. Картины распределения по глубине и времени
# =========================================================
# Окно 1: скорость и направление
fig_sd, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
hovmoller_pcolormesh(
    ax_s,
    adcp_30,
    'speed',
    'Скорость течения |V| (время–глубина, 30 мин усреднение)',
    'м/с',
    cmap='jet'
)
if np.isfinite(adcp_30['speed']).any():
    speed_max = float(np.nanmax(adcp_30['speed']))
else:
    speed_max = 1.0
for coll in ax_s.collections:
    coll.set_clim(0.0, speed_max)
hovmoller_pcolormesh(
    ax_d,
    adcp_30,
    'direction',
    'Направление течения (0–360°, время–глубина, 30 мин усреднение)',
    '°',
    cmap='hsv'
)
for coll in ax_d.collections:
    coll.set_clim(0.0, 360.0)
for ax in [ax_s, ax_d]:
    ax.set_ylabel('Глубина, м')
    ax.set_xlabel('Дата')
fig_sd.tight_layout()
plt.show()
plt.close(fig_sd)

# Окно 2: компоненты U и V
fig_uv, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
hovmoller_pcolormesh(
    ax_u,
    adcp_30,
    'U',
    'Компонента U (восток +, запад -, время–глубина, 30 мин)',
    'м/с',
    cmap='RdBu_r'
)
hovmoller_pcolormesh(
    ax_v,
    adcp_30,
    'V',
    'Компонента V (север +, юг -, время–глубина, 30 мин)',
    'м/с',
    cmap='RdBu_r'
)
for ax in [ax_u, ax_v]:
    ax.set_ylabel('Глубина, м')
    ax.set_xlabel('Дата')
fig_uv.tight_layout()
plt.show()
plt.close(fig_uv)

# =========================================================
# 9. Статистические параметры течений (30-мин ряды)
# =========================================================
rows = []
for d in horizons:
    w = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    st = current_stats_table(w['U'].values, w['V'].values, w['speed'].values)
    if 'N_rev' not in st:
        st['N_rev'] = 0
    st['Глубина, м'] = d
    rows.append(st)

st_mean = current_stats_table(
    adcp_30_mean['U'].values, adcp_30_mean['V'].values, adcp_30_mean['speed'].values
)
if 'N_rev' not in st_mean:
    st_mean['N_rev'] = 0
st_mean['Глубина, м'] = 'среднее (верт.)'
rows.append(st_mean)
stats_df = pd.DataFrame(rows)
if 'N_rev' not in stats_df.columns:
    stats_df['N_rev'] = 0
stats_df['N_rev'] = pd.to_numeric(stats_df['N_rev'], errors='coerce').fillna(0).astype(int)
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

# =========================================================
# 10. Прогрессивные векторные диаграммы (PVD)
# =========================================================
n_pvd = len(horizons) + 1
ncols, nrows = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(9, 8), squeeze=False)
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
plt.suptitle('Прогрессивные векторные диаграммы течений (интеграл U·dt, V·dt)')
plt.tight_layout()
plt.show()
plt.close('all')
