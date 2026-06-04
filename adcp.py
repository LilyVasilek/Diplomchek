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

from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend
from matplotlib.transforms import blended_transform_factory


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


def overlay_kde_on_hist(ax, values, bins, *, hist_range=None, color='crimson', lw=1.8):
    """KDE поверх гистограммы частот; оси и столбцы hist не меняются."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.ptp(x) <= 0:
        return False
    counts, edges = np.histogram(x, bins=bins, range=hist_range)
    bw = float(edges[1] - edges[0])
    lo, hi = float(edges[0]), float(edges[-1])
    x_grid = np.linspace(lo, hi, 256)
    density = stats.gaussian_kde(x)(x_grid)
    ax.plot(
        x_grid,
        density * x.size * bw,
        color=color,
        lw=lw,
        zorder=5,
        label='KDE',
    )
    return True


def annotate_period_grid_inside(ax, hours=(1, 2, 3, 4, 6, 8, 12, 24, 48), y_frac=0.94):
    """Вертикальные линии сетки по периоду с подписями внутри поля (в часах)."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for h in hours:
        t_min = float(h) * 60.0
        ax.axvline(t_min, color='0.55', lw=0.55, alpha=0.65, zorder=0)
        lbl = f"{int(h)} ч" if h == int(h) else f"{h:g} ч"
        ax.text(
            t_min, y_frac, lbl,
            transform=trans, ha='center', va='top', fontsize=7,
            color='0.35', clip_on=True,
        )


# До этой даты на общих картах не рисуем: по факту регистрация ~20 м (см. отдельный рисунок).
ADCP_FULL_PROFILE_FROM = pd.Timestamp('2023-06-09 00:00:00')
ADCP_PARTIAL_BOTTOM_DEPTH_M = 19.79  # глубина нижней ячейки, м

# Окно для статистики течений и рядов U, V (полный вертикальный охват, ADCP_3).
ADCP_ANALYSIS_START = pd.Timestamp('2023-06-22 00:00:00')
ADCP_ANALYSIS_END = pd.Timestamp('2023-06-29 23:59:59')

ADCP_JUNE_WINDOW_NOTE = """
Окно 22–29 июня 2023 для статистики ADCP и рядов U, V

Почему используем именно этот интервал:
1. Полный вертикальный охват (все ячейки глубины ~1,8–19,8 м) — после 08.06.2023;
   ранний период (только ~20 м) на общие карты и сводную статистику не включаем.
2. Непрерывная запись файла ADCP_3.txt (21.06 00:00 — 28.06 11:51) без смены конфигурации;
   22–29.06 — стабильный фрагмент внутри этой серии.
3. Согласование по времени с анализом внутренних волн на термокосе (станция 4, июнь 2023),
   чтобы описывать фоновые течения в те же сутки, что и колебания изотерм.
4. Исключение краевых дней всей кампании (01–10.06 — неполный столб; после 28.06 — обрезка записи),
   чтобы статистика не смешивала разные режимы регистрации.

Усреднение: 30 мин. Единицы скорости: см/с.
""".strip()


def split_adcp_30_by_depth_coverage(df, full_from=None, bottom_depth=None):
    """
    full — период с полным вертикальным столбом; partial — ранний период, только нижняя ячейка.
    """
    if full_from is None:
        full_from = ADCP_FULL_PROFILE_FROM
    if bottom_depth is None:
        bottom_depth = ADCP_PARTIAL_BOTTOM_DEPTH_M

    full = df[df['datetime'] >= full_from].copy()
    partial_all = df[df['datetime'] < full_from].copy()
    partial = partial_all[
        np.isclose(partial_all['Depth'], bottom_depth, atol=0.15)
    ].copy()
    return full, partial, full_from, bottom_depth


def plot_adcp_partial_single_depth(partial_df, depth_m, out_path):
    """Отдельный рисунок для раннего периода: |V| и направление на одной глубине."""
    sub = partial_df.sort_values('datetime')
    if sub.empty:
        print(f'  Нет данных для отдельного рисунка (глубина ~{depth_m:.1f} м, до {ADCP_FULL_PROFILE_FROM:%d.%m.%Y}).')
        return

    t0, t1 = sub['datetime'].min(), sub['datetime'].max()
    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax_s.plot(sub['datetime'], sub['speed'], color='steelblue', lw=0.9)
    ax_s.set_ylabel('Скорость |V|, см/с')
    ax_s.set_title(f'Скорость |V| на глубине ~{depth_m:.1f} м (до {ADCP_FULL_PROFILE_FROM:%d.%m.%Y})')
    ax_s.grid(True, alpha=0.3)

    ax_d.plot(sub['datetime'], sub['direction'], color='darkgreen', lw=0.9)
    ax_d.set_ylabel('Направление, °')
    ax_d.set_xlabel('Время')
    ax_d.set_ylim(0, 360)
    ax_d.set_title(f'Направление течения на глубине ~{depth_m:.1f} м')
    ax_d.grid(True, alpha=0.3)
    fmt_time_axis(ax_d)

    fig.suptitle(
        f'ADCP: неполный вертикальный охват ({t0:%d.%m.%Y %H:%M} — {t1:%d.%m.%Y %H:%M}), '
        f'одна ячейка ~{depth_m:.1f} м',
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Отдельный рисунок (неполный охват): {out_path}')


def filter_adcp_period(df, t_start=None, t_end=None):
    t_start = t_start or ADCP_ANALYSIS_START
    t_end = t_end or ADCP_ANALYSIS_END
    return df[(df['datetime'] >= t_start) & (df['datetime'] <= t_end)].copy()


def vertical_mean_adcp(df):
    """Среднее по глубине на каждый момент времени (30 мин)."""
    m = df.groupby('datetime', as_index=False).agg(U=('U', 'mean'), V=('V', 'mean'))
    m['speed'] = np.hypot(m['U'], m['V'])
    m['direction'] = (np.degrees(np.arctan2(m['U'], m['V'])) + 360) % 360
    m['Depth'] = np.nan
    return m


def build_current_stats_rows(df, horizons, mean_df=None):
    rows = []
    for d in horizons:
        w = df[np.isclose(df['Depth'], d)].sort_values('datetime')
        st = current_stats_table(w['U'].values, w['V'].values, w['speed'].values)
        st['Глубина, м'] = f'{d:.1f}'
        rows.append(st)
    if mean_df is not None and len(mean_df):
        st_m = current_stats_table(
            mean_df['U'].values, mean_df['V'].values, mean_df['speed'].values
        )
        st_m['Глубина, м'] = 'среднее (верт.)'
        rows.append(st_m)
    return rows


def format_stats_dataframe(rows):
    stats_df = pd.DataFrame(rows)
    stats_df['N_rev'] = pd.to_numeric(stats_df.get('N_rev', 0), errors='coerce').fillna(0).astype(int)
    col_order = [
        'Глубина, м', 'mean_U', 'mean_V', 'mean_speed', 'max_speed',
        'min_speed', 'std_speed', 'U_pos_pct', 'U_neg_pct', 'N_rev',
    ]
    stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
    rename_ru = {
        'mean_U': 'U ср, см/с', 'mean_V': 'V ср, см/с',
        'mean_speed': '|V| ср, см/с', 'max_speed': '|V| max, см/с',
        'min_speed': '|V| min, см/с', 'std_speed': 'std |V|, см/с',
        'U_pos_pct': 'U pos, %', 'U_neg_pct': 'U neg, %', 'N_rev': 'N rev',
    }
    return stats_df.rename(columns=rename_ru)


def save_stats_table_figure(stats_show, csv_path, fig_path, title):
    print(f'\n=== {title} ===')
    print(stats_show.to_string(index=False))
    stats_show.to_csv(csv_path, index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(16, max(3, len(stats_show) * 0.5 + 1.5)))
    ax.axis('off')
    tbl = ax.table(
        cellText=stats_show.values,
        colLabels=stats_show.columns,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(stats_show.columns))))
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_timeseries_uv_horizons(df, horizons, colors, out_path, suptitle, title_u, title_v):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, d in enumerate(horizons):
        sub = df[np.isclose(df['Depth'], d)].sort_values('datetime')
        if len(sub) < 2:
            continue
        sub = sub.set_index('datetime').reindex(
            pd.date_range(sub['datetime'].min(), sub['datetime'].max(), freq='30min')
        ).reset_index().rename(columns={'index': 'datetime'})
        lbl = f'{d:.1f} м'
        axes[0].plot(sub['datetime'], sub['U'], color=colors[i], lw=0.9, label=lbl)
        axes[1].plot(sub['datetime'], sub['V'], color=colors[i], lw=0.9, label=lbl)
    for ax, ylabel, title in zip(
        axes,
        ['U (зональная), см/с', 'V (меридиональная), см/с'],
        [title_u, title_v],
    ):
        ax.grid(True, lw=0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right', ncol=len(horizons), fontsize=8)
        fmt_time_axis(ax)
    axes[1].set_xlabel('Дата / Время')
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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


def progressive_vector_diagram(ax, sub, title, max_days=7):
    """
    Прогрессивная векторная диаграмма:
    X(t) = ΣU·Δt,  Y(t) = ΣV·Δt  (метод Khimchenko et al., JMSE 2022).
    U, V — см/с; Δt — с; накопленное смещение в см, на графике — в км (1 км = 10⁵ см).

    max_days: длина участка (сутки), считая от конца ряда. Рекомендуется ограничивать:
    ряд U,V — эйлеровы скорости в одной точке; длинная интеграция не описывает
    реальное перемещение водной массы на масштабе бассейна. None — весь ряд
    (только для справки, не для интерпретации как перенос на сотни км).
    """
    sub = sub.sort_values('datetime').dropna(subset=['U', 'V'])
    if len(sub) > 0 and max_days is not None:
        cutoff = sub['datetime'].max() - pd.Timedelta(days=max_days)
        sub = sub[sub['datetime'] >= cutoff]
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
    x_cm = np.cumsum(u * dt)
    y_cm = np.cumsum(v * dt)
    # На осях — км (иначе у «коротких» траекторий Matplotlib даёт обычные числа,
    # у длинных — множитель 1e6; единицы те же см, подписи разъезжаются).
    km_per_cm = 1.0 / 1e5
    x = x_cm * km_per_cm
    y = y_cm * km_per_cm
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
    ax.set_xlabel('ΔX, км', fontsize=9)
    ax.set_ylabel('ΔY, км', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title(title)


# =========================================================
# 2. ЗАГРУЗКА ДАННЫХ
# =========================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_adcp_file(path):
    """Чтение одного ADCP*.txt с перебором кодировок и разделителей."""
    encodings = ('utf-8-sig', 'utf-8', 'cp1251', 'cp1252', 'latin1')
    separators = ('\t', r'\s+')
    errors = []
    for enc in encodings:
        for sep in separators:
            try:
                kw = dict(sep=sep, encoding=enc)
                if sep != '\t':
                    kw['engine'] = 'python'
                return pd.read_csv(path, **kw)
            except UnicodeDecodeError as exc:
                errors.append(f'{enc}/{sep!r}: {exc}')
                break
            except Exception as exc:
                errors.append(f'{enc}/{sep!r}: {exc}')
    raise ValueError(
        f'Не удалось прочитать {path}. Попытки:\n  ' + '\n  '.join(errors[-8:])
    )


def load_adcp(base_dir=_SCRIPT_DIR):
    """Читает ADCP_1/2/3.txt из папки скрипта. Ve, Vn используются как см/с."""
    patterns = sorted(
        p for p in glob.glob(os.path.join(base_dir, 'ADCP*.txt'))
        if 'note' not in os.path.basename(p).lower()
    )
    if not patterns:
        raise FileNotFoundError(
            f"Файлы ADCP*.txt не найдены в {base_dir}. "
            "Положите ADCP_1.txt, ADCP_2.txt, ADCP_3.txt рядом со скриптом."
        )
    frames = []
    for path in patterns:
        df = _read_adcp_file(path)
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


# =========================================================
# 3. ЧТЕНИЕ
# =========================================================
adcp = load_adcp()

# =========================================================
# 5. ПОДГОТОВКА ADCP-ДАННЫХ (Ve, Vn → U, V в см/с)
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
horizons = [nearest_depth(depths_avail, h) for h in pick_horizons(depths_avail, n=3)]
horizons = list(dict.fromkeys([h for h in horizons if h is not None]))

# =========================================================
# 6. КАРТЫ ГЛУБИНА–ВРЕМЯ (полный столб; ранний период — отдельно)
# =========================================================
adcp_30_full, adcp_30_partial, adcp_full_from, adcp_bottom_m = split_adcp_30_by_depth_coverage(
    adcp_30
)
print(
    f'\nADCP: общие карты — с {adcp_full_from:%d.%m.%Y %H:%M} '
    f'({len(adcp_30_full)} строк, {adcp_30_full["Depth"].nunique()} глубин); '
    f'отдельный рисунок до этой даты — ~{adcp_bottom_m:.1f} м '
    f'({len(adcp_30_partial)} строк).'
)

if adcp_30_full.empty:
    print('Предупреждение: нет данных ADCP с полным вертикальным охватом для общих карт.')
else:
    t0f, t1f = adcp_30_full['datetime'].min(), adcp_30_full['datetime'].max()
    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
    hovmoller_pcolormesh(
        ax_s,
        adcp_30_full,
        'speed',
        f'Скорость |V|, см/с (с {adcp_full_from:%d.%m.%Y})',
        'см/с',
        cmap='jet',
        vmin=0,
    )
    hovmoller_pcolormesh(
        ax_d,
        adcp_30_full,
        'direction',
        f'Направление течения (с {adcp_full_from:%d.%m.%Y})',
        '°',
        cmap='hsv',
        vmin=0,
        vmax=360,
    )
    fig.suptitle(
        f'Временная изменчивость |V| (а) и направления (б), полный столб\n'
        f'{t0f:%d.%m.%Y %H:%M} — {t1f:%d.%m.%Y %H:%M}, усреднение 30 мин',
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig('fig_02_depth_time_speed_dir.png', dpi=150, bbox_inches='tight')
    plt.close()

    u_lim = float(np.nanpercentile(np.abs(adcp_30_full['U'].dropna()), 98))
    v_lim = float(np.nanpercentile(np.abs(adcp_30_full['V'].dropna()), 98))

    fig, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
    hovmoller_pcolormesh(
        ax_u,
        adcp_30_full,
        'U',
        f'Зональная U, см/с (с {adcp_full_from:%d.%m.%Y})',
        'см/с',
        cmap='RdBu_r',
        vmin=-u_lim,
        vmax=u_lim,
    )
    hovmoller_pcolormesh(
        ax_v,
        adcp_30_full,
        'V',
        f'Меридиональная V, см/с (с {adcp_full_from:%d.%m.%Y})',
        'см/с',
        cmap='RdBu_r',
        vmin=-v_lim,
        vmax=v_lim,
    )
    fig.suptitle(
        f'Поле U (а) и V (б), полный вертикальный охват\n'
        f'{t0f:%d.%m.%Y %H:%M} — {t1f:%d.%m.%Y %H:%M}, усреднение 30 мин',
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig('fig_03_depth_time_UV.png', dpi=150, bbox_inches='tight')
    plt.close()

plot_adcp_partial_single_depth(
    adcp_30_partial,
    adcp_bottom_m,
    os.path.join(_SCRIPT_DIR, 'fig_02b_depth_time_speed_dir_partial.png'),
)

# =========================================================
# 7. ВРЕМЕННЫЕ РЯДЫ U, V НА ОТДЕЛЬНЫХ ГОРИЗОНТАХ
# =========================================================
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(horizons)))

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i, d in enumerate(horizons):
    sub = adcp_30[np.isclose(adcp_30['Depth'], d)].sort_values('datetime')
    # Вставляем NaN в пропуски > 1 часа, чтобы не соединять артефакты прямыми линиями
    sub = sub.set_index('datetime').reindex(
        pd.date_range(sub['datetime'].min(), sub['datetime'].max(), freq='30min')
    ).reset_index().rename(columns={'index': 'datetime'})
    lbl = f'{d:.1f} м'
    axes[0].plot(sub['datetime'], sub['U'], color=colors[i], lw=0.9, label=lbl)
    axes[1].plot(sub['datetime'], sub['V'], color=colors[i], lw=0.9, label=lbl)
for ax, ylabel, title in zip(
    axes,
    ['U (зональная), см/с', 'V (меридиональная), см/с'],
    ['Зональная компонента скорости U (см/с)', 'Меридиональная компонента скорости V (см/с)']
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
# 7b. 22–29 ИЮНЯ: поля U, V и временные ряды (анализное окно)
# =========================================================
note_path = os.path.join(_SCRIPT_DIR, 'adcp_june22_29_note.txt')
with open(note_path, 'w', encoding='utf-8') as fp:
    fp.write(ADCP_JUNE_WINDOW_NOTE + '\n')
print('\n' + ADCP_JUNE_WINDOW_NOTE)
print(f'\nПояснение сохранено: {note_path}')

adcp_30_june = filter_adcp_period(adcp_30_full)
adcp_30_june_mean = vertical_mean_adcp(adcp_30_june)
depths_june = np.sort(adcp_30_june['Depth'].dropna().unique())
horizons_june = [
    nearest_depth(depths_june, h) for h in pick_horizons(depths_june, n=3)
]
horizons_june = list(dict.fromkeys([h for h in horizons_june if h is not None]))
colors_june = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(horizons_june), 1)))

if adcp_30_june.empty:
    print('\nПредупреждение: нет данных ADCP за 22–29 июня.')
else:
    t0j, t1j = adcp_30_june['datetime'].min(), adcp_30_june['datetime'].max()
    print(
        f'\nОкно 22–29.06: {len(adcp_30_june)} строк (30 мин), '
        f'{t0j:%d.%m.%Y %H:%M} — {t1j:%d.%m.%Y %H:%M}, глубин {len(depths_june)}.'
    )

    u_lim_j = float(np.nanpercentile(np.abs(adcp_30_june['U'].dropna()), 98))
    v_lim_j = float(np.nanpercentile(np.abs(adcp_30_june['V'].dropna()), 98))
    fig, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)
    hovmoller_pcolormesh(
        ax_u, adcp_30_june, 'U',
        'Зональная U, см/с (22–29 июня)',
        'см/с', cmap='RdBu_r', vmin=-u_lim_j, vmax=u_lim_j,
    )
    hovmoller_pcolormesh(
        ax_v, adcp_30_june, 'V',
        'Меридиональная V, см/с (22–29 июня)',
        'см/с', cmap='RdBu_r', vmin=-v_lim_j, vmax=v_lim_j,
    )
    fig.suptitle(
        f'Поле U (а) и V (б), 22–29 июня 2023\n'
        f'{t0j:%d.%m.%Y %H:%M} — {t1j:%d.%m.%Y %H:%M}, усреднение 30 мин',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig('fig_03b_depth_time_UV_june22_29.png', dpi=150, bbox_inches='tight')
    plt.close()

    plot_timeseries_uv_horizons(
        adcp_30_june,
        horizons_june,
        colors_june,
        'fig_04b_timeseries_UV_june22_29.png',
        f'Ряды U, V на горизонтах, 22–29 июня 2023 ({t0j:%d.%m} — {t1j:%d.%m}), усреднение 30 мин',
        'Зональная компонента U, см/с (22–29 июня)',
        'Меридиональная компонента V, см/с (22–29 июня)',
    )
    print('  Сохранены: fig_03b_depth_time_UV_june22_29.png, fig_04b_timeseries_UV_june22_29.png')

# =========================================================
# 8. СТАТИСТИЧЕСКИЕ ПАРАМЕТРЫ ТЕЧЕНИЙ (30 мин)
# =========================================================
rows = build_current_stats_rows(adcp_30, horizons, adcp_30_mean)
stats_show = format_stats_dataframe(rows)
save_stats_table_figure(
    stats_show,
    'adcp_statistics.csv',
    'fig_05_statistics_table.png',
    'Статистические параметры течений (вся кампания, 30 мин)',
)

# 8b. Статистика за 22–29 июня (основное окно для сопоставления с термокосой)
if not adcp_30_june.empty:
    rows_june = build_current_stats_rows(adcp_30_june, horizons_june, adcp_30_june_mean)
    stats_june = format_stats_dataframe(rows_june)
    save_stats_table_figure(
        stats_june,
        'adcp_statistics_june22_29.csv',
        'fig_05b_statistics_table_june22_29.png',
        'Статистические параметры течений (22–29 июня 2023, 30 мин)',
    )

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
    dr = (np.degrees(np.arctan2(u_h, v_h)) + 360) % 360

    axes[0, ci].hist(
        spd, bins=30, color='steelblue', edgecolor='white', lw=0.3, label='Гистограмма',
    )
    if overlay_kde_on_hist(axes[0, ci], spd, bins=30):
        axes[0, ci].legend(fontsize=7, loc='upper right')
    axes[0, ci].set_xlabel('Скорость, см/с')
    axes[0, ci].set_ylabel('Частота')
    axes[0, ci].set_title(lbl)
    axes[0, ci].grid(True, lw=0.3)

    axes[1, ci].hist(
        dr, bins=36, range=(0, 360), color='darkorange', edgecolor='white', lw=0.3,
        label='Гистограмма',
    )
    if overlay_kde_on_hist(axes[1, ci], dr, bins=36, hist_range=(0, 360), color='crimson'):
        axes[1, ci].legend(fontsize=7, loc='upper right')
    axes[1, ci].set_xlabel('Направление, °')
    axes[1, ci].set_ylabel('Частота')
    axes[1, ci].set_xticks([0, 90, 180, 270, 360])
    axes[1, ci].set_title(lbl)
    axes[1, ci].grid(True, lw=0.3)

fig.suptitle(
    'Гистограммы: |V| (верх, см/с) и направление течений (низ)',
    fontsize=11,
)
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
# 10. СПЕКТР МОЩНОСТИ (периодограмм-метод): U и V
# =========================================================
# Формула: W(f_k) = (1 / (N * f_a)) * |X_k|^2
# где X_k = DFT[x(n)], N — число отсчётов, f_a — частота дискретизации.
# Спектральный анализ проводится на исходных (не усреднённых) данных.
# Дискретность dt_sec определена выше. U, V — см/с → СПМ в (см/с)²/Гц.
# Перед FFT — удаление линейного тренда (не только среднего).

def periodogram(x, fa):
    """
    Односторонний периодограмм-спектр мощности.
    W(f_k) = (1 / (N * fa)) * |DFT[x]_k|^2
    Возвращает (f, Pxx): частоты (Гц) и СПМ ((см/с)²/Гц), если x в см/с.
    """
    x = np.asarray(x, dtype=float)
    x = detrend(x, type='linear')
    N = len(x)
    Xk = np.fft.rfft(x)         # DFT: суммирование x(k)*exp(-j*2π*k*n/N)
    Pxx = (1.0 / (N * fa)) * np.abs(Xk) ** 2
    # Двойной вес для двусторонних частот (кроме 0 и Найквиста)
    if N % 2 == 0:
        Pxx[1:-1] *= 2
    else:
        Pxx[1:] *= 2
    f = np.fft.rfftfreq(N, d=1.0 / fa)   # частоты, Гц
    return f, Pxx


for comp, col, fname_base in [
    ('U (зональная), см/с', 'U', 'U'),
    ('V (меридиональная), см/с', 'V', 'V'),
]:
    fig, axes_s = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 5), sharey=False)
    if len(horizons) == 1:
        axes_s = [axes_s]
    for ci, d in enumerate(horizons):
        sub = adcp[np.isclose(adcp['Depth'], d)].sort_values('datetime')
        ts  = sub[col].dropna().values
        ax  = axes_s[ci]
        if len(ts) < 8:
            ax.set_title(f'{d:.1f} м (мало данных)')
            continue
        f_pg, Pxx = periodogram(ts, fs)
        mask = f_pg > 0
        T_min = 1.0 / f_pg[mask] / 60.0
        ax.loglog(T_min, Pxx[mask], lw=0.7, color='steelblue')
        ax.invert_xaxis()
        ax.grid(True, which='both', lw=0.4)
        annotate_period_grid_inside(ax)
        ax.set_xlabel('Период, мин')
        if ci == 0:
            ax.set_ylabel('СПМ, (см/с)²/Гц')
        ax.set_title(f'{d:.1f} м')
    fig.suptitle(
        f'Спектр мощности компоненты {comp}\n'
        f'W(f) = (1/N·fₐ)|X(f)|², линейный тренд снят, шаг: {dt_min:.2f} мин, fₐ = {fs:.4f} Гц',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f'fig_08_spectrum_{fname_base}.png', dpi=150)
    plt.close()

# =========================================================
# 11. ПРОГРЕССИВНЫЕ ВЕКТОРНЫЕ ДИАГРАММЫ (PVD)
# Фиксированный интервал для сопоставления: 22.06–25.06.2023.
# =========================================================
PVD_START = pd.Timestamp('2023-06-22 00:00:00')
PVD_END = pd.Timestamp('2023-06-25 23:59:59')

pvd_items = [
    (d, f'Глубина {d:.1f} м (30 мин, ПВД {PVD_START:%d.%m}–{PVD_END:%d.%m})')
    for d in horizons
] + [(None, f'Среднее по вертикали (30 мин, ПВД {PVD_START:%d.%m}–{PVD_END:%d.%m})')]
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
    sub = sub[(sub['datetime'] >= PVD_START) & (sub['datetime'] <= PVD_END)]
    progressive_vector_diagram(axes_flat[i], sub, title, max_days=None)

for k in range(len(pvd_items), len(axes_flat)):
    axes_flat[k].axis('off')

fig.suptitle(
    'Прогрессивные векторные диаграммы течений\n'
    f'({PVD_START:%d.%m.%Y}–{PVD_END:%d.%m.%Y}; U, V в см/с; оси — условное смещение ΣU·Δt, ΣV·Δt, км; '
    '1 км = 10⁵ см; не лагранжева траектория)',
    fontsize=10,
)
plt.tight_layout()
plt.savefig('fig_09_progressive_vector.png', dpi=150)
plt.close()

# ПВД на нижней ячейке ~19.8 м, 22–23 июня 2023
PVD_198_START = pd.Timestamp('2023-06-22 00:00:00')
PVD_198_END = pd.Timestamp('2023-06-23 23:59:59')
PVD_DEPTH_TARGET_M = 19.8

_depths_avail = adcp_30['Depth'].dropna().unique()
d_pvd198 = nearest_depth(_depths_avail, PVD_DEPTH_TARGET_M)
_pvd198_saved = False
if d_pvd198 is not None:
    sub_pvd198 = adcp_30[np.isclose(adcp_30['Depth'], d_pvd198)].copy()
    sub_pvd198 = sub_pvd198[
        (sub_pvd198['datetime'] >= PVD_198_START)
        & (sub_pvd198['datetime'] <= PVD_198_END)
    ]
    if len(sub_pvd198) >= 2:
        fig_pvd198, ax_pvd198 = plt.subplots(figsize=(7.5, 7))
        progressive_vector_diagram(
            ax_pvd198,
            sub_pvd198,
            f'ПВД, глубина {d_pvd198:.1f} м\n'
            f'{PVD_198_START:%d.%m.%Y} – {PVD_198_END:%d.%m.%Y} (30 мин)',
            max_days=None,
        )
        fig_pvd198.tight_layout()
        fig_pvd198.savefig('fig_09b_pvd_19p8m_june22_23.png', dpi=150, bbox_inches='tight')
        plt.close(fig_pvd198)
        _pvd198_saved = True
        print(
            f'ПВД 19.8 м: fig_09b_pvd_19p8m_june22_23.png '
            f'(глубина {d_pvd198:.2f} м, {len(sub_pvd198)} точек)'
        )
    else:
        print(f'ПВД 19.8 м: мало точек ({len(sub_pvd198)}), файл не сохранён')
else:
    print('ПВД 19.8 м: нет данных по глубинам ADCP.')

# =========================================================
# ИТОГ
# =========================================================
saved = [
    'fig_02_depth_time_speed_dir.png',
    'fig_02b_depth_time_speed_dir_partial.png',
    'fig_03_depth_time_UV.png',
    'fig_04_timeseries_UV.png',
]
if not adcp_30_june.empty:
    saved += [
        'fig_03b_depth_time_UV_june22_29.png',
        'fig_04b_timeseries_UV_june22_29.png',
        'fig_05b_statistics_table_june22_29.png',
        'adcp_statistics_june22_29.csv',
        'adcp_june22_29_note.txt',
    ]
saved += [
    'fig_05_statistics_table.png',
    'fig_06_histograms.png',
    'fig_07_rose_direction.png',
    'fig_08_spectrum_U.png',
    'fig_08_spectrum_V.png',
    'fig_09_progressive_vector.png',
    'adcp_statistics.csv',
]
if _pvd198_saved:
    saved.append('fig_09b_pvd_19p8m_june22_23.png')
print('\nСохранены файлы:')
for f in saved:
    print(f'  {f}')