import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# ===================== НАСТРОЙКИ =====================
stations = {
    "а – Т1": {
        "xlsx": r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st1.xlsx",
        "dep": "dep1", "time": "ss1", "temp": "temp1"
    },
    "б – Т2": {
        "xlsx": r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st2.xlsx",
        "dep": "dep1", "time": "ss", "temp": "temp1"   # поправь здесь на реальные имена листов st2, если отличаются
    },
    "в – Т3": {
        "xlsx": r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st3.xlsx",
        "dep": "dep1", "time": "ss10s", "temp": "temp1"   # аналогично для st3
    },
    "г – Т4": {
        "xlsx": r"C:\Документы\ДИПЛОМ\Химченко_данные\Термокосы\st4.xlsx",
        "dep": "dep_n", "time": "ss", "temp": "TV"
    }
}

resample_interval = "30s"  # усреднение 30 секунд (строчная 's' во избежание FutureWarning)

# ===================== ФУНКЦИИ =====================
def load_station(cfg):
    dep_df = pd.read_excel(cfg["xlsx"], sheet_name=cfg["dep"], header=None)
    time_df = pd.read_excel(cfg["xlsx"], sheet_name=cfg["time"], header=None)
    temp_df = pd.read_excel(cfg["xlsx"], sheet_name=cfg["temp"], header=None)

    depths = dep_df.values.astype(float)
    temps = temp_df.values.astype(float)
    times = pd.to_datetime(time_df.iloc[:, 0])

    return depths, temps, times


def resample_30s(times, temps):
    df = pd.DataFrame(temps, index=times)
    df_resampled = df.resample(resample_interval).mean()
    return df_resampled.index, df_resampled.values


def match_depths_greedy(ref_depths, cand_depths, n=None):
    """
    Подбор датчиков по глубине: для каждой ref-глубины выбираем ближайший свободный датчик из cand.
    Возвращает индексы cand в порядке ref_depths.
    """
    ref_depths = np.asarray(ref_depths, dtype=float)
    cand_depths = np.asarray(cand_depths, dtype=float)

    if n is None:
        n = min(len(ref_depths), len(cand_depths))

    # берём n самых "репрезентативных" ref-глубин: равномерно по диапазону
    if len(ref_depths) > n:
        ref_sorted_idx = np.argsort(ref_depths)
        pick = np.linspace(0, len(ref_sorted_idx) - 1, n).round().astype(int)
        ref_idx = ref_sorted_idx[pick]
        ref_sel = ref_depths[ref_idx]
    else:
        ref_sel = ref_depths.copy()

    used = set()
    out = []
    for d0 in ref_sel:
        # среди неиспользованных кандидатов ищем ближайший по |depth-d0|
        diffs = np.abs(cand_depths - d0)
        diffs[list(used)] = np.inf
        j = int(np.argmin(diffs))
        if not np.isfinite(diffs[j]):
            break
        used.add(j)
        out.append(j)
    return out, ref_sel


# ===================== ЗАГРУЗКА ДАННЫХ =====================
stations_data = {}

for name, cfg in stations.items():
    depths, temps, times = load_station(cfg)
    t_new, temp_new = resample_30s(times, temps)

    stations_data[name] = {
        "depths": depths,
        "temps": temp_new,
        "times": t_new,
        "median_depths": np.nanmedian(depths.astype(float), axis=0),
    }

# Определяем общий диапазон дат по всем станциям
all_mins = []
all_maxs = []
for data in stations_data.values():
    if len(data["times"]) == 0:
        continue
    all_mins.append(data["times"].min())
    all_maxs.append(data["times"].max())

if all_mins and all_maxs:
    global_min = min(all_mins)
    global_max = max(all_maxs)
    print("Доступный диапазон по данным термокос (после усреднения):")
    print(f"  c {global_min.strftime('%Y-%m-%d %H:%M')} по {global_max.strftime('%Y-%m-%d %H:%M')}")
else:
    raise RuntimeError("Во всех файлах нет валидных временных меток.")

start_str = input("Введите начало интервала (ГГГГ-ММ-ДД ЧЧ:ММ): ").strip()
end_str = input("Введите конец интервала   (ГГГГ-ММ-ДД ЧЧ:ММ): ").strip()
t_start = pd.to_datetime(start_str)
t_end = pd.to_datetime(end_str)
if t_end <= t_start:
    raise ValueError("Конец интервала должен быть позже начала.")

# Выбираем "эталон" по глубинам: станция с минимальным числом датчиков
ref_station = min(stations_data.keys(), key=lambda k: len(stations_data[k]["median_depths"]))
ref_depths = stations_data[ref_station]["median_depths"]
target_n = min(len(stations_data[k]["median_depths"]) for k in stations_data.keys())
print(f"\nЭталонная станция для сравнения глубин: {ref_station} (N={len(ref_depths)})")
print(f"Число сравниваемых горизонтов: {target_n}")

# ===================== ПОСТРОЕНИЕ ГРАФИКОВ =====================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

for ax, (st_name, data) in zip(axes, stations_data.items()):
    times = data["times"]
    temps = data["temps"]
    depths = data["depths"]
    med_depths = data["median_depths"]

    # отбираем данные за выбранный интервал
    mask = (times >= t_start) & (times <= t_end)

    if not mask.any():
        ax.text(0.5, 0.5, f'Нет данных за интервал\n{t_start} — {t_end}', transform=ax.transAxes,
                ha='center', va='center')
        continue

    t_sel = times[mask]
    temp_sel = temps[mask, :]

    # Подбор одинаковых по глубине датчиков (особенно важно для Т4, где датчиков больше)
    sel_idx, ref_sel = match_depths_greedy(ref_depths, med_depths, n=target_n)
    sel_idx = np.array(sel_idx, dtype=int)
    temp_sel = temp_sel[:, sel_idx]
    med_sel = med_depths[sel_idx]

    # сортируем по глубине, чтобы станции шли одинаково "сверху вниз"
    order = np.argsort(med_sel)
    temp_sel = temp_sel[:, order]
    med_sel = med_sel[order]

    print(f"\n{st_name}: выбраны глубины (м) для сравнения: " + ", ".join(f"{d:.1f}" for d in med_sel))

    n_sensors = temp_sel.shape[1]
    colors = cm.viridis(np.linspace(0, 1, n_sensors))

    for i in range(n_sensors):
        ax.plot(t_sel, temp_sel[:, i], color=colors[i], lw=1.5, label=f'{med_sel[i]:.1f} м')

    ax.set_title(st_name, loc='left', fontsize=12)
    ax.set_ylabel("T, °C")
    ax.grid(alpha=0.4)

axes[-1].set_xlabel("Время")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Глубина датчика", loc='center right')

fig.suptitle(
    f"Колебания температуры на горизонтах (усреднение 30 с)\n{t_start.strftime('%Y-%m-%d %H:%M')} — {t_end.strftime('%Y-%m-%d %H:%M')}",
             fontsize=14)

fig.tight_layout(rect=[0, 0, 0.88, 0.95])
plt.show()
