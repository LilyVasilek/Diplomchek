# -*- coding: utf-8 -*-
# =========================================================
# 14. КОРОТКИЙ УЧАСТОК 21.06.2023 (цуг 21:20–23:20 ±1 ч) → st4_fig12.png
# =========================================================
SHORT_SURGE_START = pd.to_datetime(
    globals().get("SHORT_SURGE_START", "2023-06-21 21:20")
)
SHORT_SURGE_END = pd.to_datetime(
    globals().get("SHORT_SURGE_END", "2023-06-21 23:20")
)
windows_short = [
    (
        SHORT_SURGE_START - pd.Timedelta(hours=1),
        SHORT_SURGE_START,
        "час до цуга",
    ),
    (
        SHORT_SURGE_START,
        SHORT_SURGE_END,
        f"цуг ({SHORT_SURGE_START:%H:%M}–{SHORT_SURGE_END:%H:%M})",
    ),
    (
        SHORT_SURGE_END,
        SHORT_SURGE_END + pd.Timedelta(hours=1),
        "час после цуга",
    ),
]
SHORT_MAX_DEPTH_M = 18.0  # показываем только выше 18 м
SHORT_TOP_SENSOR_M = float(np.nanmin(median_depths))  # верхний датчик

temp_chunks = []
for w0, w1, _cap in windows_short:
    m = (time_30s >= w0) & (time_30s <= w1)
    if np.any(m):
        temp_chunks.append(temps_30s[m, :])
if temp_chunks:
    T_short_all = np.concatenate(temp_chunks, axis=0)
    t_vmin = float(np.nanpercentile(T_short_all, 2))
    t_vmax = float(np.nanpercentile(T_short_all, 98))
else:
    t_vmin = float(np.nanpercentile(temps_30s, 2))
    t_vmax = float(np.nanpercentile(temps_30s, 98))

t_iso_floor = int(np.floor(t_vmin))
t_iso_ceil = int(np.ceil(t_vmax))
if t_iso_ceil <= t_iso_floor:
    t_iso_ceil = t_iso_floor + 1
iso_bounds = np.arange(t_iso_floor, t_iso_ceil + 1, 1.0)
cmap_short = plt.get_cmap(TEMP_FIELD_CMAP, len(iso_bounds) - 1)
norm_short = matplotlib.colors.BoundaryNorm(iso_bounds, cmap_short.N, clip=True)

_panel_hours = [
    (w1 - w0).total_seconds() / 3600.0 for w0, w1, _ in windows_short
]
_side_h = _panel_hours[0] if _panel_hours[0] > 0 else 1.0
_widths = [max(0.2, h / _side_h) for h in _panel_hours] + [0.07]

fig_short = plt.figure(figsize=(11.5, 5.0))
gs_short = fig_short.add_gridspec(
    1, 4, width_ratios=_widths, wspace=0.38,
)
axes_short = [fig_short.add_subplot(gs_short[0, 0])]
for i in range(1, 3):
    axes_short.append(
        fig_short.add_subplot(gs_short[0, i], sharey=axes_short[0]),
    )
cax_short = fig_short.add_subplot(gs_short[0, 3])

last_cf = None
for j, (ax, (w0, w1, caption)) in enumerate(zip(axes_short, windows_short)):
    try:
        m = (time_30s >= w0) & (time_30s <= w1)
        if not np.any(m):
            raise ValueError("Нет данных в этом временном окне")
        t_sel = time_30s[m]
        temps_sel = temps_30s[m, :]
        TT_sel, DD_sel = np.meshgrid(t_sel, median_depths)
        last_cf = ax.contourf(
            TT_sel,
            DD_sel,
            temps_sel.T,
            levels=iso_bounds,
            cmap=cmap_short,
            norm=norm_short,
            extend="both",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Время")
        if j == 0:
            ax.set_ylabel("Глубина, м")
        ax.set_title(f"{w0.strftime('%H:%M')} — {w1.strftime('%H:%M')}")
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(SHORT_MAX_DEPTH_M, SHORT_TOP_SENSOR_M)
    except ValueError:
        ax.text(
            0.5,
            0.5,
            "Нет данных\nв этом временном окне",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
        )
        ax.set_title(f"{w0.strftime('%H:%M')} — {w1.strftime('%H:%M')}")
        ax.set_xlabel("Время")
        if j == 0:
            ax.set_ylabel("Глубина, м")
        ax.set_ylim(SHORT_MAX_DEPTH_M, SHORT_TOP_SENSOR_M)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
    if j > 0:
        ax.tick_params(labelleft=False)

if last_cf is not None:
    fig_short.colorbar(
        last_cf,
        cax=cax_short,
        label="Температура, °C",
        ticks=np.arange(t_iso_floor, t_iso_ceil + 1, 1),
    )
fig_short.suptitle(
    f"Температурное поле вокруг цуга ({SHORT_SURGE_START:%d.%m.%Y %H:%M}–{SHORT_SURGE_END:%H:%M})",
    fontsize=11,
    y=0.98,
)
fig_short.subplots_adjust(left=0.07, right=0.90, top=0.88, bottom=0.12, wspace=0.38)
fig_short.savefig(os.path.join(BASE_DIR, "st4_fig12.png"), dpi=150)
plt.close(fig_short)
print(
    "Сохранено: st4_fig12.png "
    f"({windows_short[0][0]:%H:%M}–{windows_short[-1][1]:%H:%M}, "
    f"цуг {SHORT_SURGE_START:%d.%m %H:%M}–{SHORT_SURGE_END:%H:%M}, шкала отдельно)"
)

# ---------------------------------------------------------
# Статистика волн на коротком участке цуга (центральное окно)
# ---------------------------------------------------------
short_mask_wave = (t_segment >= SHORT_SURGE_START) & (t_segment <= SHORT_SURGE_END)
if np.any(short_mask_wave):
    t_short = t_segment[short_mask_wave]
    z_short = z_segment[short_mask_wave]
    z_short_shift = z_short - np.nanmean(z_short)

    selected_waves_short = detect_waves(
        z_short_shift,
        dt_seconds=dt,
        min_period_min=3.0,
        min_height_m=WAVE_MIN_HEIGHT_M,
    )

    print(
        f"\nКороткий участок (цуг): {t_short[0].strftime('%d.%m.%Y %H:%M')} — "
        f"{t_short[-1].strftime('%d.%m.%Y %H:%M')}"
    )
    print(
        f"Найдено волн: {len(selected_waves_short)} "
        f"(h >= {WAVE_MIN_HEIGHT_M} м, T >= 3 мин)"
    )

    if len(selected_waves_short) > 0:
        rows_short = []
        for n, (i0, i1, _imax, h_wave, period_min) in enumerate(selected_waves_short, start=1):
            rows_short.append(
                {
                    "№": n,
                    "Начало": t_short[i0].strftime("%d.%m %H:%M:%S"),
                    "Конец": t_short[i1].strftime("%d.%m %H:%M:%S"),
                    "Высота H, м": h_wave,
                    "Период T, мин": period_min,
                }
            )
        df_waves_short = pd.DataFrame(rows_short)
        print("\nТаблица волн на коротком участке:")
        print(df_waves_short.to_string(index=False, justify="center", float_format=lambda x: f"{x:.2f}"))

        h_vals = df_waves_short["Высота H, м"].to_numpy(dtype=float)
        t_vals = df_waves_short["Период T, мин"].to_numpy(dtype=float)
        print("\nСводная статистика (короткий участок):")
        print(
            f"  H, м: min={np.min(h_vals):.2f}, mean={np.mean(h_vals):.2f}, max={np.max(h_vals):.2f}"
        )
        print(
            f"  T, мин: min={np.min(t_vals):.2f}, mean={np.mean(t_vals):.2f}, max={np.max(t_vals):.2f}"
        )
    else:
        print("На коротком участке волны по заданным критериям не обнаружены.")
else:
    print(
        f"\nНет данных изотермы {wave_iso:.1f}°C в окне цуга "
        f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%H:%M}."
    )