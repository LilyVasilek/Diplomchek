# -*- coding: utf-8 -*-
# =========================================================
# 14. КОРОТКИЕ УЧАСТКИ (ЦУГИ) — термокоса T4
#   a) 20.06 23:30 — 21.06 01:00  (st4_fig12)
#   b) 20.06 01:30 — 03:00         (st4_fig19)
#   c) авто: 2 ч с макс. суммой высот волн (st4_fig20)
# =========================================================

# Основной пример (как раньше)
SHORT_SURGE_START = pd.to_datetime("2023-06-20 23:30")
SHORT_SURGE_END = pd.to_datetime("2023-06-21 01:00")

# Дополнительный фиксированный пример
EXAMPLE_SURGE_2_START = pd.to_datetime("2023-06-20 01:30")
EXAMPLE_SURGE_2_END = pd.to_datetime("2023-06-20 03:00")

SURGE_WINDOW_HOURS_AUTO = 2.0
SURGE_SCAN_STEP_MIN = 10

SHORT_MAX_DEPTH_M = 18.0
SHORT_TOP_SENSOR_M = float(np.nanmin(median_depths))


def _fmt_surge_time_axis(ax, w0, w1):
    """Деления времени каждые 30 мин."""
    ax.set_xlim(w0, w1)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
    ax.tick_params(axis="x", labelsize=8)


def _plot_surge_temperature_field(w0, w1, out_name, panel_title):
    """Тепловая карта T(z,t) на [w0, w1] → PNG в BASE_DIR."""
    w0, w1 = pd.to_datetime(w0), pd.to_datetime(w1)
    m = (time_30s >= w0) & (time_30s <= w1)

    fig_short, (ax_short, cax_short) = plt.subplots(
        1, 2, figsize=(11.0, 5.0),
        gridspec_kw={"width_ratios": [1, 0.04], "wspace": 0.12},
    )

    if np.any(m):
        t_sel = time_30s[m]
        temps_sel = temps_30s[m, :]
        t_vmin = float(np.nanpercentile(temps_sel, 2))
        t_vmax = float(np.nanpercentile(temps_sel, 98))
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

    try:
        if not np.any(m):
            raise ValueError("Нет данных в окне")
        TT_sel, DD_sel = np.meshgrid(t_sel, median_depths)
        cf_short = ax_short.contourf(
            TT_sel, DD_sel, temps_sel.T,
            levels=iso_bounds, cmap=cmap_short, norm=norm_short, extend="both",
        )
        ax_short.invert_yaxis()
        ax_short.set_ylabel("Глубина, м", fontsize=10)
        ax_short.set_xlabel("Время", fontsize=10)
        _fmt_surge_time_axis(ax_short, w0, w1)
        ax_short.grid(True, alpha=0.25)
        ax_short.set_ylim(SHORT_MAX_DEPTH_M, SHORT_TOP_SENSOR_M)
        fig_short.colorbar(
            cf_short, cax=cax_short, label="Температура, °C",
            ticks=np.arange(t_iso_floor, t_iso_ceil + 1, 1),
        )
    except ValueError:
        ax_short.text(
            0.5, 0.5, "Нет данных\nв выбранном окне",
            transform=ax_short.transAxes, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
        )
        ax_short.set_xlabel("Время", fontsize=10)
        ax_short.set_ylabel("Глубина, м", fontsize=10)
        _fmt_surge_time_axis(ax_short, w0, w1)
        ax_short.set_ylim(SHORT_MAX_DEPTH_M, SHORT_TOP_SENSOR_M)

    fig_short.suptitle(
        f"Температурное поле, термокоса T4\n{panel_title}",
        fontsize=11, y=0.98,
    )
    fig_short.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = os.path.join(BASE_DIR, out_name)
    fig_short.savefig(out_path, dpi=150)
    plt.close(fig_short)
    print(f"Сохранено: {out_name}")
    return out_path


def _waves_in_time_window(waves, time_arr, w0, w1):
    """Волны, целиком попавшие в [w0, w1]."""
    t = pd.to_datetime(time_arr)
    w0, w1 = pd.to_datetime(w0), pd.to_datetime(w1)
    inside = []
    for i0, i1, imax, h_wave, period_min in waves:
        if i0 < 0 or i1 >= len(t):
            continue
        if t[i0] >= w0 and t[i1] <= w1:
            inside.append((i0, i1, imax, h_wave, period_min))
    return inside


def find_best_fixed_duration_window(
    z_segment,
    time_segment,
    dt_seconds,
    duration_hours=2.0,
    start_step_min=10,
    min_period_min=3.0,
    min_height_m=WAVE_MIN_HEIGHT_M,
):
    """
    Скользящее окно фиксированной длительности (по умолчанию 2 ч).
    Критерий: макс. сумма высот волн, полностью попавших в окно;
    при равенстве — больше число волн, затем больше max(H).
    """
    z_seg = np.asarray(z_segment, dtype=float)
    t = pd.to_datetime(time_segment)
    if len(t) < 8:
        return None

    z_shift = z_seg - np.nanmean(z_seg)
    waves = detect_waves(
        z_shift,
        dt_seconds=dt_seconds,
        min_period_min=min_period_min,
        min_height_m=min_height_m,
    )
    if not waves:
        return None

    dur = pd.Timedelta(hours=float(duration_hours))
    step = pd.Timedelta(minutes=int(start_step_min))
    t_lo, t_hi = t[0], t[-1]

    best = None
    ts = t_lo
    while ts + dur <= t_hi:
        te = ts + dur
        inside = _waves_in_time_window(waves, t, ts, te)
        if not inside:
            ts += step
            continue
        heights = [w[3] for w in inside]
        h_sum = float(sum(heights))
        h_max = float(max(heights))
        score = (h_sum, len(inside), h_max)
        cand = {
            "start": ts,
            "end": te,
            "n_waves": len(inside),
            "h_sum": h_sum,
            "h_max": h_max,
            "waves": inside,
            "score": score,
        }
        if best is None or score > best["score"]:
            best = cand
        ts += step
    return best


def _print_waves_in_window(w0, w1, label):
    """Статистика волн на изотерме wave_iso в заданном окне."""
    w0, w1 = pd.to_datetime(w0), pd.to_datetime(w1)
    mask = (t_segment >= w0) & (t_segment <= w1)
    if not np.any(mask):
        print(
            f"\n{label}: нет данных изотермы {wave_iso:.1f} °C "
            f"({w0:%d.%m.%Y %H:%M} — {w1:%d.%m.%Y %H:%M})."
        )
        return

    t_short = t_segment[mask]
    z_short = z_segment[mask]
    z_short_shift = z_short - np.nanmean(z_short)
    selected = detect_waves(
        z_short_shift, dt_seconds=dt,
        min_period_min=3.0, min_height_m=WAVE_MIN_HEIGHT_M,
    )

    print(
        f"\n{label} — волны на изотерме {wave_iso:.1f} °C: "
        f"{w0:%d.%m.%Y %H:%M} — {w1:%d.%m.%Y %H:%M}"
    )
    print(
        f"  Найдено волн: {len(selected)} "
        f"(H >= {WAVE_MIN_HEIGHT_M} м, T >= 3 мин)"
    )
    if not selected:
        print("  Волны по критериям не обнаружены.")
        return

    rows = []
    for n, (i0, i1, _imax, h_wave, period_min) in enumerate(selected, start=1):
        rows.append({
            "№": n,
            "Начало": t_short[i0].strftime("%d.%m %H:%M:%S"),
            "Конец": t_short[i1].strftime("%d.%m %H:%M:%S"),
            "H, м": h_wave,
            "T, мин": period_min,
        })
    df_w = pd.DataFrame(rows)
    print(df_w.to_string(index=False, justify="center", float_format=lambda x: f"{x:.2f}"))
    h_vals = df_w["H, м"].to_numpy(dtype=float)
    t_vals = df_w["T, мин"].to_numpy(dtype=float)
    print(
        f"  H: min={h_vals.min():.2f}, mean={h_vals.mean():.2f}, max={h_vals.max():.2f} м; "
        f"T: min={t_vals.min():.1f}, mean={t_vals.mean():.1f}, max={t_vals.max():.1f} мин"
    )


# --- a) Основной цуг ---
print(
    f"\nКороткий участок (основной): "
    f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%d.%m.%Y %H:%M}"
)
_plot_surge_temperature_field(
    SHORT_SURGE_START, SHORT_SURGE_END, "st4_fig12.png",
    f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%d.%m.%Y %H:%M}",
)
_print_waves_in_window(SHORT_SURGE_START, SHORT_SURGE_END, "Основной цуг")

# --- b) Дополнительный пример 20.06 01:30–03:00 ---
print(
    f"\nДополнительный пример цуга: "
    f"{EXAMPLE_SURGE_2_START:%d.%m.%Y %H:%M} — {EXAMPLE_SURGE_2_END:%d.%m.%Y %H:%M}"
)
_plot_surge_temperature_field(
    EXAMPLE_SURGE_2_START, EXAMPLE_SURGE_2_END, "st4_fig19_surge_20june_0130.png",
    f"{EXAMPLE_SURGE_2_START:%d.%m.%Y %H:%M} — {EXAMPLE_SURGE_2_END:%d.%m.%Y %H:%M}",
)
_print_waves_in_window(EXAMPLE_SURGE_2_START, EXAMPLE_SURGE_2_END, "Доп. пример 20.06 01:30–03:00")

# --- c) Авто: 2 ч с наибольшими высотами волн ---
print(
    f"\nПоиск окна {SURGE_WINDOW_HOURS_AUTO:.0f} ч с макс. суммой высот волн "
    f"(изотерма {wave_iso:.1f} °C, шаг {SURGE_SCAN_STEP_MIN} мин)..."
)
best_2h = find_best_fixed_duration_window(
    z_segment, t_segment, dt,
    duration_hours=SURGE_WINDOW_HOURS_AUTO,
    start_step_min=SURGE_SCAN_STEP_MIN,
)

if best_2h is None:
    print("  Подходящее 2-часовое окно с волнами не найдено.")
else:
    AUTO_SURGE_START = best_2h["start"]
    AUTO_SURGE_END = best_2h["end"]
    print(
        f"  Лучшее окно: {AUTO_SURGE_START:%d.%m.%Y %H:%M} — "
        f"{AUTO_SURGE_END:%d.%m.%Y %H:%M}"
    )
    print(
        f"  Волн в окне: {best_2h['n_waves']}, "
        f"ΣH = {best_2h['h_sum']:.2f} м, max H = {best_2h['h_max']:.2f} м"
    )
    _plot_surge_temperature_field(
        AUTO_SURGE_START, AUTO_SURGE_END, "st4_fig20_surge_best_2h_waves.png",
        f"{AUTO_SURGE_START:%d.%m.%Y %H:%M} — {AUTO_SURGE_END:%d.%m.%Y %H:%M}",
    )
    _print_waves_in_window(AUTO_SURGE_START, AUTO_SURGE_END, "Авто-окно 2 ч (макс. ΣH)")
