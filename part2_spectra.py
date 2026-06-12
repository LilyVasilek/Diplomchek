# -*- coding: utf-8 -*-
# =========================================================
# 9–11. СПЕКТРЫ НА ОБЩЕМ УЧАСТКЕ (ДВА ГРАФИКА)
#   1) Амплитудные спектры — три изотермы на одной сетке
#   2) PSD — три изотермы + модель Гарретта–Манка на одной сетке
# =========================================================
fig_sp, (ax_amp, ax_psd) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
amp_colors = ["royalblue", "seagreen", "coral"]
psd_colors = ["darkviolet", "sienna", "darkgoldenrod"]

print(f"\nМодель Гарретта–Манка: S(f) = C_M·f_in·√(f²−f_in²) / (N_max·f³)")
print(f"  Модель определена в диапазоне f_in < f < N_max  (N_max = {N_max_cph:.4f} ч⁻¹)")
print(f"  f_in (инерционная частота) = {fin:.4f} ч⁻¹")

spec_console_data = []

for idx, T_iso in enumerate(iso_values):
    z_iso = iso_depths[T_iso]
    z_segment = z_iso[c_start:c_end]
    z_mean = np.nanmean(z_segment)

    signal = detrend(z_segment, type='linear')
    N_pts = len(signal)

    # --- Амплитудный спектр: F(ν) = ∫ f(t)·exp(−i2πνt) dt ---
    Feta = np.fft.fft(signal) / N_pts
    Fv = np.linspace(0, Fn, N_pts // 2 + 1)
    amplitude = 2 * np.abs(Feta[:len(Fv)])

    # --- Спектральная плотность мощности: Ŵ(ω) = (1/(N·fₐ)) |Σ x(k)·exp(−jωkT)|² ---
    f_hz = np.fft.rfftfreq(N_pts, d=dt)
    f_psd = f_hz * 3600.0
    fa = 1.0 / dt
    X = np.fft.rfft(signal)
    Pxx = ((1.0 / (N_pts * fa)) * (np.abs(X) ** 2)) / 3600.0

    # --- Модель Гарретта–Манка (N = N_max по профилю В–Б) ---
    N_used = N_max_cph
    S_GM = garrett_munk_psd(f_psd, N_used, fin)
    mg = (f_psd > fin) & (f_psd < N_used)

    gm_slope = np.nan
    mgp_fit = mg & (S_GM > 0) & np.isfinite(S_GM)
    if np.count_nonzero(mgp_fit) >= 3:
        lf = np.log10(f_psd[mgp_fit])
        ls = np.log10(S_GM[mgp_fit])
        gm_slope, _ = np.polyfit(lf, ls, 1)

    gm_slope_str = f"{gm_slope:.2f}" if np.isfinite(gm_slope) else "—"
    print(
        f"  {T_iso}°C: z̄ = {z_mean:.1f} м, N_max = {N_used:.4f} ч⁻¹, "
        f"наклон Г–М (log–log) = {gm_slope_str}"
    )

    spec_console_data.append(
        {
            "T_iso": T_iso,
            "z_mean": z_mean,
            "N_used": N_used,
            "gm_slope_str": gm_slope_str,
            "N_pts": N_pts,
            "Fv": Fv.copy(),
            "amplitude": amplitude.copy(),
            "f_psd": f_psd.copy(),
            "Pxx": Pxx.copy(),
            "S_GM": S_GM.copy(),
        }
    )

    lbl = f"{T_iso}°C"

    # --- 1) Амплитудный спектр ---
    mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
    if np.any(mfft):
        ax_amp.loglog(Fv[mfft], amplitude[mfft], color=amp_colors[idx], lw=1, label=lbl)

    # --- 2) PSD ---
    mpsd = (f_psd > 0) & np.isfinite(Pxx) & (Pxx > 0)
    if np.any(mpsd):
        ax_psd.loglog(f_psd[mpsd], Pxx[mpsd], color=psd_colors[idx], lw=1, label=f"PSD {lbl}")

    # --- Модель Г–М (чёрная, разные стили линий для различения) ---
    gm_styles = ["-", "--", "-."]
    mgp = (f_psd > 0) & np.isfinite(S_GM) & (S_GM > 0)
    if np.any(mgp):
        ax_psd.loglog(f_psd[mgp], S_GM[mgp], color="black", ls=gm_styles[idx], lw=1.5,
                      alpha=0.8, label=f"Г–М {T_iso}°C (N_max)")

if np.isfinite(N_max_cph) and N_max_cph > fin:
    ax_amp.axvline(
        N_max_cph,
        color="crimson",
        ls="--",
        lw=1.3,
        alpha=0.95,
        zorder=6,
        label=f"$N_{{max}}$ = {N_max_cph:.2f} ч⁻¹",
    )
    ax_psd.axvline(
        N_max_cph,
        color="crimson",
        ls="--",
        lw=1.3,
        alpha=0.95,
        zorder=6,
        label=f"$N_{{max}}$ = {N_max_cph:.2f} ч⁻¹",
    )

_write_spectra_detail_txt(spec_console_data, fin, N_max_cph, seg_len, seg_hours, dt)

ax_amp.set_ylabel("Амплитуда, м")
ax_amp.set_title("Амплитудные спектры изотерм (общий участок)")
ax_amp.grid(True, which="both", alpha=0.3)
ax_amp.legend(fontsize=8, loc="best")
_add_amp_spectrum_period_reference_lines(ax_amp, t_max_h=17.1)
ax_psd.set_xlabel("Частота, цикл/час")
ax_psd.set_ylabel("PSD, м²·час")
ax_psd.set_title("Спектральная плотность мощности + модель Гарретта–Манка (общий участок)")
ax_psd.grid(True, which="both", alpha=0.3)
ax_psd.legend(fontsize=8, loc="best")
_add_amp_spectrum_period_reference_lines(ax_psd, t_max_h=17.1)

fig_sp.suptitle(f"Спектральный анализ на общем участке ({seg_len} точ., {seg_hours:.1f} ч)",
                y=1.01, fontsize=13)
fig_sp.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig08.png"), dpi=150)
plt.close("all")


def _isotherm_depth_series(T_iso, *, native=False):
    """Глубина изотермы: native=False — сетка 30 с; native=True — исходный шаг записи."""
    if native:
        t_arr = time
        t_vals = temps
        z_sensors = np.nanmedian(depths, axis=0)
    else:
        t_arr = time_30s
        t_vals = temps_30s
        z_sensors = median_depths
    z = np.full(len(t_arr), np.nan)
    for t in range(len(t_arr)):
        z[t] = interp1d(
            t_vals[t, :],
            z_sensors,
            bounds_error=False,
            fill_value=np.nan,
        )(T_iso)
    return z


def plot_single_isotherm_spectrum(
    T_iso,
    *,
    c_start=None,
    c_end=None,
    use_common_segment=False,
    use_native_dt=False,
    period_mark_h=17.1,
    out_file="st4_fig08_iso13.png",
    add_gm=True,
):
    """Спектр одной изотермы: амплитуда + PSD; Г–М с N_max; пунктиры периодов в часах."""
    if skip_standalone_isotherm_spectrum(T_iso):
        return
    z_iso = np.asarray(_isotherm_depth_series(T_iso, native=use_native_dt), dtype=float)
    if use_common_segment:
        if c_start is None or c_end is None:
            raise ValueError("Для use_common_segment нужны c_start и c_end")
        z_work = z_iso[c_start:c_end]
        seg_note = "общий участок"
    else:
        z_work = z_iso
        seg_note = "полная запись"

    loc_s, loc_e = longest_continuous_segment(z_work)
    if loc_s is None:
        print(f"Изотерма {T_iso}°C: нет непрерывного ряда без пропусков ({seg_note}).")
        return
    z_use = z_work[loc_s:loc_e]
    if len(z_use) < 32:
        print(f"Изотерма {T_iso}°C: мало валидных точек ({len(z_use)}) для спектра.")
        return
    z_mean = float(np.mean(z_use))

    if use_native_dt:
        t_slice = time[c_start:c_end] if use_common_segment else time
        t_use = pd.to_datetime(t_slice[loc_s:loc_e])
        dt_s = float(np.median(np.diff(t_use.values.astype("datetime64[ns]"))) / 1e9)
        if not np.isfinite(dt_s) or dt_s <= 0:
            dt_s = float(np.median(np.diff(time.values.astype("datetime64[ns]"))) / 1e9)
        fn_local = (1.0 / dt_s) * 3600.0 / 2.0
        dt_note = f"исходный шаг Δt ≈ {dt_s:.0f} с"
    else:
        dt_s = float(dt)
        fn_local = Fn
        dt_note = f"шаг Δt = {dt_s:.0f} с"

    seg_hours_local = (len(z_use) - 1) * dt_s / 3600.0

    signal = detrend(z_use, type="linear")
    N_pts = len(signal)
    Feta = np.fft.fft(signal) / N_pts
    Fv = np.linspace(0, fn_local, N_pts // 2 + 1)
    amplitude = 2 * np.abs(Feta[: len(Fv)])

    f_hz = np.fft.rfftfreq(N_pts, d=dt_s)
    f_psd = f_hz * 3600.0
    fa = 1.0 / dt_s
    X = np.fft.rfft(signal)
    Pxx = ((1.0 / (N_pts * fa)) * (np.abs(X) ** 2)) / 3600.0

    N_used = N_max_cph
    S_GM = garrett_munk_psd(f_psd, N_used, fin) if add_gm else None

    fig_i, (ax_a, ax_p) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    mfft = (Fv > 0) & (amplitude > 0) & np.isfinite(amplitude)
    if np.any(mfft):
        ax_a.loglog(Fv[mfft], amplitude[mfft], color="royalblue", lw=1.2, label=f"{T_iso}°C")

    ax_a.set_ylabel("Амплитуда, м")
    ax_a.set_title(
        f"Амплитудный спектр изотермы {T_iso}°C ({seg_note}, {dt_note}, z̄ = {z_mean:.1f} м)"
    )
    ax_a.grid(True, which="both", alpha=0.3)
    _add_amp_spectrum_period_reference_lines(ax_a, t_max_h=period_mark_h)

    mpsd = (f_psd > 0) & np.isfinite(Pxx) & (Pxx > 0)
    if np.any(mpsd):
        ax_p.loglog(f_psd[mpsd], Pxx[mpsd], color="darkviolet", lw=1.2, label=f"PSD {T_iso}°C")

    if add_gm and S_GM is not None:
        mgp = (f_psd > 0) & np.isfinite(S_GM) & (S_GM > 0)
        if np.any(mgp):
            ax_p.loglog(
                f_psd[mgp],
                S_GM[mgp],
                color="black",
                ls="-",
                lw=1.5,
                alpha=0.85,
                label=f"Г–М (N_max = {N_used:.2f} ч⁻¹)",
            )

    ax_p.set_xlabel("Частота, цикл/час")
    ax_p.set_ylabel("PSD, м²·ч")
    ax_p.set_title(f"Спектральная плотность мощности + Г–М, {T_iso}°C")
    ax_p.grid(True, which="both", alpha=0.3)
    _add_amp_spectrum_period_reference_lines(ax_p, t_max_h=period_mark_h)

    if np.isfinite(N_used) and N_used > fin:
        for ax in (ax_a, ax_p):
            ax.axvline(
                N_used,
                color="crimson",
                ls="--",
                lw=1.3,
                zorder=5,
                label=f"$N_{{max}}$ = {N_used:.2f} ч⁻¹",
            )

    ax_a.legend(fontsize=8, loc="best")
    ax_p.legend(fontsize=8, loc="best")

    fig_i.suptitle(
        f"Спектр изотермы {T_iso}°C  |  {seg_note}, {len(z_use)} отсч., {seg_hours_local:.1f} ч  |  "
        f"$N_{{max}}$ = {N_used:.2f} ч⁻¹",
        fontsize=11,
        y=1.01,
    )
    fig_i.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_file if os.path.isabs(out_file) else os.path.join(BASE_DIR, out_file)
    fig_i.savefig(out_path, dpi=200)
    plt.close(fig_i)

    print(
        f"\nИзотерма {T_iso}°C (отдельный график): {seg_note}, {dt_note}, "
        f"{len(z_use)} отсч., {seg_hours_local:.1f} ч, z̄ = {z_mean:.1f} м, "
        f"N_max = {N_used:.4f} ч⁻¹"
    )
    print(f"  Сохранено: {out_path}")


# Отдельные спектры одной изотермы: только 13 °C (для 23 °C отдельный график не строим).
plot_single_isotherm_spectrum(
    T_iso=13.0,
    use_native_dt=False,
    use_common_segment=False,
    period_mark_h=17.1,
    out_file="st4_fig08_iso13.png",
    add_gm=True,
)

# =========================================================
# 12. ЛУЧШАЯ ИЗОТЕРМА СРЕДИ ВСЕХ ИЗОТЕРМ
# =========================================================
def add_day_boundaries(ax, t_arr):
    day_starts = pd.date_range(t_arr[0].normalize(), t_arr[-1].normalize(), freq="D")
    for d in day_starts:
        ax.axvline(d, color="black", ls="--", lw=0.8, alpha=0.6)


all_iso_values = np.arange(np.floor(T_min), np.ceil(T_max) + 0.01, 0.5)
all_iso_depths = {}
for T_iso in all_iso_values:
    z_iso = np.full(len(time_30s), np.nan)
    for t in range(len(time_30s)):
        z_iso[t] = interp1d(
            temps_30s[t, :],
            median_depths,
            bounds_error=False,
            fill_value=np.nan,
        )(T_iso)
    all_iso_depths[T_iso] = z_iso

best_iso = None
best_len = 0
best_start, best_end = None, None

for T_iso in all_iso_values:
    z_iso = all_iso_depths[T_iso]
    s, e = longest_continuous_segment(z_iso)
    if s is None:
        continue
    length = e - s
    if length > best_len:
        best_len = length
        best_iso = T_iso
        best_start, best_end = s, e

if best_iso is None:
    raise RuntimeError("Нет непрерывных участков для всех изотерм!")

print(
    f"\nЛучшая изотерма среди всех: {best_iso:.1f}°C "
    f"({best_len} точек, {(best_len - 1) * dt / 3600:.1f} часов)"
)


WAVE_SEG_MIN_POINTS = 32


def _iso_z_series(T_iso):
    """Полный ряд глубины изотермы T_iso (с NaN в пропусках)."""
    z_full = all_iso_depths.get(T_iso)
    if z_full is None:
        z_full = np.full(len(time_30s), np.nan)
        for t in range(len(time_30s)):
            z_full[t] = interp1d(
                temps_30s[t, :],
                median_depths,
                bounds_error=False,
                fill_value=np.nan,
            )(T_iso)
        all_iso_depths[T_iso] = z_full
    return z_full


def _longest_segment_for_iso(T_iso, min_points=WAVE_SEG_MIN_POINTS):
    """Самый длинный непрерывный участок глубины изотермы T_iso."""
    z_full = _iso_z_series(T_iso)
    s, e = longest_continuous_segment(z_full)
    if s is None or (e - s) < min_points:
        return None
    return int(s), int(e), z_full[s:e], time_30s[s:e]


def _count_waves_on_iso(T_iso, dt_s, min_points=WAVE_SEG_MIN_POINTS):
    seg = _longest_segment_for_iso(T_iso, min_points=min_points)
    if seg is None:
        return 0, None
    n_w = _count_waves_on_segment(seg[2], dt_s)
    return n_w, seg


def detect_waves(z_shifted, dt_seconds, min_period_min=3.0, min_height_m=WAVE_MIN_HEIGHT_M):
    """Поиск волн между соседними минимумами (T >= min_period_min, H >= min_height_m)."""
    minima, _ = find_peaks(-z_shifted)
    waves = []
    for i in range(len(minima) - 1):
        i0, i1 = minima[i], minima[i + 1]
        period_min = (i1 - i0) * dt_seconds / 60.0
        if period_min < min_period_min:
            continue
        seg = z_shifted[i0:i1 + 1]
        imax = i0 + np.argmax(seg)
        zmax = z_shifted[imax]
        h_front = zmax - z_shifted[i0]
        h_rear = zmax - z_shifted[i1]
        h_wave = 0.5 * (h_front + h_rear)
        if h_wave >= min_height_m:
            waves.append((i0, i1, imax, h_wave, period_min))
    return waves


def _count_waves_on_segment(z_seg, dt_s):
    z_seg = np.asarray(z_seg, dtype=float)
    if z_seg.size < 32:
        return 0
    z_shift = z_seg - np.nanmean(z_seg)
    return len(detect_waves(z_shift, dt_seconds=dt_s, min_period_min=3.0, min_height_m=WAVE_MIN_HEIGHT_M))


def select_best_surge_window(
    waves,
    time_arr,
    *,
    min_hours=1.0,
    max_hours=1.5,
    duration_step_min=10,
    start_step_min=10,
):
    """
    Скользящее окно 1–1.5 ч по всему участку изотермы.
    Выбирается интервал с максимальным числом волн (волна целиком внутри окна).
    """
    if not waves:
        return None
    t = pd.to_datetime(time_arr)
    if len(t) < 2:
        return None

    rec = []
    for i0, i1, _imax, h_wave, _period in waves:
        if i0 < 0 or i1 >= len(t) or i1 <= i0:
            continue
        rec.append({"t0": t[i0], "t1": t[i1], "h": float(h_wave)})
    if not rec:
        return None

    t_lo, t_hi = t[0], t[-1]
    start_step = pd.Timedelta(minutes=start_step_min)
    dur_min_lo = int(min_hours * 60)
    dur_max = int(max_hours * 60)

    best = None
    ts = t_lo
    while ts <= t_hi:
        for dur_min in range(dur_min_lo, dur_max + 1, duration_step_min):
            te = ts + pd.Timedelta(minutes=dur_min)
            if te > t_hi:
                continue
            inside = [r for r in rec if r["t0"] >= ts and r["t1"] <= te]
            n_in = len(inside)
            if n_in == 0:
                continue
            h_sum = float(sum(r["h"] for r in inside))
            score = (n_in, h_sum, -dur_min)
            cand = {
                "start": ts,
                "end": te,
                "n_waves": n_in,
                "duration_min": dur_min,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = cand
        ts += start_step
    return best


_wave_prompt = input(
    f"\nИзотерма для анализа волн, °C (Enter — авто, {best_iso:.1f}°C): "
).strip()
try:
    wave_iso = float(_wave_prompt) if _wave_prompt else float(best_iso)
except ValueError:
    wave_iso = float(best_iso)
    print(f"  Некорректный ввод, используется {wave_iso:.1f}°C")

print(
    f"\nЧисло волн (H >= {WAVE_MIN_HEIGHT_M} м, T >= 3 мин) "
    f"на самом длинном непрерывном участке:"
)
_compare_set = list(dict.fromkeys(list(iso_values) + [float(best_iso)]))
for T_try in _compare_set:
    n_w, seg_try = _count_waves_on_iso(T_try, dt)
    if seg_try is None:
        print(f"  {T_try:.1f}°C: нет непрерывного ряда")
        continue
    mark = "  <-- анализ" if np.isclose(T_try, wave_iso) else ""
    print(
        f"  {T_try:.1f}°C: {n_w} волн, участок "
        f"{seg_try[3][0]:%d.%m %H:%M} — {seg_try[3][-1]:%d.%m %H:%M}{mark}"
    )

_wave_seg = _longest_segment_for_iso(wave_iso)
if _wave_seg is None:
    raise RuntimeError(f"Нет непрерывного участка для изотермы {wave_iso:.1f}°C!")
analysis_start, analysis_end, z_segment, t_segment = _wave_seg
wave_iso_tag = f"{wave_iso:.1f}C".replace(".", "p")
print(
    f"\nАнализ волн: изотерма {wave_iso:.1f}°C, "
    f"{analysis_end - analysis_start} точек, "
    f"{(analysis_end - analysis_start - 1) * dt / 3600:.1f} ч"
)

z_mean_wave = float(np.nanmean(z_segment))

fig_best, ax_best = plt.subplots(figsize=(15, 5))
ax_best.plot(t_segment, z_segment, lw=0.8, color="teal", label=f"{wave_iso:.1f}°C")
ax_best.axhline(z_mean_wave, color="red", ls="--", lw=1, label=f"z̄ = {z_mean_wave:.1f} м")
add_day_boundaries(ax_best, t_segment)
ax_best.plot([], [], color="black", ls="--", lw=1, label="Границы суток")
ax_best.invert_yaxis()
ax_best.set_ylabel("Глубина, м")
ax_best.set_xlabel("Время")
ax_best.set_title(
    f"Изотерма {wave_iso:.1f}°C — участок для анализа волн "
    f"({(analysis_end - analysis_start - 1) * dt / 3600:.1f} ч)"
)
ax_best.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M"))
ax_best.grid(True, alpha=0.3)
ax_best.legend(fontsize=9, loc="best")
fig_best.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig09.png"), dpi=150)
plt.close("all")

# Фиксированное окно короткого участка (part4, st4_fig12).
SHORT_SURGE_START = pd.to_datetime("2023-06-20 23:30")
SHORT_SURGE_END = pd.to_datetime("2023-06-21 01:00")
print(
    f"\nОкно короткого участка (цуг): "
    f"{SHORT_SURGE_START:%d.%m.%Y %H:%M} — {SHORT_SURGE_END:%d.%m.%Y %H:%M}"
)

