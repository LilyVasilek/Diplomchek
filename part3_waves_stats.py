# -*- coding: utf-8 -*-
# =========================================================
# 13. АНАЛИЗ ВОЛН НА ВЫБРАННОЙ ИЗОТЕРМЕ (wave_iso, непрерывный участок)
#     Статистика короткопериодных волн + PSD повторяемых изотерм
# =========================================================
t_win = t_segment
z_best_win = z_segment
# Смещаем относительно среднего по всему непрерывному участку.
z_best_shift = z_best_win - np.nanmean(z_best_win)

print(f"\nИнтервал анализа: {t_win[0].strftime('%d.%m.%Y %H:%M')} — {t_win[-1].strftime('%d.%m.%Y %H:%M')}")


def _grouped_moments_stats(values, n_bins):
    """Статистика группированных данных через центры интервалов и частоты."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None

    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = int(np.sum(counts))
    if n == 0:
        return None

    fi = counts.astype(float)
    c = centers

    mean = np.sum(c * fi) / n
    var = np.sum(fi * (c - mean) ** 2) / n
    std = np.sqrt(var)
    mad = np.sum(fi * np.abs(c - mean)) / n

    m1 = np.sum(fi * (c ** 1)) / n
    m2 = np.sum(fi * (c ** 2)) / n
    m3 = np.sum(fi * (c ** 3)) / n
    m4 = np.sum(fi * (c ** 4)) / n

    mu2 = np.sum(fi * ((c - mean) ** 2)) / n
    mu3 = np.sum(fi * ((c - mean) ** 3)) / n
    mu4 = np.sum(fi * ((c - mean) ** 4)) / n

    if mu2 > 0:
        skew = mu3 / (mu2 ** 1.5)
        kurt = (mu4 / (mu2 ** 2)) - 3.0
    else:
        skew = np.nan
        kurt = np.nan

    return {
        "mean": mean,
        "mad": mad,
        "var": var,
        "std": std,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
        "mu2": mu2,
        "mu3": mu3,
        "mu4": mu4,
        "skew": skew,
        "kurt": kurt,
    }


def _print_grouped_stats_block(values, n_bins, name, unit):
    s = _grouped_moments_stats(values, n_bins=n_bins)
    if s is None:
        print(f"  {name}: недостаточно данных для группированной статистики.")
        return
    unit2 = f"{unit}²" if unit else ""
    print(f"  {name}:")
    print(f"    Выборочное среднее (математическое ожидание): {s['mean']:.4f} {unit}")
    print(f"    Среднее отклонение: {s['mad']:.4f} {unit}")
    print(f"    Выборочная дисперсия: {s['var']:.6f} {unit2}")
    print(f"    Среднеквадратичное отклонение: {s['std']:.4f} {unit}")
    print(
        "    Начальные моменты: "
        f"m1={s['m1']:.6f}, m2={s['m2']:.6f}, m3={s['m3']:.6f}, m4={s['m4']:.6f}"
    )
    print(
        "    Центральные моменты: "
        f"μ2={s['mu2']:.6f}, μ3={s['mu3']:.6f}, μ4={s['mu4']:.6f}"
    )
    print(f"    Коэффициент асимметрии: {s['skew']:.6f}")
    print(f"    Коэффициент эксцесса: {s['kurt']:.6f}")


def print_wave_statistics(H, Tm, title, n_bins):
    """Печать статистики группированных данных по формулам через (c_i, f_i)."""
    print(f"\nСводная статистика ({title}):")
    print(f"  Диапазон высот: {np.min(H):.2f}–{np.max(H):.2f} м")
    print(f"  Диапазон периодов: {np.min(Tm):.2f}–{np.max(Tm):.2f} мин")
    print(f"  Число интервалов (Стерджесс): k={n_bins}")
    _print_grouped_stats_block(H, n_bins=n_bins, name="Высота H", unit="м")
    _print_grouped_stats_block(Tm, n_bins=n_bins, name="Период T", unit="мин")


def _kde_density(values, x_grid):
    """Непараметрическая оценка плотности (ядро Гаусса, bandwidth по правилу Скотта)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.ptp(x) <= 0:
        return None
    try:
        return stats.gaussian_kde(x)(x_grid)
    except Exception:
        return None


def _frozen_from_fit(dist, fit_params):
    if fit_params is None or not np.all(np.isfinite(fit_params)):
        return None
    *shapes, loc, scale = fit_params
    return dist(*shapes, loc=loc, scale=scale)


def _fit_positive(dist, x, **fit_kw):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 5:
        return None
    try:
        par = dist.fit(x, **fit_kw)
        if not np.all(np.isfinite(par)):
            return None
        return par
    except Exception:
        return None


def _fit_sample(dist, x, min_n=5, **fit_kw):
    """MLE-подгонка; для трёхпараметрических законов — все параметры свободны."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < min_n:
        return None
    try:
        par = dist.fit(x, **fit_kw)
        if not np.all(np.isfinite(par)):
            return None
        return par
    except Exception:
        return None


def _sturges_k(n):
    """Число интервалов по Стерджессу (как в m.py)."""
    return int(np.ceil(1 + np.log2(n))) if n > 1 else 1


def _histogram_sturges(x, k_init):
    """Равные интервалы h=(max-min)/k на [min, max], k по Стерджессу (np.histogram)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    k0 = _sturges_k(len(x)) if k_init is None else max(1, int(k_init))
    counts, edges = np.histogram(x, bins=k0)
    return counts, edges, k0


def _combine_bins_mpy(observed, edges, expected, min_exp=5.0):
    """
    Объединение интервалов как в m.py: слева направо накапливаем n_i и E_i,
    закрываем группу, когда суммарная теоретическая частота E_i >= min_exp;
    остаток добавляется к последней группе.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    edges = np.asarray(edges, dtype=float)

    groups_obs, groups_exp, groups_lo, groups_hi = [], [], [], []
    obs_sum = 0.0
    exp_sum = 0.0
    start = float(edges[0])

    for i in range(len(observed)):
        obs_sum += observed[i]
        exp_sum += expected[i]
        if exp_sum >= min_exp:
            groups_obs.append(obs_sum)
            groups_exp.append(exp_sum)
            groups_lo.append(start)
            groups_hi.append(float(edges[i + 1]))
            obs_sum = 0.0
            exp_sum = 0.0
            if i + 1 < len(edges) - 1:
                start = float(edges[i + 1])

    if exp_sum > 0:
        if groups_obs:
            groups_obs[-1] += obs_sum
            groups_exp[-1] += exp_sum
            groups_hi[-1] = float(edges[-1])
        else:
            groups_obs.append(obs_sum)
            groups_exp.append(exp_sum)
            groups_lo.append(start)
            groups_hi.append(float(edges[-1]))

    new_edges = [groups_lo[0]]
    for hi in groups_hi:
        new_edges.append(hi)
    return np.array(groups_obs), np.array(new_edges), np.array(groups_exp)


def _bin_probabilities(frozen, edges):
    """p_i = F(e_{i+1}) - F(e_i) на равных интервалах; нормировка Σp_i = 1."""
    p = frozen.cdf(edges[1:]) - frozen.cdf(edges[:-1])
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s <= 0:
        return None
    return p / s


def _expected_bin_counts_raw(frozen, edges, n):
    """E_i = n·(F(e_{i+1}) - F(e_i)) на исходных равных интервалах (как в m.py)."""
    p = frozen.cdf(edges[1:]) - frozen.cdf(edges[:-1])
    return n * np.clip(p, 0.0, None)


def _n_estimated_params(fit_params, fixed_loc=False):
    """Число оценённых параметров (при floc фиксирован — на один меньше)."""
    if fit_params is None:
        return 0
    if fixed_loc:
        return max(0, len(fit_params) - 1)
    return len(fit_params)


def chi2_gof_test(values, dist, fit_kw, k_init, alpha=0.05, min_exp=5.0):
    """
    Критерий согласия χ²: ρ = Σ (n_i - E_i)² / E_i,  E_i = n·p_i.
    k₀ по Стерджессу, равные Δ₀; объединение интервалов как в m.py (ΣE_i >= min_exp).
    Степени свободы: ν = k - m - 1.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 5:
        return None

    fit_params = _fit_sample(dist, x, **fit_kw)
    frozen = _frozen_from_fit(dist, fit_params)
    if frozen is None:
        return None

    m = _n_estimated_params(fit_params, fixed_loc=("floc" in fit_kw))
    counts0, edges0, k0 = _histogram_sturges(x, k_init)
    n = int(np.sum(counts0))
    exp0 = _expected_bin_counts_raw(frozen, edges0, n)
    n_obs, edges, exp = _combine_bins_mpy(counts0, edges0, exp0, min_exp=min_exp)
    k = len(n_obs)
    p = exp / n if n > 0 else None
    widths = np.diff(edges)
    df = k - m - 1

    if exp is None or df <= 0 or np.any(exp <= 0):
        return {
            "chi2": np.nan,
            "df": df,
            "crit": np.nan,
            "p_value": np.nan,
            "reject": None,
            "k": k,
            "k0": k0,
            "k_merged": k,
            "h0": float((edges0[-1] - edges0[0]) / k0) if k0 > 0 else np.nan,
            "m": m,
            "n": n,
            "counts": n_obs,
            "expected": exp,
            "p": p,
            "edges": edges,
            "widths": widths,
            "fit_params": fit_params,
        }

    chi2 = float(np.sum((n_obs - exp) ** 2 / exp))
    crit = float(stats.chi2.ppf(1.0 - alpha, df))
    p_value = float(stats.chi2.sf(chi2, df))
    return {
        "chi2": chi2,
        "df": df,
        "crit": crit,
        "p_value": p_value,
        "reject": chi2 > crit,
        "k": k,
        "k0": k0,
        "k_merged": k,
        "h0": float((edges0[-1] - edges0[0]) / k0) if k0 > 0 else np.nan,
        "m": m,
        "n": n,
        "counts": n_obs,
        "expected": exp,
        "p": p,
        "edges": edges,
        "widths": widths,
        "fit_params": fit_params,
    }


_CHI2_GOF_DISTRIBUTIONS = (
    ("Логнормальное", stats.lognorm, {"floc": 0}),
    ("Обр. гаусс. (Вальд)", stats.invgauss, {"floc": 0}),
    ("Обратная гамма", stats.invgamma, {"floc": 0}),
)


def _format_p_value(p):
    if not np.isfinite(p):
        return "—"
    if p == 0.0 or p < 1e-6:
        return f"{p:.2e}"
    return f"{p:.6f}"


def _print_chi2_table(result, var_name, unit):
    if result is None:
        print(f"  {var_name}: недостаточно данных.")
        return
    edges = result["edges"]
    counts = result["counts"]
    exp = result["expected"]
    p = result.get("p")
    widths = result.get("widths")
    nu = result["df"]
    k_merged = result.get("k_merged", result["k"])
    h0 = result.get("h0", np.nan)
    print(
        f"    n={result['n']}, k₀={result['k0']} (равные Δ₀"
        + (f"={h0:.4f} {unit}" if np.isfinite(h0) else "")
        + f") → k={k_merged} после объединения (E_i≥5), "
        f"m={result['m']} оценённых параметров"
    )
    print(f"    Степени свободы: ν = k - m - 1 = {result['k']} - {result['m']} - 1 = {nu}")
    if widths is not None and len(widths) > 0:
        w_uniq = np.unique(np.round(widths, 8))
        if len(w_uniq) == 1:
            print(f"    Ширина объединённых интервалов Δ = {w_uniq[0]:.4f} {unit}")
        else:
            print(
                f"    Ширины после объединения: min={widths.min():.4f}, "
                f"max={widths.max():.4f} {unit} (кратные Δ₀)"
            )
    if np.isfinite(result["chi2"]) and np.isfinite(result["crit"]):
        print(
            f"    ρ(χ²) = {result['chi2']:.4f},  "
            f"χ²_крит(ν={nu}, α=0.05) = {result['crit']:.4f},  "
            f"p = {_format_p_value(result['p_value'])}"
        )
        if result["p_value"] == 0.0 and result["chi2"] > result["crit"]:
            print(
                "    (p ≈ 0: ρ сильно больше χ²_крит — расхождение с законом статистически значимо, "
                "не ошибка расчёта)"
            )
    if result["reject"] is None:
        print("    Тест не применим (ν ≤ 0 или E_i ≤ 0).")
    elif result["reject"]:
        print("    H₀ отвергается (ρ > χ²_крит).")
    else:
        print("    H₀ не отвергается (ρ ≤ χ²_крит).")
    print(
        f"    {'Интервал':<24} {'Δ':>7} {'n_i':>5} {'p_i':>9} {'E_i=n·p_i':>10} {'вклад':>10}"
    )
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        wi = widths[i] if widths is not None else (hi - lo)
        pi = p[i] if p is not None else np.nan
        ei = exp[i] if exp is not None else np.nan
        contrib = (counts[i] - ei) ** 2 / ei if np.isfinite(ei) and ei > 0 else np.nan
        print(
            f"    [{lo:6.3f}, {hi:6.3f}) {unit} "
            f"{wi:7.4f} {int(counts[i]):5d} {pi:9.5f} {ei:10.3f} {contrib:10.4f}"
        )
    if exp is not None:
        print(f"    Σn_i = {int(np.sum(counts))},  ΣE_i = {np.sum(exp):.3f}")


def print_wave_chi2_tests(H, Tm, k_sturges, alpha=0.05):
    """χ²-критерий для логнормального, обратного гауссовского и обратной гаммы."""
    print(
        f"\n=== Критерий согласия χ² (α={alpha:.2f}, как m.py: "
        f"k₀ Стерджесс, равные Δ₀, объединение при ΣE_i≥5) ==="
    )
    for var_name, values, unit in (
        ("Высота H", np.asarray(H, dtype=float), "м"),
        ("Период T", np.asarray(Tm, dtype=float), "мин"),
    ):
        print(f"\n--- {var_name} ---")
        for name, dist, fkw in _CHI2_GOF_DISTRIBUTIONS:
            res = chi2_gof_test(values, dist, fkw, k_init=k_sturges, alpha=alpha)
            print(f"  {name}:")
            if res is None:
                print("    подгонка/данные недоступны.")
                continue
            _print_chi2_table(res, var_name, unit)


def _hist_edges_chi2_mpy(values, dist, fit_kw, k_init, min_exp=5.0):
    """Границы интервалов после объединения как в m.py (для данного закона)."""
    res = chi2_gof_test(values, dist, fit_kw, k_init=k_init, min_exp=min_exp)
    if res is not None and res.get("edges") is not None:
        return res["edges"]
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    _, edges, _ = _histogram_sturges(x, k_init)
    return edges


def _plot_hist_with_kde(ax, values, edges, color, kde_color, xlabel, title):
    ax.hist(
        values,
        bins=edges,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color=color,
        label="Гистограмма",
    )
    x_valid = np.asarray(values, dtype=float)
    x_valid = x_valid[np.isfinite(x_valid)]
    if x_valid.size >= 2 and np.ptp(x_valid) > 0:
        lo, hi = float(np.min(x_valid)), float(np.max(x_valid))
        x_grid = np.linspace(lo, hi, 256)
        kde_y = _kde_density(values, x_grid)
        if kde_y is not None:
            ax.plot(
                x_grid,
                kde_y,
                color=kde_color,
                lw=2.0,
                label="KDE (ядро Гаусса)",
            )
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Плотность")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def _sturges_edges(values, k_sturges):
    n = len(values)
    k = k_sturges if k_sturges is not None else max(1, int(np.ceil(1 + np.log2(n))))
    return np.histogram_bin_edges(values, bins=k), k


def _hist_bin_centers(values, edges):
    """Центры интервалов x_p, выборочные средние в интервале и число точек."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    edges = np.asarray(edges, dtype=float)
    xp, means, counts = [], [], []
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (values >= lo) & (values < hi)
        else:
            mask = (values >= lo) & (values <= hi)
        if np.any(mask):
            xp.append(0.5 * (lo + hi))
            means.append(float(np.mean(values[mask])))
            counts.append(int(np.sum(mask)))
    return (
        np.asarray(xp, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(counts, dtype=int),
    )


def _regression_mean_response_ci(x, y, x_eval, alpha=0.05):
    """
    ДИ 95% для M(y_p) по средним в интервалах гистограммы:
    регрессия средних по центрам интервалов (не Y=X по всем волнам — иначе s^2=0).
    s_{ŷ}^2 = (s^2/n)(1 + (x_p-x̄)^2/Var(x)), n — число интервалов.
    """
    x_eval = np.asarray(x_eval, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3 or x_eval.size == 0:
        return None

    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    if var_x <= 1e-12:
        return None

    ss_xx = float(np.sum((x - x_bar) ** 2))
    b1 = float(np.sum((x - x_bar) * (y - y_bar)) / ss_xx)
    b0 = y_bar - b1 * x_bar
    y_hat_obs = b0 + b1 * x
    sse = float(np.sum((y - y_hat_obs) ** 2))
    df = n - 2
    if df <= 0:
        return None
    s2 = sse / df

    y_hat_p = b0 + b1 * x_eval
    se = np.sqrt(np.maximum(s2 / n * (1.0 + (x_eval - x_bar) ** 2 / var_x), 0.0))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    lower = y_hat_p - t_crit * se
    upper = y_hat_p + t_crit * se
    return {
        "y_hat": y_hat_p,
        "lower": lower,
        "upper": upper,
        "se": se,
        "s2": s2,
        "n": n,
        "df": df,
        "t_crit": t_crit,
        "b0": b0,
        "b1": b1,
        "x_bar": x_bar,
    }


def _mean_ci_table(values, var_name, unit, k_sturges, alpha=0.05):
    """Таблица ДИ для M(Y) в центрах интервалов гистограммы (Стерджесс)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    edges, k = _sturges_edges(values, k_sturges)
    centers, bin_means, counts = _hist_bin_centers(values, edges)
    ci = _regression_mean_response_ci(centers, bin_means, centers, alpha=alpha)
    if ci is None:
        return None
    rows = []
    for i in range(len(centers)):
        rows.append({
            "Центр интервала x_p": centers[i],
            "Число волн n_p": counts[i],
            "Среднее в интервале": bin_means[i],
            "Прогноз y_p": ci["y_hat"][i],
            "s(y_p)": ci["se"][i],
            "Нижняя граница ДИ": ci["lower"][i],
            "Верхняя граница ДИ": ci["upper"][i],
        })
    return {
        "var_name": var_name,
        "unit": unit,
        "alpha": alpha,
        "k": k,
        "edges": edges,
        "centers": centers,
        "bin_means": bin_means,
        "counts": counts,
        "ci": ci,
        "df": pd.DataFrame(rows),
    }


def print_wave_mean_ci_report(table_h, table_t):
    """Структурированный вывод ДИ для M(Y) по интервалам Стерджесса (не по кластерам)."""
    print("\n" + "=" * 72)
    print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ СРЕДНЕГО M(Y) В ЦЕНТРАХ ИНТЕРВАЛОВ ГИСТОГРАММЫ (Стерджесс)")
    print("=" * 72)
    print(
        "Формула (95%):  y_p - t*s(y_p) <= M(Y) <= y_p + t*s(y_p),\n"
        "          s(y_p)^2 = (s^2/n) * (1 + (x_p - x_mean)^2 / Var(x)),\n"
        "  Регрессия по точкам (центр интервала, среднее в интервале); "
        "s^2 — разброс средних относительно линии."
    )
    for pack in (table_h, table_t):
        if pack is None:
            continue
        ci = pack["ci"]
        print("\n" + "-" * 72)
        print(f"Переменная: {pack['var_name']} ({pack['unit']})")
        print(f"  Доверительная вероятность: {100 * (1 - pack['alpha']):.0f}% (alpha = {pack['alpha']:.2f})")
        print(f"  Число волн n = {ci['n']}, степени свободы nu = {ci['df']}")
        print(f"  Число интервалов Стерджесса k = {pack['k']}")
        print(
            f"  Регрессия Y = {ci['b0']:.4f} + {ci['b1']:.4f} * X,  "
            f"остаточная s^2 = {ci['s2']:.6f} ({pack['unit']})^2,  "
            f"t_{{alpha/2}} = {ci['t_crit']:.4f}"
        )
        print(f"  Среднее X: x_mean = {ci['x_bar']:.4f} {pack['unit']}")
        print(pack["df"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _plot_mean_ci_panel(ax, pack, xlabel, panel_title):
    """Панель: линия регрессии, точки-данные, доверительная трубка 95%."""
    if pack is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center")
        return
    centers = pack["centers"]
    ci = pack["ci"]
    bin_means = pack["bin_means"]
    x_line = np.linspace(float(pack["edges"][0]), float(pack["edges"][-1]), 200)
    y_line = ci["b0"] + ci["b1"] * x_line
    y_lo = ci["b0"] + ci["b1"] * x_line - ci["t_crit"] * np.sqrt(
        np.maximum(ci["s2"] / ci["n"] * (1.0 + (x_line - ci["x_bar"]) ** 2 / np.var(centers, ddof=1)), 0.0)
    )
    y_hi = ci["b0"] + ci["b1"] * x_line + ci["t_crit"] * np.sqrt(
        np.maximum(ci["s2"] / ci["n"] * (1.0 + (x_line - ci["x_bar"]) ** 2 / np.var(centers, ddof=1)), 0.0)
    )
    ax.fill_between(x_line, y_lo, y_hi, color="crimson", alpha=0.25, label="ДИ 95%")
    ax.plot(x_line, y_line, color="crimson", lw=2.0, label="Линия регрессии", zorder=3)
    ax.scatter(
        centers,
        bin_means,
        s=65,
        c="steelblue",
        edgecolors="white",
        linewidths=1.0,
        label="Среднее в интервале",
        zorder=4,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Среднее, {pack['unit']}")
    ax.set_title(panel_title, fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.35)


def plot_wave_mean_ci_figure(H, Tm, title_suffix, out_file, k_sturges=None, alpha=0.05):
    """Отдельный рисунок: только доверительные интервалы для H и T."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = out_file if os.path.isabs(out_file) else os.path.join(base_dir, out_file)
    txt_path = os.path.splitext(out_path)[0] + ".txt"

    table_h = _mean_ci_table(H, "Высота волн H", "м", k_sturges, alpha=alpha)
    table_t = _mean_ci_table(Tm, "Период волн T", "мин", k_sturges, alpha=alpha)

    fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Доверительные интервалы 95% | {title_suffix}\n"
        f"Интервалы гистограммы: Стерджесс",
        fontsize=12,
        y=1.02,
    )
    _plot_mean_ci_panel(
        ax_h,
        table_h,
        xlabel="Центр интервала x_p, м",
        panel_title="Высота H",
    )
    _plot_mean_ci_panel(
        ax_t,
        table_t,
        xlabel="Центр интервала x_p, мин",
        panel_title="Период T",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    buf_lines = [
        "Доверительные интервалы для среднего M(Y) в центрах интервалов гистограммы",
        f"Участок: {title_suffix}",
        "",
        "Формула: y_p - t*s(y_p) <= M(Y) <= y_p + t*s(y_p)",
        "         s(y_p)^2 = (s^2/n) * (1 + (x_p - x_mean)^2 / Var(x))",
    ]
    for pack in (table_h, table_t):
        if pack is None:
            continue
        ci = pack["ci"]
        buf_lines.append("\n" + "-" * 60)
        buf_lines.append(f"Переменная: {pack['var_name']} ({pack['unit']})")
        buf_lines.append(f"alpha={pack['alpha']}, n={ci['n']}, nu={ci['df']}, k={pack['k']}")
        buf_lines.append(f"Y = {ci['b0']:.4f} + {ci['b1']:.4f}*X, s^2={ci['s2']:.6f}")
        buf_lines.append(pack["df"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(buf_lines))
    print(f"\nДоверительные интервалы: график {out_path}")
    print(f"  Текстовый отчет: {txt_path}")
    return table_h, table_t


def _plot_hist_with_pdf(ax, values, edges, frozen, color, pdf_color, xlabel, title):
    ax.hist(
        values,
        bins=edges,
        density=True,
        alpha=0.7,
        edgecolor="black",
        color=color,
        label="Гистограмма",
    )
    if frozen is not None:
        x_grid = np.linspace(float(edges[0]), float(edges[-1]), 300)
        pdf_y = frozen.pdf(x_grid)
        m = np.isfinite(pdf_y) & (pdf_y > 0)
        if np.any(m):
            ax.plot(
                x_grid[m],
                pdf_y[m],
                color=pdf_color,
                lw=2.0,
                label="Подогнанная PDF",
            )
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Плотность")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_wave_histograms_kde(H, Tm, title_suffix, out_file, k_sturges=None):
    """Гистограммы H и T (Стерджесс) + непараметрическая KDE."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = out_file if os.path.isabs(out_file) else os.path.join(base_dir, out_file)

    fig_hist, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(13, 5))
    edges_h, _ = _sturges_edges(H, k_sturges)
    edges_t, _ = _sturges_edges(Tm, k_sturges)

    _plot_hist_with_kde(
        ax_h,
        H,
        edges_h,
        color="steelblue",
        kde_color="darkred",
        xlabel="Высота H, м",
        title=f"Гистограмма высот волн ({title_suffix})",
    )
    _plot_hist_with_kde(
        ax_t,
        Tm,
        edges_t,
        color="seagreen",
        kde_color="darkred",
        xlabel="Период T, мин",
        title=f"Гистограмма периодов волн ({title_suffix})",
    )

    fig_hist.tight_layout()
    fig_hist.savefig(out_path, dpi=200)
    fig_hist.savefig(os.path.join(base_dir, "st4_fig10.png"), dpi=150)
    print(f"\nГистограммы (KDE) сохранены: {out_path}, st4_fig10.png")
    plt.close(fig_hist)


def plot_wave_histograms_fitted(H, Tm, title_suffix, out_file, k_sturges=None):
    """Одно окно: гистограммы H и T с PDF проверяемых законов (χ²-тест)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = out_file if os.path.isabs(out_file) else os.path.join(base_dir, out_file)

    H = np.asarray(H, dtype=float)
    Tm = np.asarray(Tm, dtype=float)

    n_dist = len(_CHI2_GOF_DISTRIBUTIONS)
    fig, axes = plt.subplots(
        2, n_dist, figsize=(4.2 * n_dist, 7), squeeze=False,
    )
    fig.suptitle(
        f"Гистограммы и подогнанные распределения ({title_suffix})\n"
        f"k₀ Стерджесс, объединение интервалов при ΣE_i ≥ 5 (как m.py)",
        fontsize=11,
        y=1.01,
    )

    panels = (
        (H, "steelblue", "H", "Высота H, м"),
        (Tm, "seagreen", "T", "Период T, мин"),
    )
    for row, (values, color, row_lbl, xlab) in enumerate(panels):
        x_pos = values[np.isfinite(values) & (values > 0)]
        for col, (dist_name, dist, fkw) in enumerate(_CHI2_GOF_DISTRIBUTIONS):
            ax = axes[row, col]
            frozen = _frozen_from_fit(dist, _fit_positive(dist, x_pos, **fkw))
            res = chi2_gof_test(values, dist, fkw, k_init=k_sturges)
            edges = res["edges"] if res is not None else _hist_edges_chi2_mpy(
                values, dist, fkw, k_sturges
            )
            if res is not None and np.isfinite(res.get("chi2", np.nan)):
                title = (
                    f"{dist_name}\n"
                    f"ρ={res['chi2']:.2f}, ν={res['df']}, "
                    f"p={_format_p_value(res['p_value'])}"
                )
            else:
                title = dist_name
            _plot_hist_with_pdf(
                ax,
                values,
                edges,
                frozen,
                color=color,
                pdf_color="darkred",
                xlabel=xlab,
                title=title,
            )
            if col == 0:
                ax.set_ylabel(f"{row_lbl}\nПлотность")
            else:
                ax.set_ylabel("")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(base_dir, "st4_fig11.png"), dpi=150, bbox_inches="tight")
    print(f"Гистограммы (подогнанные законы) сохранены: {out_path}, st4_fig11.png")
    plt.close(fig)


# Число кластеров при иерархической классификации.
WAVE_N_CLUSTERS = 4
HIER_LINKAGE_METHOD = "ward"


def _euclidean_dist(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.sum((x - y) ** 2)))


def _cluster_centroid(x_group):
    x_group = np.asarray(x_group, dtype=float)
    if x_group.ndim == 1:
        return x_group.copy()
    return np.mean(x_group, axis=0)


def _hierarchical_labels(x, n_clusters, method=HIER_LINKAGE_METHOD):
    """Иерархическая кластеризация: linkage + fcluster (maxclust)."""
    from scipy.cluster.hierarchy import linkage, fcluster

    x = np.asarray(x, dtype=float)
    xs = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    z_link = linkage(xs, method=method)
    lab = fcluster(z_link, t=n_clusters, criterion="maxclust")
    return (lab - 1).astype(int), z_link


def _print_hierarchical_cluster_stats(centers, k):
    print("  Центры кластеров (среднее H и T):")
    for j in range(k):
        print(f"    Кластер {j + 1}: H={centers[j, 0]:.3f} м, T={centers[j, 1]:.3f} мин")
    print("  Расстояния между центрами:")
    hdr = "         " + "".join(f"{j + 1:>10d}" for j in range(k))
    print(hdr)
    for i in range(k):
        row = f"    {i + 1:>3d}   "
        for j in range(k):
            row += f"{_euclidean_dist(centers[i], centers[j]):10.4f}"
        print(row)


def _cluster_mean_ci(mean_val, values, alpha=0.05):
    """Классическое 95% ДИ для среднего в кластере (t-распределение, n точек)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return mean_val, np.nan, mean_val, mean_val
    s = float(np.std(values, ddof=1))
    se = s / np.sqrt(n)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    return mean_val, se, mean_val - t_crit * se, mean_val + t_crit * se


def build_cluster_summary_table(heights_m, periods_min, labels, alpha=0.05):
    """
    Сводка по кластерам: число волн и средние H, T с ДИ для среднего (не Стерджесс).
    Отдельно от таблицы ДИ по интервалам гистограммы.
    """
    heights_m = np.asarray(heights_m, dtype=float)
    periods_min = np.asarray(periods_min, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        n = int(np.sum(mask))
        if n == 0:
            continue
        h_vals = heights_m[mask]
        t_vals = periods_min[mask]
        h_mean = float(np.mean(h_vals))
        t_mean = float(np.mean(t_vals))
        h_mean, se_h, h_lo, h_hi = _cluster_mean_ci(h_mean, h_vals, alpha=alpha)
        t_mean, se_t, t_lo, t_hi = _cluster_mean_ci(t_mean, t_vals, alpha=alpha)
        rows.append({
            "Кластер": int(c) + 1,
            "Число волн n": n,
            "Средняя высота H, м": h_mean,
            "Средний период T, мин": t_mean,
            "s(H̄)": se_h,
            "Нижняя граница ДИ H": h_lo,
            "Верхняя граница ДИ H": h_hi,
            "s(T̄)": se_t,
            "Нижняя граница ДИ T": t_lo,
            "Верхняя граница ДИ T": t_hi,
        })
    return pd.DataFrame(rows)


def print_cluster_summary_table(df_clusters, alpha=0.05, out_txt=None):
    """Печать сводки по кластерам (не путать с ДИ по интервалам гистограммы)."""
    print("\n" + "=" * 72)
    print("СВОДКА ПО КЛАСТЕРАМ ВОЛН")
    print("=" * 72)
    print(
        f"ДИ {100 * (1 - alpha):.0f}% для среднего в кластере: "
        "x̄ ± t·s/√n (t-распределение, не регрессия по интервалам Стерджесса)."
    )
    fmt = lambda x: f"{x:.4f}"
    print(df_clusters.to_string(index=False, float_format=fmt))
    if out_txt:
        lines = [
            "СВОДКА ПО КЛАСТЕРАМ ВОЛН",
            f"ДИ {100 * (1 - alpha):.0f}%: t-интервал для среднего в кластере.",
            "",
            df_clusters.to_string(index=False, float_format=fmt),
        ]
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Текст: {out_txt}")


def cluster_and_plot_waves(
    heights_m,
    periods_min,
    title_suffix,
    out_file,
    n_clusters=WAVE_N_CLUSTERS,
):
    """Иерархическая кластеризация волн (H, T), K кластеров."""
    n = len(heights_m)
    if n < n_clusters:
        print(
            f"  Кластеризация: n={n} < {n_clusters}, пропуск "
            f"(нужно хотя бы {n_clusters} волн)."
        )
        return None

    x = np.column_stack([heights_m, periods_min])
    k = n_clusters
    labels, z_link = _hierarchical_labels(x, k)

    centers = np.zeros((k, 2), dtype=float)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            centers[c] = _cluster_centroid(x[mask])

    print(
        f"\nИерархическая кластеризация волн "
        f"(метод связи: {HIER_LINKAGE_METHOD}, K={k}, признаки H и T):"
    )
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            print(f"  Кластер {c + 1}: нет волн")
            continue
        print(
            f"  Кластер {c + 1}: n={int(np.sum(mask))}, "
            f"H = {heights_m[mask].min():.2f}-{heights_m[mask].max():.2f} м, "
            f"T = {periods_min[mask].min():.1f}-{periods_min[mask].max():.1f} мин"
        )
    _print_hierarchical_cluster_stats(centers, k)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tag = os.path.splitext(os.path.basename(out_file))[0]

    fig_c, ax_c = plt.subplots(figsize=(8, 6))
    cmap_c = plt.cm.get_cmap("tab10", k)
    for c in range(k):
        mask = labels == c
        if not np.any(mask):
            continue
        ax_c.scatter(
            heights_m[mask],
            periods_min[mask],
            s=55,
            c=[cmap_c(c)],
            label=f"Кластер {c + 1} (n={int(np.sum(mask))})",
            edgecolors="k",
            linewidths=0.4,
        )
    ax_c.scatter(
        centers[:, 0],
        centers[:, 1],
        s=180,
        c="black",
        marker="X",
        linewidths=1.5,
        label="Центр кластера",
        zorder=10,
    )
    ax_c.set_xlabel("Амплитуда (высота H), м")
    ax_c.set_ylabel("Период T, мин")
    ax_c.set_title(
        f"Кластеры волн по амплитуде и периоду (K={k}, Ward)\n{title_suffix}",
        fontsize=10,
    )
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="best", fontsize=9)
    fig_c.tight_layout()
    out_path = os.path.join(base_dir, out_file)
    fig_c.savefig(out_path, dpi=150)
    plt.close(fig_c)

    print(f"  График кластеров: {out_file}")

    df_clusters = build_cluster_summary_table(heights_m, periods_min, labels)
    txt_path = os.path.join(base_dir, f"{tag}_summary.txt")
    print_cluster_summary_table(df_clusters, alpha=0.05, out_txt=txt_path)
    csv_path = os.path.join(base_dir, f"{tag}_summary.csv")
    df_clusters.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {os.path.basename(csv_path)}")
    return labels, df_clusters


# Волны на выбранной изотерме wave_iso (не обязательно «лучшей» 15.5 °C).
# Ищем все локальные минимумы, затем фильтруем по T >= 3 мин и h >= WAVE_MIN_HEIGHT_M.
selected_waves = detect_waves(z_best_shift, dt_seconds=dt, min_period_min=3.0, min_height_m=WAVE_MIN_HEIGHT_M)

print(
    f"Найдено волн на изотерме {wave_iso:.1f}°C "
    f"(h >= {WAVE_MIN_HEIGHT_M} м, T >= 3 мин): {len(selected_waves)}"
)


def _auto_select_surge_window(
    waves,
    time_arr,
    *,
    min_hours=1.0,
    max_hours=2.0,
    top_n=3,
    duration_step_min=10,
):
    """Автовыбор окна цуга: 1–2 ч, максимум топ-волн по высоте."""
    if not waves:
        return None
    t = pd.to_datetime(time_arr)
    if len(t) < 2:
        return None

    rec = []
    for i0, i1, _imax, h_wave, _period in waves:
        if i0 < 0 or i1 >= len(t) or i1 <= i0:
            continue
        rec.append(
            {
                "t0": t[i0],
                "t1": t[i1],
                "h": float(h_wave),
            }
        )
    if len(rec) < 3:
        return None

    # Номера top_n самых высоких волн.
    heights = np.array([r["h"] for r in rec], dtype=float)
    top_idx = set(np.argsort(heights)[::-1][: min(top_n, len(rec))].tolist())

    # Кандидаты стартов: начала/концы волн + равномерная сетка.
    starts = {r["t0"] for r in rec} | {r["t1"] for r in rec}
    step = pd.Timedelta(minutes=duration_step_min)
    t_lo = t[0]
    t_hi = t[-1]
    cur = t_lo
    while cur <= t_hi:
        starts.add(cur)
        cur += step

    best = None
    best_relaxed = None
    for ts in sorted(starts):
        for dur_min in range(int(min_hours * 60), int(max_hours * 60) + 1, duration_step_min):
            te = ts + pd.Timedelta(minutes=dur_min)
            if te > t_hi:
                continue
            idx_inside = [
                k for k, r in enumerate(rec)
                if (r["t0"] >= ts and r["t1"] <= te)
            ]
            if not idx_inside:
                continue
            n_top = sum(1 for k in idx_inside if k in top_idx)
            n_all = len(idx_inside)
            h_sum = float(np.sum([rec[k]["h"] for k in idx_inside]))
            # Приоритет: топ-волны -> число волн -> сумма высот -> короче окно.
            score = (n_top, n_all, h_sum, -dur_min)
            cand = {"start": ts, "end": te, "score": score, "n_top": n_top, "n_all": n_all}
            if (best is None) or (score > best["score"]):
                best = cand
            # Более мягкий fallback: минимум 3 волны в окне (даже если не все из топ-3).
            if n_all >= 3 and ((best_relaxed is None) or (score > best_relaxed["score"])):
                best_relaxed = cand

    if best is None:
        return None
    # Строгое требование: от трёх самых высоких волн в окне.
    if best["n_top"] < min(3, len(top_idx)):
        if best_relaxed is None:
            return None
        best_relaxed["is_relaxed"] = True
        return best_relaxed
    best["is_relaxed"] = False
    return best


# Фиксированное окно цуга для fig12 и кросс-корреляции (21.06.2023).
SHORT_SURGE_START = pd.to_datetime("2023-06-21 21:20")
SHORT_SURGE_END = pd.to_datetime("2023-06-21 23:20")
print(
    f"Окно цуга (фиксированное): {SHORT_SURGE_START:%d.%m.%Y %H:%M} — "
    f"{SHORT_SURGE_END:%d.%m.%Y %H:%M}"
)

_auto_surge = _auto_select_surge_window(selected_waves, t_win, min_hours=1.0, max_hours=2.0, top_n=3)
if _auto_surge is not None:
    _mode = "fallback" if _auto_surge.get("is_relaxed", False) else "strict"
    print(
        f"  (справочно, авто-подбор {_mode}): {_auto_surge['start']:%d.%m.%Y %H:%M} — "
        f"{_auto_surge['end']:%d.%m.%Y %H:%M}, "
        f"топ-волн: {_auto_surge['n_top']}, всех: {_auto_surge['n_all']})"
    )
else:
    print("  (справочно: авто-подбор окна 1–2 ч не нашёл кандидата с >=3 топ-волнами)")

rows = []
for n, (i0, i1, _imax, h_wave, period_min) in enumerate(selected_waves, start=1):
    rows.append({
        "№": n,
        "Начало": t_win[i0].strftime("%d.%m %H:%M:%S"),
        "Конец": t_win[i1].strftime("%d.%m %H:%M:%S"),
        "Высота H, м": h_wave,
        "Период T, мин": period_min,
    })

if len(rows) > 0:
    df_waves = pd.DataFrame(rows)
    print(f"\nТаблица волн на изотерме {wave_iso:.1f}°C:")
    print(df_waves.to_string(index=False, justify="center", float_format=lambda x: f"{x:.2f}"))

    H = df_waves["Высота H, м"].to_numpy()
    Tm = df_waves["Период T, мин"].to_numpy()
    n_waves = len(df_waves)
    k_st = max(1, int(np.ceil(1 + np.log2(n_waves))))
    print(f"  Стерджесс: n={n_waves}, k={k_st}")
    wave_title = f"изотерма {wave_iso:.1f}°C"
    print_wave_statistics(H, Tm, title=wave_title, n_bins=k_st)
    ci_tables = plot_wave_mean_ci_figure(
        H,
        Tm,
        title_suffix=wave_title,
        out_file=f"waves_{wave_iso_tag}_mean_ci.png",
        k_sturges=k_st,
        alpha=0.05,
    )
    print_wave_mean_ci_report(ci_tables[0], ci_tables[1])
    print_wave_chi2_tests(H, Tm, k_sturges=k_st, alpha=0.05)
    plot_wave_histograms_kde(
        H,
        Tm,
        title_suffix=wave_title,
        out_file=f"waves_{wave_iso_tag}_hist_full_segment.png",
        k_sturges=k_st,
    )
    plot_wave_histograms_fitted(
        H,
        Tm,
        title_suffix=wave_title,
        out_file=f"waves_{wave_iso_tag}_hist_distributions.png",
        k_sturges=k_st,
    )

    cluster_result = cluster_and_plot_waves(
        H,
        Tm,
        title_suffix=wave_title,
        out_file=f"waves_{wave_iso_tag}_clusters.png",
    )
    if cluster_result is not None:
        wave_cluster_labels, _df_clusters = cluster_result
        nums_by_cluster = {}
        for num, lab in enumerate(wave_cluster_labels + 1, start=1):
            nums_by_cluster.setdefault(int(lab), []).append(num)
        print("\nНомера волн по кластерам:")
        for c in sorted(nums_by_cluster):
            print(f"  Кластер {c}: {nums_by_cluster[c]}")
else:
    print(f"Недостаточно волн на изотерме {wave_iso:.1f}°C для статистики и гистограмм.")

# Повторяемость волн по всем изотермам 11–24 °C в том же участке (окно wave_iso).
iso_window_values = np.arange(11.0, 25.0, 1.0)
window_iso_series = {}
for T_iso in iso_window_values:
    window_iso_series[T_iso] = all_iso_depths.get(
        T_iso, np.full(len(time_30s), np.nan)
    )[analysis_start:analysis_end]

iso_repeats = []
for T_iso in iso_window_values:
    z_iso_win = window_iso_series[T_iso]
    support = 0
    for i0, i1, _imax, _h_best, _per in selected_waves:
        if i1 >= len(z_iso_win):
            continue
        seg = z_iso_win[i0:i1 + 1]
        if np.any(np.isnan(seg)):
            continue
        imax_iso = i0 + np.argmax(seg)
        zmax_iso = z_iso_win[imax_iso]
        h_front_iso = zmax_iso - z_iso_win[i0]
        h_rear_iso = zmax_iso - z_iso_win[i1]
        h_iso = 0.5 * (h_front_iso + h_rear_iso)
        if h_iso >= WAVE_MIN_HEIGHT_M:
            support += 1
    iso_repeats.append((T_iso, support))

iso_repeats.sort(key=lambda x: x[1], reverse=True)
top_k = 3
top_isos = [x[0] for x in iso_repeats[:top_k] if x[1] > 0]
print("\nИзотермы с наибольшей повторяемостью волн:")
for T_iso, cnt in iso_repeats[:top_k]:
    print(f"  {T_iso:.1f}°C: {cnt} волн")

# PSD + наклон (лог-лог) для выбранных изотерм.
if len(top_isos) > 0:
    repeat_counts = dict(iso_repeats)
    repeat_psd_blocks = []
    fig_psd, ax_psd = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for T_iso in top_isos:
        z_win = window_iso_series[T_iso]
        if len(z_win) < 8:
            continue
        if np.any(np.isnan(z_win)):
            valid = np.isfinite(z_win)
            if np.sum(valid) < 8:
                continue
            # Линейно заполняем пропуски, чтобы корректно оценить PSD.
            z_win = np.interp(np.arange(len(z_win)), np.where(valid)[0], z_win[valid])
        sig = detrend(z_win, type="linear")
        n_pts = len(sig)
        fa = 1.0 / dt
        f_hz = np.fft.rfftfreq(n_pts, d=dt)
        f_cph = f_hz * 3600.0
        X = np.fft.rfft(sig)
        Pxx = ((1.0 / (n_pts * fa)) * (np.abs(X) ** 2)) / 3600.0

        m = (f_cph > 0) & (Pxx > 0) & np.isfinite(Pxx)
        if np.sum(m) < 3:
            continue

        fv, pv = f_cph[m], Pxx[m]
        log_f = np.log10(fv)
        log_P = np.log10(pv)
        slope, intercept = np.polyfit(log_f, log_P, 1)
        fit = 10 ** (intercept + slope * log_f)

        repeat_psd_blocks.append(
            {
                "T_iso": float(T_iso),
                "repeat_count": int(repeat_counts.get(T_iso, 0)),
                "n_pts": int(n_pts),
                "f_cph": fv.copy(),
                "Pxx": pv.copy(),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )

        ax_psd.loglog(fv, pv, lw=1.2, label=f"PSD {T_iso:.1f}°C")
        plotted_any = True

        ax_psd.loglog(fv, fit, ls="--", lw=1.0, label=f"{T_iso:.1f}°C: наклон={slope:.2f}")
        print(f"  {T_iso:.1f}°C: наклон PSD (лог-лог) = {slope:.2f}")

    _write_top_repeated_isos_psd_txt(
        repeat_psd_blocks,
        dt_s=dt,
        best_iso=float(wave_iso),
        t_start_str=t_win[0].strftime("%d.%m.%Y %H:%M"),
        t_end_str=t_win[-1].strftime("%d.%m.%Y %H:%M"),
        iso_repeats_sorted=iso_repeats,
        top_k=top_k,
    )

    ax_psd.set_xlabel("Частота, цикл/час")
    ax_psd.set_ylabel("PSD, м²·час")
    ax_psd.set_title("PSD для изотерм с максимальной повторяемостью волн")
    ax_psd.grid(True, which="both", alpha=0.3)
    if plotted_any:
        if np.isfinite(N_max_cph) and N_max_cph > 0:
            ax_psd.axvline(
                N_max_cph,
                color="crimson",
                ls="--",
                lw=1.3,
                zorder=6,
                label=f"$N_{{max}}$ (В–Б) = {N_max_cph:.2f} ч⁻¹",
            )
        _add_amp_spectrum_period_reference_lines(ax_psd, t_max_h=17.1)
        ax_psd.legend(fontsize=8, loc="best")
    fig_psd.tight_layout()
    out_psd = "top_repeated_isotherms_psd_full_segment.png"
    fig_psd.savefig(out_psd, dpi=200)
    print(f"График PSD сохранен: {out_psd}")
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st4_fig11.png"), dpi=150)
    plt.close("all")
else:
    print("Нет изотерм с повторяемыми волнами для построения PSD.")

