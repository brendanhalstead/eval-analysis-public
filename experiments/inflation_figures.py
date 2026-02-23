"""
Publication-quality figure: a synthetic agent with known true p50
where METR's logistic demonstrably inflates.

Uses METR's own plotting conventions (log x-axis, logarithmic bins,
weighted success rates, ±2SE error bars).
"""

import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.special import expit

from horizon.utils.logistic import logistic_regression, get_x_for_quantile
from horizon.compute_task_weights import compute_sample_weights
from horizon.utils.plots import format_time_label, logarithmic_ticks, get_logarithmic_bins


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_v1_1_task_scaffold():
    runs = pd.read_json(
        os.path.join(_repo_root, "reports", "time-horizon-1-1", "data", "raw", "runs.jsonl"),
        lines=True, orient="records",
    )
    tasks = runs.drop_duplicates("task_id")[["task_id", "human_minutes", "task_family"]].copy()
    return tasks.sort_values("human_minutes").reset_index(drop=True)


def make_synthetic_agent_df(tasks, outcomes):
    rows = []
    for _, task in tasks.iterrows():
        for run_idx, score in enumerate(outcomes[task["task_id"]]):
            rows.append({
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "human_minutes": task["human_minutes"],
                "score_binarized": float(score),
                "alias": "synthetic",
                "run_id": f"synth_{task['task_id']}_{run_idx}",
            })
    return pd.DataFrame(rows)


def generate_outcomes(tasks, true_fn, n_runs, rng):
    outcomes = {}
    for _, task in tasks.iterrows():
        hm = max(task["human_minutes"], 0.1)
        p = float(np.clip(np.asarray(true_fn(np.log2(hm))).ravel()[0], 0, 1))
        outcomes[task["task_id"]] = rng.binomial(1, p, size=n_runs)
    return outcomes


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def plateau_cliff(log2_t, plateau=0.93, floor=0.07,
                  ramp_start_log2=8.0, ramp_end_log2=np.log2(960)):
    """
    P(success) = plateau for t < 2^ramp_start,
                 linear drop to floor at t = 2^ramp_end.

    True p50 (algebra):
        0.5 = plateau - (plateau - floor) * (x - start) / (end - start)
        x = start + (plateau - 0.5) / (plateau - floor) * (end - start)
    """
    x = np.atleast_1d(log2_t).astype(float)
    result = np.full_like(x, plateau)
    ramp = (x >= ramp_start_log2) & (x <= ramp_end_log2)
    frac = (x[ramp] - ramp_start_log2) / (ramp_end_log2 - ramp_start_log2)
    result[ramp] = plateau - (plateau - floor) * frac
    result[x > ramp_end_log2] = floor
    return result


# True p50 from the specification
PLATEAU, FLOOR = 0.93, 0.07
RAMP_START, RAMP_END = 8.0, np.log2(960)
TRUE_P50_LOG2 = RAMP_START + (PLATEAU - 0.5) / (PLATEAU - FLOOR) * (RAMP_END - RAMP_START)
TRUE_P50 = 2 ** TRUE_P50_LOG2  # minutes


# ---------------------------------------------------------------------------
# Plotting helpers (METR style)
# ---------------------------------------------------------------------------

def log_x_axis(ax, min_t, max_t):
    """Configure log x-axis with METR-style time labels."""
    ax.set_xscale("log")
    ticks = logarithmic_ticks[
        (logarithmic_ticks >= min_t * 0.8) & (logarithmic_ticks <= max_t * 1.2)
    ]
    labels = [format_time_label(t * 60) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_major_locator(
        matplotlib.ticker.FixedLocator([float(x) for x in ticks])
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def plot_empirical_bars(ax, times, successes, weights, color, alpha=0.5):
    """Plot METR-style weighted empirical success rate bars with ±2SE error bars."""
    bins = get_logarithmic_bins(times.min(), times.max())

    w_success, _ = np.histogram(
        times[successes == 1], bins=bins, weights=weights[successes == 1]
    )
    w_total, _ = np.histogram(times, bins=bins, weights=weights)

    rates = np.zeros_like(w_total, dtype=float)
    mask = w_total > 0
    rates[mask] = w_success[mask] / w_total[mask]

    widths = np.diff(bins)
    ax.bar(bins[:-1], rates, width=widths, alpha=alpha, color=color,
           align="edge", zorder=2)

    # ±2SE error bars
    se = np.zeros_like(rates)
    for i in range(len(bins) - 1):
        if mask[i]:
            bin_mask = (times >= bins[i]) & (times < bins[i + 1])
            w_bin = weights[bin_mask]
            n_eff = np.sum(w_bin) ** 2 / np.sum(w_bin ** 2) if np.sum(w_bin ** 2) > 0 else 0
            if n_eff > 0:
                var = rates[i] * (1 - rates[i]) / n_eff
                if var > 0:
                    se[i] = np.sqrt(var)

    centers = bins[:-1] + widths / 2
    from matplotlib.colors import to_rgb
    dark = np.clip(np.array(to_rgb(color)) - np.array([0.0, 0.15, -0.1]), 0, 1)
    ax.errorbar(centers[mask], rates[mask], yerr=2 * se[mask],
                fmt="o", color=dark, alpha=0.9, markersize=5, capsize=3,
                label="Empirical success\nrates w/ ± 2SE", zorder=3)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(tasks, output_path):
    # Logistic DGP: P(success) = sigmoid(-k * (log2(t) - c))
    # True p50 = 2^c by definition of sigmoid midpoint
    DGP_K = 0.8       # standard slope
    DGP_CENTER = 9.5   # log2(minutes) → true p50 = 2^9.5 = 724 min ≈ 12.1h
    dgp_true_p50 = 2 ** DGP_CENTER

    dgp_fn = lambda x: expit(-DGP_K * (np.atleast_1d(x) - DGP_CENTER))

    # Use 20 runs/task to reduce noise, pick seed closest to median inflation
    n_runs = 20
    candidate_p50s = {}
    for seed in range(50):
        rng = np.random.default_rng(seed)
        outcomes = generate_outcomes(tasks, dgp_fn, n_runs, rng)
        adf = make_synthetic_agent_df(tasks, outcomes)
        wdf = compute_sample_weights(adf)
        adf = adf.join(wdf)
        xl = np.log2(adf["human_minutes"].values).reshape(-1, 1)
        yy = adf["score_binarized"].values
        ww = adf["invsqrt_task_weight"].values
        m = logistic_regression(xl, yy, ww, regularization=0.1)
        candidate_p50s[seed] = np.exp2(get_x_for_quantile(m, 0.5))

    med_p50 = np.median(list(candidate_p50s.values()))
    best_seed = min(candidate_p50s, key=lambda s: abs(candidate_p50s[s] - med_p50))
    print(f"  True p50 = {dgp_true_p50:.0f} min ({dgp_true_p50/60:.1f}h)")
    print(f"  DGP: sigmoid(-{DGP_K} * (log2(t) - {DGP_CENTER}))")
    print(f"  Median fitted p50 across 50 seeds: {med_p50:.0f} min ({med_p50/60:.1f}h)")
    print(f"  Using seed {best_seed} (p50={candidate_p50s[best_seed]:.0f} min)")

    rng = np.random.default_rng(best_seed)
    outcomes = generate_outcomes(tasks, dgp_fn, n_runs, rng)
    agent_df = make_synthetic_agent_df(tasks, outcomes)

    # METR fit
    weight_df = compute_sample_weights(agent_df)
    agent_df = agent_df.join(weight_df)
    x_log2 = np.log2(agent_df["human_minutes"].values).reshape(-1, 1)
    y = agent_df["score_binarized"].values
    w = agent_df["invsqrt_task_weight"].values
    model = logistic_regression(x_log2, y, w, regularization=0.1)
    fit_p50 = np.exp2(get_x_for_quantile(model, 0.5))
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    bias_pct = (fit_p50 - dgp_true_p50) / dgp_true_p50 * 100

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 5.5))

    times = agent_df["human_minutes"].values
    successes = agent_df["score_binarized"].values

    # Empirical bars (subtle)
    bar_color = "#b0a8c8"
    plot_empirical_bars(ax, times, successes, w, color=bar_color, alpha=0.35)

    # True DGP curve
    x_curve = np.logspace(np.log10(0.3), np.log10(3500), 1000)
    y_true = dgp_fn(np.log2(x_curve))
    ax.plot(x_curve, y_true, color="#2ca02c", linewidth=2.5,
            label=f"True DGP: sigmoid(−{DGP_K} · (log₂t − {DGP_CENTER}))",
            zorder=5)

    # METR fitted logistic
    y_fit = expit(coef * np.log2(x_curve) + intercept)
    ax.plot(x_curve, y_fit, color="#1f77b4", linewidth=3,
            label=f"Fitted logistic (C=10, weighted)",
            zorder=4)

    # 0.5 reference line
    ax.axhline(0.5, linestyle="dotted", alpha=0.3, color="black", zorder=1)

    # Shade the inflation gap
    lo, hi = min(dgp_true_p50, fit_p50), max(dgp_true_p50, fit_p50)
    ymax_axes = 0.59 / 1.14  # axes-coord for y=0.5 given ylim=[-0.09, 1.05]
    ax.axvspan(lo, hi, ymin=0, ymax=ymax_axes,
               alpha=0.10, color="#d62728", zorder=1)

    # True p50 marker
    ax.axvline(dgp_true_p50, color="#2ca02c", linestyle="--", linewidth=2,
               ymax=ymax_axes, zorder=6)
    ax.plot(dgp_true_p50, -0.06, "x", color="#2ca02c", markersize=14,
            markeredgewidth=3, zorder=6, clip_on=False)
    ax.annotate(f"True p50: {format_time_label(dgp_true_p50 * 60)}",
                xy=(dgp_true_p50, 0.52), xytext=(-130, 30),
                textcoords="offset points", fontsize=12,
                color="#2ca02c", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                lw=1.5, mutation_scale=12))

    # Fitted p50 marker
    ax.axvline(fit_p50, color="#1f77b4", linestyle="--", linewidth=2,
               ymax=ymax_axes, zorder=6)
    ax.plot(fit_p50, -0.06, "x", color="#1f77b4", markersize=14,
            markeredgewidth=3, zorder=6, clip_on=False)
    ax.annotate(f"Fitted p50: {format_time_label(fit_p50 * 60)}",
                xy=(fit_p50, 0.52), xytext=(100, 30),
                textcoords="offset points", fontsize=12,
                color="#1f77b4", fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                lw=1.5, mutation_scale=12))

    # Inflation label between the two p50 lines
    mid_gap = np.sqrt(dgp_true_p50 * fit_p50)
    ax.annotate("", xy=(hi, 0.18), xytext=(lo, 0.18),
                arrowprops=dict(arrowstyle="<->", color="#d62728",
                                lw=2, mutation_scale=15))
    ax.text(mid_gap, 0.23, f"+{bias_pct:.0f}%",
            ha="center", va="bottom", fontsize=13, color="#d62728",
            fontweight="bold")

    # Axes
    log_x_axis(ax, max(times.min(), 0.5), max(times.max(), 1800))
    ax.set_xlim(0.3, 3500)
    ax.set_ylim(-0.09, 1.05)
    ax.set_xlabel("Task length (human time)", fontsize=13)
    ax.set_ylabel("Success Probability", fontsize=13)

    true_coef = -DGP_K
    true_int = DGP_K * DGP_CENTER
    ax.set_title(
        f"Logistic DGP (k={DGP_K}, true p50 = {dgp_true_p50/60:.1f}h): "
        f"fitted p50 overestimates by {bias_pct:.0f}%\n"
        f"Median seed out of 50   "
        f"[v1.1, {len(tasks)} tasks, {n_runs} runs/task, "
        f"regularization λ={0.1}]",
        fontsize=10.5, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.12, zorder=0)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")
    print(f"  True p50:  {dgp_true_p50:.0f} min ({dgp_true_p50/60:.1f}h)")
    print(f"  METR p50:  {fit_p50:.0f} min ({fit_p50/60:.1f}h)")
    print(f"  Inflation: {bias_pct:+.0f}%")
    print(f"  True params: coef={true_coef:.1f}, intercept={true_int:.1f}")
    print(f"  Fitted params: coef={coef:.3f}, intercept={intercept:.3f}")

    return fit_p50, outcomes, model


def make_scatter_figure(tasks, outcomes, model, output_path):
    """Raw task-level scatter: each task's empirical success rate vs human_minutes.

    No binning, no weighting — just the data the logistic sees.
    """
    DGP_K = 0.8
    DGP_CENTER = 9.5
    dgp_true_p50 = 2 ** DGP_CENTER
    dgp_fn = lambda x: expit(-DGP_K * (np.atleast_1d(x) - DGP_CENTER))

    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    fit_p50 = np.exp2(get_x_for_quantile(model, 0.5))

    # Compute per-task empirical success rate
    task_times = []
    task_rates = []
    task_n_runs = []
    for _, task in tasks.iterrows():
        tid = task["task_id"]
        hm = max(task["human_minutes"], 0.1)
        oc = outcomes[tid]
        task_times.append(hm)
        task_rates.append(oc.mean())
        task_n_runs.append(len(oc))

    task_times = np.array(task_times)
    task_rates = np.array(task_rates)
    task_n_runs = np.array(task_n_runs)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Scatter: each task as a dot, size proportional to runs
    ax.scatter(task_times, task_rates, s=20, alpha=0.45,
               color="#555555", edgecolors="none", zorder=3,
               label=f"Per-task success rate ({task_n_runs[0]} runs each)")

    # True DGP curve
    x_curve = np.logspace(np.log10(0.3), np.log10(3500), 1000)
    y_true = dgp_fn(np.log2(x_curve))
    ax.plot(x_curve, y_true, color="#2ca02c", linewidth=2.5,
            label=f"True DGP: sigmoid(−{DGP_K} · (log₂t − {DGP_CENTER}))",
            zorder=5)

    # Fitted logistic
    y_fit = expit(coef * np.log2(x_curve) + intercept)
    ax.plot(x_curve, y_fit, color="#1f77b4", linewidth=2.5,
            label=f"Fitted logistic (C=10, weighted)",
            zorder=4)

    # 0.5 reference line
    ax.axhline(0.5, linestyle="dotted", alpha=0.3, color="black", zorder=1)

    # p50 markers
    ymax_axes = 0.59 / 1.14  # axes-coord for y=0.5 given ylim=[-0.09, 1.05]
    ax.axvline(dgp_true_p50, color="#2ca02c", linestyle="--", linewidth=1.5,
               ymax=ymax_axes, zorder=6, alpha=0.7)
    ax.axvline(fit_p50, color="#1f77b4", linestyle="--", linewidth=1.5,
               ymax=ymax_axes, zorder=6, alpha=0.7)

    bias_pct = (fit_p50 - dgp_true_p50) / dgp_true_p50 * 100
    ax.annotate(f"True p50\n{format_time_label(dgp_true_p50 * 60)}",
                xy=(dgp_true_p50, 0.5), xytext=(-100, 30),
                textcoords="offset points", fontsize=10,
                color="#2ca02c", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#2ca02c",
                                lw=1.2, mutation_scale=10))
    ax.annotate(f"Fitted p50\n{format_time_label(fit_p50 * 60)} (+{bias_pct:.0f}%)",
                xy=(fit_p50, 0.5), xytext=(90, 30),
                textcoords="offset points", fontsize=10,
                color="#1f77b4", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                                lw=1.2, mutation_scale=10))

    # Task density rug along bottom
    ax.vlines(task_times, -0.07, -0.03, color="#888888", alpha=0.3,
              linewidth=0.5, zorder=2)
    ax.text(0.5, -0.05, "task density", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#888888", ha="left", va="center")

    # Axes
    log_x_axis(ax, max(task_times.min(), 0.5), max(task_times.max(), 1800))
    ax.set_xlim(0.3, 3500)
    ax.set_ylim(-0.09, 1.05)
    ax.set_xlabel("Task length (human time)", fontsize=13)
    ax.set_ylabel("Success Rate (per task)", fontsize=13)

    ax.set_title(
        f"Raw task-level outcomes (no binning, no weighting)\n"
        f"Each dot = one task's success rate over "
        f"{task_n_runs[0]} runs   [v1.1, {len(tasks)} tasks]",
        fontsize=10.5, pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.12, zorder=0)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def make_regularization_figure(tasks, output_path):
    """Sweep regularization strength for several DGP settings.

    For each (k, c) pair and each regularization value, run 50 seeds and
    report the median fitted p50.  Produces a grouped bar chart.
    """
    N_SEEDS = 50
    N_RUNS = 20
    regs = [0.0001, 0.001, 0.01, 0.1]
    dgps = [
        (0.8, 9.5, "k=0.8, p50=12.1h"),
        (0.5, 9.5, "k=0.5, p50=12.1h"),
        (0.5, 7.0, "k=0.5, p50=2.1h"),
    ]

    # rows: one per (dgp, reg)
    results = []
    for k, c, label in dgps:
        true_p50 = 2 ** c
        true_fn = lambda x, _k=k, _c=c: expit(-_k * (np.atleast_1d(x) - _c))
        for reg in regs:
            p50s = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                outcomes = generate_outcomes(tasks, true_fn, N_RUNS, rng)
                adf = make_synthetic_agent_df(tasks, outcomes)
                wdf = compute_sample_weights(adf)
                adf = adf.join(wdf)
                xl = np.log2(adf["human_minutes"].values).reshape(-1, 1)
                yy = adf["score_binarized"].values
                ww = adf["invsqrt_task_weight"].values
                m = logistic_regression(xl, yy, ww, regularization=reg)
                p50s.append(np.exp2(get_x_for_quantile(m, 0.5)))
            med = np.median(p50s)
            bias = (med - true_p50) / true_p50 * 100
            results.append({
                "dgp": label, "k": k, "c": c, "true_p50": true_p50,
                "reg": reg, "C": 1 / reg,
                "median_p50": med, "bias_pct": bias,
            })
            print(f"  {label}  reg={reg:<8g}  "
                  f"median p50={med:>8.0f} min  bias={bias:>+7.1f}%")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 5))

    n_dgps = len(dgps)
    n_regs = len(regs)
    group_width = 0.8
    bar_width = group_width / n_dgps
    x_positions = np.arange(n_regs)

    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for di, (k, c, label) in enumerate(dgps):
        biases = [r["bias_pct"] for r in results
                  if r["k"] == k and r["c"] == c]
        offset = (di - (n_dgps - 1) / 2) * bar_width
        bars = ax.bar(x_positions + offset, biases, bar_width * 0.9,
                      label=label, color=colors[di], alpha=0.8, zorder=3)
        for bar, b in zip(bars, biases):
            va = "bottom" if b >= 0 else "top"
            y_off = 1.0 if b >= 0 else -1.0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_off,
                    f"{b:+.1f}%", ha="center", va=va, fontsize=8,
                    fontweight="bold" if abs(b) > 10 else "normal")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"λ={r}\n(C={1/r:g})" for r in regs], fontsize=9)
    ax.set_xlabel("Regularization strength", fontsize=12)
    ax.set_ylabel("p50 bias (%)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    # Highlight METR's default
    metr_idx = regs.index(0.1)
    ax.axvspan(metr_idx - 0.5, metr_idx + 0.5,
               alpha=0.08, color="#d62728", zorder=1)
    ax.text(metr_idx, ax.get_ylim()[1] * 0.95, "METR\ndefault",
            ha="center", va="top", fontsize=9, color="#d62728",
            fontweight="bold")

    ax.set_title(
        f"Effect of regularization on p50 bias (logistic DGP)\n"
        f"Median over {N_SEEDS} seeds, {N_RUNS} runs/task, "
        f"v1.1 ({len(tasks)} tasks)",
        fontsize=10.5, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.15, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def get_real_agent_params(runs_path):
    """Compute fitted logistic params for real agents."""
    runs = pd.read_json(runs_path, lines=True, orient="records")
    results = []
    for agent in sorted(runs.alias.unique()):
        ag = runs[runs.alias == agent].copy()
        y = ag["score_binarized"].values.astype(float)
        if len(ag) < 20 or len(np.unique(y)) < 2:
            continue
        if "invsqrt_task_weight" not in ag.columns:
            ag = ag.join(compute_sample_weights(ag))
        w = ag["invsqrt_task_weight"].values
        x = np.log2(ag["human_minutes"].values).reshape(-1, 1)
        model = logistic_regression(x, y, w, regularization=0.1)
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        p50 = np.exp2(get_x_for_quantile(model, 0.5))
        results.append({
            "agent": agent.replace(" (Inspect)", ""),
            "coef": coef, "intercept": intercept,
            "k": -coef, "p50_min": p50, "p50_h": p50 / 60,
        })
    return pd.DataFrame(results)


def make_grid_heatmap(tasks, output_path):
    """Heatmap of bias across logistic DGP parameter space."""

    centers_log2 = np.array([4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.3])
    steepnesses = np.array([0.3, 0.5, 0.8, 1.0, 1.5])
    N_TRIALS = 30

    bias = np.zeros((len(steepnesses), len(centers_log2)))
    fitted_p50 = np.zeros((len(steepnesses), len(centers_log2)))
    total = len(steepnesses) * len(centers_log2)
    done = 0

    for i, k in enumerate(steepnesses):
        for j, c in enumerate(centers_log2):
            true_p50 = 2 ** c
            true_fn = lambda x, _k=k, _c=c: expit(-_k * (np.atleast_1d(x) - _c))

            p50s = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(42 + trial)
                outcomes = generate_outcomes(tasks, true_fn, 6, rng)
                agent_df = make_synthetic_agent_df(tasks, outcomes)
                weight_df = compute_sample_weights(agent_df)
                agent_df = agent_df.join(weight_df)
                x_log2 = np.log2(agent_df["human_minutes"].values).reshape(-1, 1)
                y = agent_df["score_binarized"].values
                w = agent_df["invsqrt_task_weight"].values
                model = logistic_regression(x_log2, y, w, regularization=0.1)
                p50s.append(np.exp2(get_x_for_quantile(model, 0.5)))

            med = np.median(p50s)
            fitted_p50[i, j] = med
            bias[i, j] = (med - true_p50) / true_p50 * 100
            done += 1
            if done % 12 == 0 or done == total:
                print(f"  Grid: {done}/{total}", flush=True)

    # Get real agent fitted params for context
    runs_path = os.path.join(_repo_root, "reports", "time-horizon-1-1",
                             "data", "raw", "runs.jsonl")
    real = get_real_agent_params(runs_path)
    print("\n  Real agent fitted parameters:")
    for _, r in real.iterrows():
        print(f"    {r['agent']:<35} k={r['k']:.3f}  p50={r['p50_h']:.1f}h")

    centers_hours = 2 ** centers_log2 / 60

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = min(abs(bias).max(), 200)
    im = ax.imshow(bias, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax, origin="lower")

    ax.set_xticks(range(len(centers_log2)))
    ax.set_xticklabels([format_time_label(h * 3600) for h in centers_hours],
                       fontsize=9)
    ax.set_yticks(range(len(steepnesses)))
    ytick_labels = []
    for k in steepnesses:
        if 0.41 <= k <= 0.59:
            ytick_labels.append(f"k = {k}  (real agents)")
        else:
            ytick_labels.append(f"k = {k}")
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.set_xlabel("True p50 of logistic DGP", fontsize=12)
    ax.set_ylabel("Steepness (k)", fontsize=12)

    for i in range(len(steepnesses)):
        for j in range(len(centers_log2)):
            fp = fitted_p50[i, j]
            display = format_time_label(fp * 60)
            b = bias[i, j]
            color = "white" if abs(b) > vmax * 0.55 else "black"
            ax.text(j, i, display, ha="center", va="center",
                    fontsize=7.5, color=color,
                    fontweight="bold" if abs(b) > 20 else "normal")

    # Mark the fitted-k band of real agents (k ≈ 0.41–0.59)
    # All real agents fit to k between 0.41 and 0.59, spanning the k=0.3 and k=0.5 rows
    k_lo_idx = np.interp(0.41, steepnesses, range(len(steepnesses)))
    k_hi_idx = np.interp(0.59, steepnesses, range(len(steepnesses)))
    for row_idx in range(len(steepnesses)):
        if k_lo_idx - 0.5 <= row_idx <= k_hi_idx + 0.5:
            for col_idx in range(len(centers_log2)):
                rect = plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1,
                                     fill=False, edgecolor="black",
                                     linewidth=0.8, linestyle="-", zorder=9)
                ax.add_patch(rect)

    cbar = plt.colorbar(im, label="p50 bias (%)", pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "p50 overestimation across logistic DGP parameters\n"
        f"Cell text = fitted p50, color = bias %   "
        f"[v1.1, {len(tasks)} tasks, {N_TRIALS} trials/cell, "
        f"regularization λ=0.1]",
        fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def make_true_vs_fitted_figure(tasks, output_path):
    """Log-log plot of true p50 vs fitted p50 for k=0.5, true p50 from 4h to 16h."""
    K = 0.5
    N_TRIALS = 50
    N_RUNS = 20
    true_p50_hours = np.arange(2, 17)  # 2h to 16h inclusive, 1h increments
    true_p50_minutes = true_p50_hours * 60
    centers_log2 = np.log2(true_p50_minutes)

    median_fitted = np.zeros(len(centers_log2))
    q25_fitted = np.zeros(len(centers_log2))
    q75_fitted = np.zeros(len(centers_log2))

    for j, c in enumerate(centers_log2):
        true_fn = lambda x, _c=c: expit(-K * (np.atleast_1d(x) - _c))
        p50s = []
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(42 + trial)
            outcomes = generate_outcomes(tasks, true_fn, N_RUNS, rng)
            agent_df = make_synthetic_agent_df(tasks, outcomes)
            weight_df = compute_sample_weights(agent_df)
            agent_df = agent_df.join(weight_df)
            xl = np.log2(agent_df["human_minutes"].values).reshape(-1, 1)
            yy = agent_df["score_binarized"].values
            ww = agent_df["invsqrt_task_weight"].values
            m = logistic_regression(xl, yy, ww, regularization=0.1)
            p50s.append(np.exp2(get_x_for_quantile(m, 0.5)))
        p50s = np.array(p50s)
        median_fitted[j] = np.median(p50s)
        q25_fitted[j] = np.percentile(p50s, 25)
        q75_fitted[j] = np.percentile(p50s, 75)
        bias_pct = (median_fitted[j] - true_p50_minutes[j]) / true_p50_minutes[j] * 100
        print(f"  True p50={true_p50_hours[j]:>2.0f}h  "
              f"median fitted={median_fitted[j]/60:.1f}h  bias={bias_pct:+.1f}%")

    # Convert to hours for plotting
    med_h = median_fitted / 60
    q25_h = q25_fitted / 60
    q75_h = q75_fitted / 60

    fig, ax = plt.subplots(figsize=(7, 7))

    # y = x reference line
    span = np.array([3, 30])
    ax.plot(span, span, color="black", linestyle="--", linewidth=1, alpha=0.4,
            label="y = x (no bias)", zorder=1)

    # IQR band
    ax.fill_between(true_p50_hours, q25_h, q75_h,
                     alpha=0.18, color="#1f77b4", zorder=2,
                     label="IQR (25th–75th percentile)")

    # Median fitted
    ax.plot(true_p50_hours, med_h, "o-", color="#1f77b4", linewidth=2,
            markersize=6, zorder=4, label="Median fitted p50")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True p50 (hours)", fontsize=13)
    ax.set_ylabel("Fitted p50 (hours)", fontsize=13)

    # Use hour ticks — thin out x to avoid crowding
    x_ticks = [2, 4, 6, 8, 10, 12, 14, 16]
    y_ticks = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{h}h" for h in x_ticks], fontsize=9)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{h}h" for h in y_ticks], fontsize=9)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    ax.set_xlim(1.8, 18)
    ax.set_ylim(1.8, 28)

    ax.set_title(
        f"True vs fitted p50 (logistic DGP, k={K})\n"
        f"Median over {N_TRIALS} seeds, {N_RUNS} runs/task, "
        f"v1.1 ({len(tasks)} tasks), λ=0.1",
        fontsize=10.5, pad=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15, zorder=0)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    tasks = load_v1_1_task_scaffold()
    print(f"Loaded {len(tasks)} tasks from v1.1\n")

    os.makedirs(os.path.join(_script_dir, "figures"), exist_ok=True)

    print("--- Figure 1: Representative inflation example ---")
    fit_p50, outcomes, model = make_figure(
        tasks,
        os.path.join(_script_dir, "figures", "inflation_representative.png"))

    print("\n--- Figure 2: Raw task-level scatter ---")
    make_scatter_figure(
        tasks, outcomes, model,
        os.path.join(_script_dir, "figures", "inflation_scatter.png"))

    print("\n--- Figure 3: Grid search heatmap ---")
    make_grid_heatmap(tasks,
                      os.path.join(_script_dir, "figures", "logistic_dgp_grid_heatmap.png"))

    print("\n--- Figure 4: Regularization sweep ---")
    make_regularization_figure(
        tasks,
        os.path.join(_script_dir, "figures", "regularization_sweep.png"))

    print("\n--- Figure 5: True vs fitted p50 ---")
    make_true_vs_fitted_figure(
        tasks,
        os.path.join(_script_dir, "figures", "true_vs_fitted_p50.png"))
