"""
Compute sigmoid slopes (logits per doubling of task length) for each agent in TH1.1,
with bootstrap 95% confidence intervals.

The logistic model is:
    logit(p) = intercept + coefficient * log2(human_minutes)

Since log2(human_minutes) increases by 1 when human_minutes doubles,
the coefficient is directly in units of "logits per doubling of task length".
"""

import pathlib
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression


# ---------- inlined from horizon.utils.logistic ----------

def logistic_regression(
    X: NDArray[Any],
    y: NDArray[Any],
    sample_weight: NDArray[Any],
    regularization: float,
    ensure_weights_sum_to_1: bool = True,
) -> LogisticRegression:
    assert np.all((y >= 0) & (y <= 1)), "y values must be in [0,1]"

    if ensure_weights_sum_to_1:
        assert np.allclose(np.sum(sample_weight), 1.0), (
            f"sample_weight must sum to 1.0, got {np.sum(sample_weight)}"
        )

    original_weight_sum = np.sum(sample_weight)
    original_average = np.average(y, weights=sample_weight)

    fractional_mask = (y > 0) & (y < 1)
    if np.any(fractional_mask):
        X_frac = X[fractional_mask]
        y_frac = y[fractional_mask]
        w_frac = sample_weight[fractional_mask]
        X_split = np.vstack([X_frac, X_frac])
        y_split = np.zeros(2 * len(y_frac))
        y_split[len(y_frac):] = 1
        w_split = np.concatenate([(1 - y_frac) * w_frac, y_frac * w_frac])
        X = np.vstack([X[~fractional_mask], X_split])
        y = np.concatenate([y[~fractional_mask], y_split])
        sample_weight = np.concatenate([sample_weight[~fractional_mask], w_split])

    model = LogisticRegression(C=1 / regularization)
    model.fit(X, y, sample_weight=sample_weight)
    return model


# ---------- inlined from horizon.wrangle.bootstrap ----------

def bootstrap_runs_by_task_agent(
    task_col: np.ndarray,
    agent_col: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    task_agent = np.char.add(
        np.char.add(task_col.astype(str), "|||"), agent_col.astype(str)
    )
    task_agents = task_agent[indices]
    unique_task_agents, task_agent_indices, counts = np.unique(
        task_agents, return_inverse=True, return_counts=True
    )
    random_nums = rng.random(len(indices))
    offsets = np.cumsum([0] + list(counts)[:-1])
    all_new_indices = [
        indices[task_agent_indices == j][
            (random_nums[offset : offset + count] * count).astype(np.int64)
        ]
        for j, (count, offset) in enumerate(zip(counts, offsets))
    ]
    return np.concatenate(all_new_indices)


def bootstrap_sample(
    data: pd.DataFrame,
    categories: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    has_run_id = "run_id" in categories
    categories = [c for c in categories if c != "run_id"]

    category_arrays = {category: data[category].to_numpy() for category in categories}

    indices = np.arange(len(data))
    split_ids = np.zeros(len(data), dtype=np.int32)
    new_split_id = 0

    for i, category in enumerate(categories):
        is_last_category = i == len(categories) - 1
        all_new_indices = []
        all_new_split_ids = []

        for group_id in np.unique(split_ids):
            group_indices = indices[split_ids == group_id]
            category_values = category_arrays[category][group_indices]
            values, value_indices = np.unique(category_values, return_inverse=True)
            n_values = len(values)
            sampled_values = rng.choice(n_values, size=n_values, replace=True)

            for j, sampled_value in enumerate(sampled_values):
                sampled_indices = group_indices[value_indices == sampled_value]
                all_new_indices.append(sampled_indices)
                if not is_last_category:
                    all_new_split_ids.append(
                        np.full(len(sampled_indices), new_split_id + j)
                    )
            new_split_id += n_values

        indices = np.concatenate(all_new_indices)
        if not is_last_category:
            split_ids = np.concatenate(all_new_split_ids)

    if has_run_id:
        task_col = data["task_id"].to_numpy()
        agent_col = data["agent"].to_numpy()
        indices = bootstrap_runs_by_task_agent(task_col, agent_col, indices, rng)

    return data.iloc[indices].copy()


# ---------- analysis ----------

REGULARIZATION = 0.1
WEIGHTING = "invsqrt_task_weight"
CATEGORIES = ["task_family", "task_id", "run_id"]
N_BOOTSTRAP = 1000


def fit_slope(agent_data: pd.DataFrame, ensure_weights_sum_to_1: bool = True) -> float:
    """Fit logistic regression and return slope (coefficient) in logits per doubling."""
    x = np.log2(agent_data["human_minutes"].values).reshape(-1, 1)
    y = agent_data["score_binarized"].values
    weights = agent_data[WEIGHTING].values

    if len(np.unique(y)) < 2:
        return float("nan")

    model = logistic_regression(
        x, y,
        sample_weight=weights,
        regularization=REGULARIZATION,
        ensure_weights_sum_to_1=ensure_weights_sum_to_1,
    )
    return model.coef_[0][0]


def bootstrap_slopes(data: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Run bootstrap and return DataFrame of slopes per agent per iteration."""

    def process_batch(batch_idx: int, batch_size: int) -> list[dict[str, float]]:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, N_BOOTSTRAP)
        batch_results = []
        for i in range(start_idx, end_idx):
            rng = np.random.default_rng(seed + i)
            bs_data = bootstrap_sample(data, CATEGORIES, rng)
            result = {}
            for agent_name in bs_data["agent"].unique():
                agent_data = bs_data[bs_data["agent"] == agent_name]
                x = np.log2(agent_data["human_minutes"].values).reshape(-1, 1)
                y = agent_data["score_binarized"].values
                weights = agent_data[WEIGHTING].values

                if len(np.unique(y)) < 2:
                    continue

                model = logistic_regression(
                    x, y,
                    sample_weight=weights,
                    regularization=REGULARIZATION,
                    ensure_weights_sum_to_1=False,
                )
                result[agent_name] = model.coef_[0][0]
            batch_results.append(result)
        return batch_results

    batch_size = 10
    n_batches = (N_BOOTSTRAP + batch_size - 1) // batch_size
    n_jobs = max(1, Parallel(n_jobs=-1)._effective_n_jobs())

    print(f"Running {N_BOOTSTRAP} bootstrap iterations in {n_batches} batches on {n_jobs} cores...")
    batched_results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_batch)(i, batch_size) for i in range(n_batches)
    )

    results = [r for batch in batched_results for r in batch]
    return pd.DataFrame(results)


def main() -> None:
    report_dir = pathlib.Path(__file__).resolve().parent
    runs_file = report_dir / "data" / "raw" / "runs.jsonl"

    print(f"Loading data from {runs_file}")
    data = pd.read_json(runs_file, lines=True, orient="records", convert_dates=False)
    data.rename(columns={"alias": "agent"}, inplace=True)

    # Exclude "human" from agent analysis
    agents = sorted([a for a in data["agent"].unique() if a != "human"])
    print(f"Found {len(agents)} agents (excluding human)")

    # --- Point estimates ---
    print("\nFitting point estimates...")
    point_estimates = {}
    for agent in agents:
        agent_data = data[data["agent"] == agent]
        slope = fit_slope(agent_data, ensure_weights_sum_to_1=True)
        point_estimates[agent] = slope

    # --- Bootstrap CIs ---
    bs_slopes = bootstrap_slopes(data)

    # --- Compile results ---
    print("\n" + "=" * 90)
    print(f"{'Agent':<40} {'Slope':>10} {'95% CI':>25}")
    print(f"{'':40} {'(logits/':>10} {'':>25}")
    print(f"{'':40} {'doubling)':>10} {'':>25}")
    print("=" * 90)

    results = []
    for agent in agents:
        slope = point_estimates[agent]
        if agent in bs_slopes.columns:
            bs_vals = bs_slopes[agent].dropna()
            ci_low = np.nanquantile(bs_vals, 0.025)
            ci_high = np.nanquantile(bs_vals, 0.975)
        else:
            ci_low = float("nan")
            ci_high = float("nan")

        results.append({
            "agent": agent,
            "slope_logits_per_doubling": round(slope, 4),
            "ci_low_2.5%": round(ci_low, 4),
            "ci_high_97.5%": round(ci_high, 4),
        })
        print(f"{agent:<40} {slope:>10.4f}   [{ci_low:>8.4f}, {ci_high:>8.4f}]")

    print("=" * 90)

    # Save to CSV
    output_file = report_dir / "data" / "wrangled" / "sigmoid_slopes.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
