"""
Plot sigmoid slopes (logits per doubling of task length) vs model release date for TH1.1.
"""

import pathlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


def main() -> None:
    report_dir = pathlib.Path(__file__).resolve().parent
    slopes_file = report_dir / "data" / "wrangled" / "sigmoid_slopes.csv"
    release_dates_file = report_dir.parent.parent / "data" / "external" / "release_dates.yaml"

    df = pd.read_csv(slopes_file)
    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]

    # Map release dates
    df["release_date"] = df["agent"].map(release_dates)
    # For agents without "(Inspect)" suffix in release_dates, try without
    for i, row in df.iterrows():
        if pd.isna(row["release_date"]):
            bare = row["agent"].replace(" (Inspect)", "")
            if bare in release_dates:
                df.at[i, "release_date"] = release_dates[bare]

    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df.sort_values("release_date")

    # Determine provider for coloring
    def get_provider(agent: str) -> str:
        if "Claude" in agent:
            return "Anthropic"
        return "OpenAI"

    df["provider"] = df["agent"].apply(get_provider)
    colors = {"Anthropic": "#d97757", "OpenAI": "#18a683"}
    markers = {"Anthropic": "o", "OpenAI": "s"}

    # Short labels
    rename = {
        "Claude 3 Opus (Inspect)": "Opus 3",
        "Claude 3.5 Sonnet (Old) (Inspect)": "Sonnet 3.5",
        "Claude 3.5 Sonnet (New) (Inspect)": "Sonnet 3.6",
        "Claude 3.7 Sonnet (Inspect)": "Sonnet 3.7",
        "Claude 4 Opus (Inspect)": "Opus 4",
        "Claude Opus 4.5 (Inspect)": "Opus 4.5",
        "GPT-4 0314": "GPT-4",
        "GPT-4 1106 (Inspect)": "GPT-4 Nov'23",
        "GPT-4 Turbo (Inspect)": "GPT-4 Turbo",
        "GPT-4o (Inspect)": "GPT-4o",
        "GPT-5 (Inspect)": "GPT-5",
        "o1 (Inspect)": "o1",
        "o1-preview": "o1-preview",
        "o3 (Inspect)": "o3",
    }
    df["label"] = df["agent"].map(rename).fillna(df["agent"])

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for provider in ["Anthropic", "OpenAI"]:
        mask = df["provider"] == provider
        sub = df[mask]
        yerr_low = sub["slope_logits_per_doubling"] - sub["ci_low_2.5%"]
        yerr_high = sub["ci_high_97.5%"] - sub["slope_logits_per_doubling"]

        ax.errorbar(
            sub["release_date"],
            sub["slope_logits_per_doubling"],
            yerr=[yerr_low, yerr_high],
            fmt="none",
            ecolor=colors[provider],
            alpha=0.4,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            zorder=5,
        )
        ax.scatter(
            sub["release_date"],
            sub["slope_logits_per_doubling"],
            color=colors[provider],
            marker=markers[provider],
            s=100,
            edgecolor="black",
            linewidth=0.5,
            label=provider,
            zorder=10,
        )

    # Add labels
    for _, row in df.iterrows():
        offset_x = 8
        offset_y = 0.008
        ha = "left"
        # Handle overlapping labels
        if row["label"] in ("Opus 3", "Sonnet 3.5"):
            offset_y = 0.012
        if row["label"] in ("GPT-4o", "GPT-5"):
            offset_y = -0.012
            ha = "left"
        if row["label"] in ("Opus 4", "Opus 4.5"):
            offset_x = -8
            ha = "right"

        ax.annotate(
            row["label"],
            (row["release_date"], row["slope_logits_per_doubling"]),
            textcoords="offset points",
            xytext=(offset_x, offset_y * 1000),
            fontsize=8,
            ha=ha,
            va="center",
            color="dimgray",
        )

    ax.set_xlabel("Model release date", fontsize=13)
    ax.set_ylabel("Sigmoid slope (logits per doubling of task length)", fontsize=13)
    ax.set_title("Sigmoid slopes for TH1.1 agents vs. release date", fontsize=15)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    ax.axhline(y=df["slope_logits_per_doubling"].mean(), color="grey", linestyle="--",
               alpha=0.5, linewidth=1, label=f"Mean = {df['slope_logits_per_doubling'].mean():.3f}")

    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, color="grey")

    plt.tight_layout()
    output_path = report_dir / "data" / "wrangled" / "sigmoid_slopes_vs_release_date.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
