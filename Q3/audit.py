"""
audit.py
========
Sound Check Audit – Identifies Documentation Debt and Representation Bias
in the SPS Corpus (or any compatible TSV-based audio dataset).

Usage:
    # Single TSV (SPS corpus format):
    python audit.py --tsv data/sps-corpus-3.0-2026-03-09-en/ss-corpus-en.tsv \
                    --output_dir audit_output

    # Common Voice directory with split files:
    python audit.py --data_dir /path/to/cv-corpus-en --output_dir audit_output

    # No files needed:
    python audit.py --demo

Outputs:
    audit_output/audit_plots.pdf
    audit_output/audit_report.csv
    audit_output/bias_scores.json
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import entropy

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────────────────
# Real SPS corpus label sets
# ─────────────────────────────────────────────────────────────────────────────
GENDER_LABELS = [
    "female_feminine",
    "male_masculine",
    "intersex",
    "transgender",
    "non-binary",
    "do_not_wish_to_say",
    "",
]

AGE_LABELS = [
    "teens", "twenties", "thirties", "fourties",
    "fifties", "sixties", "seventies", "eighties", "nineties", "",
]

DIALECT_COL = "accents"

GENDER_COLOURS = [
    "#3B8BD4",  # female_feminine
    "#E24B4A",  # male_masculine
    "#7F77DD",  # intersex
    "#D4537E",  # transgender
    "#5DCAA5",  # non-binary
    "#888780",  # do_not_wish_to_say
    "#cccccc",  # missing
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(tsv_path: str) -> pd.DataFrame:
    """Load a single TSV file (SPS corpus format)."""
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    if "split" not in df.columns:
        df["split"] = "all"
    print(f"  Loaded {len(df):,} rows from {tsv_path}")
    return df


def load_common_voice(data_dir: str) -> pd.DataFrame:
    """Load all standard Common Voice TSV splits from a directory."""
    splits = ["train", "dev", "test", "validated", "other", "invalidated"]
    frames = []
    for split in splits:
        p = Path(data_dir) / f"{split}.tsv"
        if p.exists():
            df = pd.read_csv(p, sep="\t", low_memory=False)
            df["split"] = split
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No TSV files found in {data_dir}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(combined):,} rows from {data_dir}")
    return combined


def generate_demo_data(n: int = 6382, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataframe that mirrors the real SPS corpus statistics
    (6382 rows, same gender/age/accent proportions from your log output).
    """
    rng = np.random.default_rng(seed)

    genders = rng.choice(
        ["female_feminine", "male_masculine", "intersex",
         "transgender", "non-binary", "do_not_wish_to_say", None],
        p=[0.381, 0.082, 0.026, 0.006, 0.002, 0.030, 0.473],
        size=n,
    )
    ages = rng.choice(
        ["twenties", "teens", "thirties", "sixties",
         "fifties", "fourties", "seventies", "eighties", "nineties", None],
        p=[0.510, 0.124, 0.102, 0.078, 0.063, 0.021, 0.022, 0.001, 0.0005, 0.0785],
        size=n,
    )
    accents = rng.choice(
        [
            "United States English",
            "European English|French improved|Some say Pakistani|international accent",
            "England English",
            "Australian English",
            "United States English|England English",
            "Southern African (South Africa, Zimbabwe, Namibia)",
            "First language:Russian",
            "Eastern European heavily influenced by the Greek clear pronounciation of vowels and consonants.",
            "England English|Southern England English",
            None,
        ],
        p=[0.0608, 0.0268, 0.0238, 0.0038, 0.0036, 0.0031, 0.0028, 0.0025, 0.0024, 0.8704],
        size=n,
    )

    return pd.DataFrame({
        "client_id":  [f"spk_{i:05d}" for i in range(n)],
        "path":       [f"clips/sample_{i:05d}.mp3" for i in range(n)],
        "sentence":   ["placeholder sentence"] * n,
        "up_votes":   rng.integers(0, 10, n),
        "down_votes": rng.integers(0, 5, n),
        "age":        ages,
        "gender":     genders,
        "accents":    accents,
        "split":      "all",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Accent normalisation
# ─────────────────────────────────────────────────────────────────────────────

def explode_accents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split pipe-separated accent strings and explode to one row per label.
    e.g. "European English|French improved" becomes two rows.
    Empty strings and NaN are kept as-is (blank label).
    """
    if DIALECT_COL not in df.columns:
        return df
    tmp = df.copy()
    tmp[DIALECT_COL] = (
        tmp[DIALECT_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split(r"\|")
    )
    exploded = tmp.explode(DIALECT_COL).reset_index(drop=True)
    exploded[DIALECT_COL] = exploded[DIALECT_COL].str.strip()
    return exploded


# ─────────────────────────────────────────────────────────────────────────────
# Audit functions
# ─────────────────────────────────────────────────────────────────────────────

def documentation_debt_report(df: pd.DataFrame) -> dict:
    total = len(df)
    report = {}
    for col in ["gender", "age", DIALECT_COL]:
        if col not in df.columns:
            report[col] = {
                "total": total, "missing": total,
                "missing_pct": 100.0, "filled_pct": 0.0,
            }
            continue
        missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        n_miss  = int(missing.sum())
        report[col] = {
            "total":       total,
            "missing":     n_miss,
            "missing_pct": round(n_miss / total * 100, 2),
            "filled_pct":  round((total - n_miss) / total * 100, 2),
        }
    return report


def representation_bias(df: pd.DataFrame, col: str) -> dict:
    s            = df[col].fillna("").astype(str).str.strip().str.lower()
    vc           = s.value_counts(normalize=True)
    vc_no_blank  = vc[vc.index != ""]

    ent      = float(entropy(vc_no_blank.values, base=2))
    max_ent  = float(np.log2(len(vc_no_blank))) if len(vc_no_blank) > 1 else 1.0
    norm_ent = ent / max_ent if max_ent > 0 else 0.0

    counts = vc_no_blank.values.astype(float)
    counts.sort()
    n    = len(counts)
    gini = (
        (2 * np.sum(np.arange(1, n + 1) * counts) / (n * counts.sum()) - (n + 1) / n)
        if n > 1 else 0.0
    )
    return {
        "distribution":       vc.to_dict(),
        "entropy_bits":       round(ent, 4),
        "normalized_entropy": round(float(norm_ent), 4),
        "gini_coefficient":   round(float(gini), 4),
        "num_categories":     int(len(vc_no_blank)),
    }


def intersectional_bias(df: pd.DataFrame) -> pd.DataFrame:
    g = df["gender"].fillna("unknown").astype(str).str.strip().str.lower()
    a = df["age"].fillna("unknown").astype(str).str.strip().str.lower()
    return pd.crosstab(g, a, normalize="all") * 100


def chi_square_uniformity(df: pd.DataFrame, col: str):
    from scipy.stats import chisquare
    counts = (
        df[col].fillna("").astype(str).str.strip().str.lower()
        .value_counts()
    )
    counts = counts[counts.index != ""]
    if len(counts) < 2:
        return None, None
    expected = np.full(len(counts), counts.sum() / len(counts))
    stat, p  = chisquare(counts.values, f_exp=expected)
    return round(float(stat), 4), round(float(p), 6)


def split_bias_report(df: pd.DataFrame) -> dict:
    """Per-split demographic distribution (only meaningful if splits exist)."""
    if "split" not in df.columns or df["split"].nunique() < 2:
        return {}
    results = {}
    for col in ["gender", "age"]:
        if col not in df.columns:
            continue
        ct = pd.crosstab(
            df["split"].fillna("unknown"),
            df[col].fillna("unknown").astype(str).str.lower(),
            normalize="index",
        ) * 100
        results[col] = ct.to_dict()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hbar(ax, labels, values, colours, title, xlabel="Percentage (%)"):
    clipped_colours = colours[:len(labels)]
    bars = ax.barh(labels, values, color=clipped_colours)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", fontsize=11)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8,
        )
    ax.invert_yaxis()


# ─────────────────────────────────────────────────────────────────────────────
# Create all plots → single PDF
# ─────────────────────────────────────────────────────────────────────────────

def create_plots(
    df: pd.DataFrame,
    df_acc: pd.DataFrame,
    debt: dict,
    bias_scores: dict,
    output_dir: str,
) -> str:
    pdf_path = os.path.join(output_dir, "audit_plots.pdf")

    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Documentation Debt ────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Documentation Debt – Missing Metadata (%)",
                     fontsize=14, fontweight="bold")
        for ax, (col, colour) in zip(
            axes,
            {"gender": "#E07B54", "age": "#5B8DB8", DIALECT_COL: "#6DBF67"}.items(),
        ):
            d    = debt.get(col, {})
            vals = [d.get("filled_pct", 0), d.get("missing_pct", 0)]
            ax.pie(
                vals, labels=["Filled", "Missing"], autopct="%1.1f%%",
                colors=[colour, "#cccccc"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            ax.set_title(col.capitalize(), fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 2: Gender Distribution ───────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Gender Representation – SPS Corpus",
                     fontsize=14, fontweight="bold")

        g_dist = bias_scores.get("gender", {}).get("distribution", {})
        g_order = [
            "female_feminine", "male_masculine", "do_not_wish_to_say",
            "intersex", "transgender", "non-binary", "",
        ]
        g_labels = [k if k else "(missing)" for k in g_order if k in g_dist]
        g_vals   = [g_dist[k] * 100 for k in g_order if k in g_dist]
        _hbar(ax1, g_labels, g_vals, GENDER_COLOURS,
              "Gender distribution (all rows)")

        labelled = sum(v for k, v in g_dist.items() if k)
        missing  = g_dist.get("", 0)
        ax2.pie(
            [labelled * 100, missing * 100],
            labels=[f"Labelled\n({labelled*100:.1f}%)",
                    f"Missing\n({missing*100:.1f}%)"],
            colors=["#3B8BD4", "#cccccc"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax2.set_title("Labelled vs missing gender", fontsize=11, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 3: Age Distribution ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Age Group Representation – SPS Corpus",
                     fontsize=14, fontweight="bold")
        a_dist  = bias_scores.get("age", {}).get("distribution", {})
        a_order = [
            "teens", "twenties", "thirties", "fourties",
            "fifties", "sixties", "seventies", "eighties", "nineties", "",
        ]
        a_labels = [k if k else "(missing)" for k in a_order if k in a_dist]
        a_vals   = [a_dist[k] * 100 for k in a_order if k in a_dist]
        colours  = [
            "#EF9F27" if v > 30 else "#E24B4A" if v < 3 else "#5B8DB8"
            for v in a_vals
        ]
        _hbar(ax, a_labels, a_vals, colours, "Age group distribution")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4: Accent Distribution (pipe-labels exploded) ────────────
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle("Accent/Dialect Distribution (pipe-labels split separately)",
                     fontsize=14, fontweight="bold")
        acc_dist = bias_scores.get(DIALECT_COL + "_exploded", {}).get(
            "distribution", {}
        )
        top_acc = dict(
            sorted(
                {k: v for k, v in acc_dist.items() if k}.items(),
                key=lambda x: x[1], reverse=True,
            )[:12]
        )
        if top_acc:
            acc_labels = [k[:45] for k in top_acc]
            acc_vals   = [v * 100 for v in top_acc.values()]
            _hbar(ax, acc_labels, acc_vals,
                  ["#6DBF67"] * len(acc_labels),
                  "Top 12 accents (after splitting on '|')")
        else:
            ax.text(0.5, 0.5, "No accent data available",
                    ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 5: Intersectional Heatmap ────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 6))
        ct = intersectional_bias(df)
        top_ages = ct.sum(axis=0).nlargest(8).index
        ct_plot  = ct[top_ages]
        sns.heatmap(
            ct_plot, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, linewidths=0.5,
            cbar_kws={"label": "% of total dataset"},
        )
        ax.set_title("Intersectional Bias – Gender × Age (% of total)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Gender")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 6: Bias Metric Summary ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Bias Metric Summary", fontsize=14, fontweight="bold")
        metrics   = ["gender", "age", DIALECT_COL]
        ent_vals  = [bias_scores.get(m, {}).get("normalized_entropy", 0)
                     for m in metrics]
        gini_vals = [bias_scores.get(m, {}).get("gini_coefficient", 0)
                     for m in metrics]
        bar_cols  = ["#E07B54", "#5B8DB8", "#6DBF67"]

        axes[0].bar(metrics, ent_vals, color=bar_cols)
        axes[0].set_title("Normalised Entropy (↑ better)",
                          fontsize=11, fontweight="bold")
        axes[0].set_ylim(0, 1.15)
        axes[0].axhline(1.0, color="green", linestyle="--",
                        label="Perfect balance")
        axes[0].legend(fontsize=8)
        for i, v in enumerate(ent_vals):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

        axes[1].bar(metrics, gini_vals, color=bar_cols)
        axes[1].set_title("Gini Coefficient (↓ better)",
                          fontsize=11, fontweight="bold")
        axes[1].set_ylim(0, 1.15)
        axes[1].axhline(0.0, color="green", linestyle="--",
                        label="Perfect balance")
        axes[1].legend(fontsize=8)
        for i, v in enumerate(gini_vals):
            axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        d = pdf.infodict()
        d["Title"]  = "Sound Check – SPS Corpus Bias Audit"
        d["Author"] = "audit.py"

    return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 64)
    print("  SOUND CHECK – SPS CORPUS BIAS AUDIT")
    print("=" * 64)
    print(f"  Total utterances : {len(df):,}")
    if "client_id" in df.columns:
        print(f"  Unique speakers  : {df['client_id'].nunique():,}")
    print(f"  Splits present   : {sorted(df['split'].dropna().astype(str).unique())}")

    # ── Documentation Debt ───────────────────────────────────────────────
    debt = documentation_debt_report(df)
    print("\n── Documentation Debt ──────────────────────────────────────────")
    for col, info in debt.items():
        bar = "█" * int(info["missing_pct"] / 5)
        severity = (
            "CRITICAL" if info["missing_pct"] > 70
            else "HIGH" if info["missing_pct"] > 40
            else "MODERATE" if info["missing_pct"] > 15
            else "LOW"
        )
        print(f"  {col:10s}: {info['missing_pct']:5.1f}% missing  "
              f"[{severity}]  {bar}")

    # ── Representation Bias ──────────────────────────────────────────────
    df_acc = explode_accents(df)   # exploded for fair accent counting

    bias_scores = {}
    print("\n── Representation Bias ─────────────────────────────────────────")

    for col in ["gender", "age"]:
        if col not in df.columns:
            continue
        scores = representation_bias(df, col)
        bias_scores[col] = scores
        chi_stat, chi_p = chi_square_uniformity(df, col)

        print(f"\n  [{col.upper()}]")
        print(f"    Norm. entropy  : {scores['normalized_entropy']:.3f}  "
              "(1.0 = perfectly balanced)")
        print(f"    Gini coeff.    : {scores['gini_coefficient']:.3f}  "
              "(0.0 = perfectly balanced)")
        if chi_stat is not None:
            verdict = "BIASED (p<0.05)" if chi_p < 0.05 else "OK"
            print(f"    χ² p-value     : {chi_p:.4f}  → {verdict}")
        top5 = sorted(
            scores["distribution"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        print("    Top-5 groups   : " +
              ", ".join(f"{k or '(missing)'}({v*100:.1f}%)" for k, v in top5))

    # Accent bias on pipe-exploded frame
    if DIALECT_COL in df_acc.columns:
        scores_acc = representation_bias(df_acc, DIALECT_COL)
        bias_scores[DIALECT_COL]                  = scores_acc
        bias_scores[DIALECT_COL + "_exploded"]    = scores_acc
        chi_stat, chi_p = chi_square_uniformity(df_acc, DIALECT_COL)
        print(f"\n  [ACCENTS]  (pipe-labels split → {len(df_acc):,} rows)")
        print(f"    Norm. entropy  : {scores_acc['normalized_entropy']:.3f}")
        print(f"    Gini coeff.    : {scores_acc['gini_coefficient']:.3f}")
        if chi_stat is not None:
            verdict = "BIASED (p<0.05)" if chi_p < 0.05 else "OK"
            print(f"    χ² p-value     : {chi_p:.4f}  → {verdict}")
        top5a = sorted(
            {k: v for k, v in scores_acc["distribution"].items() if k}.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        print("    Top-5 accents  : " +
              ", ".join(f"{k[:35]}({v*100:.1f}%)" for k, v in top5a))

    # ── Intersectional Gap ───────────────────────────────────────────────
    ct = intersectional_bias(df)
    print("\n── Intersectional Gap (Gender × Age, top 6 cells) ─────────────")
    stacked = ct.stack().sort_values(ascending=False)
    for idx, val in stacked.head(6).items():
        print(f"    {str(idx[0]):25s} × {str(idx[1]):10s}: {val:.2f}%")

    # ── Audit flags & recommendations ────────────────────────────────────
    flags = []

    g_info = bias_scores.get("gender", {})
    if g_info.get("normalized_entropy", 1) < 0.85:
        flags.append(
            "⚠  Gender is heavily skewed. This corpus uses 6 categories "
            "(female_feminine, male_masculine, intersex, transgender, "
            "non-binary, do_not_wish_to_say). Ensure ALL six are in the "
            "PrivacyModule vocabulary and fairness-loss group definitions."
        )

    a_info = bias_scores.get("age", {})
    if a_info.get("gini_coefficient", 0) > 0.30:
        twenties_pct = (
            a_info.get("distribution", {}).get("twenties", 0) * 100
        )
        flags.append(
            f"⚠  Age skewed: 'twenties' = {twenties_pct:.1f}% of labelled rows. "
            "Speakers aged 40–90 are critically under-represented."
        )

    acc_debt = debt.get(DIALECT_COL, {}).get("missing_pct", 0)
    if acc_debt > 70:
        flags.append(
            f"⚠  Accent metadata {acc_debt:.1f}% empty – largest documentation "
            "debt. Priority: community re-labelling or automatic accent detection."
        )

    acc_dist = bias_scores.get(DIALECT_COL + "_exploded", {}).get(
        "distribution", {}
    )
    top_pct = max((v for k, v in acc_dist.items() if k), default=0) * 100
    if top_pct > 30:
        flags.append(
            f"⚠  Single dominant accent = {top_pct:.1f}% of labelled rows. "
            "Non-US / non-UK accents need more coverage."
        )

    if debt.get("gender", {}).get("missing_pct", 0) > 40:
        flags.append(
            f"⚠  {debt['gender']['missing_pct']:.1f}% gender labels missing. "
            "Privacy-preserving transformation cannot target these rows."
        )

    print("\n── Audit Flags & Recommendations ──────────────────────────────")
    if flags:
        for f in flags:
            print(f"  {f}")
    else:
        print("  No critical issues found.")

    # ── Save outputs ─────────────────────────────────────────────────────
    rows = []
    for col in ["gender", "age", DIALECT_COL]:
        src = df_acc if col == DIALECT_COL else df
        if col not in src.columns:
            continue
        vc = (
            src[col].fillna("(missing)").astype(str)
            .str.strip().str.lower().value_counts()
        )
        for k, v in vc.items():
            rows.append({
                "field":    col,
                "category": k,
                "count":    int(v),
                "pct":      round(v / len(src) * 100, 2),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "audit_report.csv"), index=False
    )

    with open(os.path.join(output_dir, "bias_scores.json"), "w") as f:
        json.dump(
            {"documentation_debt": debt,
             "representation_bias": bias_scores,
             "flags": flags},
            f, indent=2,
        )

    pdf_path = create_plots(df, df_acc, debt, bias_scores, output_dir)

    print(f"\n  Plots  → {pdf_path}")
    print(f"  CSV    → {os.path.join(output_dir, 'audit_report.csv')}")
    print(f"  JSON   → {os.path.join(output_dir, 'bias_scores.json')}")
    print("=" * 64 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sound Check – SPS/Common Voice Bias Audit"
    )
    parser.add_argument(
        "--tsv", type=str, default=None,
        help="Path to a single TSV file (SPS corpus format)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Common Voice dir containing train/dev/test.tsv splits",
    )
    parser.add_argument("--output_dir", type=str, default="audit_output")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on synthetic data mirroring your real corpus statistics",
    )
    args = parser.parse_args()

    if args.demo or (args.tsv is None and args.data_dir is None):
        print("Running in DEMO mode (synthetic data matching real SPS stats).")
        df = generate_demo_data()
    elif args.tsv:
        df = load_tsv(args.tsv)
    else:
        df = load_common_voice(args.data_dir)

    run_audit(df, args.output_dir)


if __name__ == "__main__":
    main()