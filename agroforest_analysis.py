#!/usr/bin/env python3
"""
Agroforestry Potential Analysis

Analyzes field blocks for agroforestry potential based on environmental features.
Uses quantile-based potential index for individual field assessment.

Usage:
    python agroforest_analysis.py --input final_alpha7_tree_zones/feldblocks_with_features.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def map_columns_to_features(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map actual column names to f_* feature names according to specification.

    Args:
        df: DataFrame with field data

    Returns:
        Dictionary mapping feature names to actual column names
    """
    # Define the mapping from actual columns to f_* names
    column_mapping = {
        "area_ha": "f_size",
        "compactness": "f_shape",
        "eros_wasser_mean": "f_eros_wasser",
        "eros_wind_mean": "f_eros_wind",
        "dry_mean": "f_dry",
        "nfk_mean": "f_nfk",
        "bodenertrag_class": "f_soil",
        "tree_coverage_pct": "f_tof_gap",
        "HBN_KAT": "f_hbn_kat",
    }

    # Check which columns exist and create the mapping
    available_mapping = {}
    for actual_col, f_col in column_mapping.items():
        if actual_col in df.columns:
            available_mapping[actual_col] = f_col
        else:
            log.warning(f"Column {actual_col} not found in data")

    return available_mapping


def prepare_features(df, benefit_features, cost_features):
    """
    Prepare features according to specification:
    1. Median imputation for missing values
    2. Winsorizing (1st-99th percentile clipping)
    3. Create both quantile-based and standardized versions
    """
    log.info("Preparing features according to specification...")

    # Combine all features
    feature_columns = benefit_features + cost_features

    # Create feature dataframe
    f_data = df[feature_columns].copy()

    log.info(f"Working with {len(feature_columns)} features: {feature_columns}")

    # Special handling for TOF gap: convert tree coverage to gap
    if "f_tof_gap" in f_data.columns:
        raw = f_data["f_tof_gap"].astype(float)
        # robust auf 0..1 bringen und in "Gap" (hoch = wenig Bäume) invertieren
        if raw.max() > 1.0:  # wohl 0..100 %
            f_data["f_tof_gap"] = 1.0 - (raw.clip(0, 100) / 100.0)
        else:  # bereits 0..1
            f_data["f_tof_gap"] = 1.0 - raw.clip(0, 1)
        log.info("  Converted tree coverage to gap (high = more potential)")

    # Special handling for HBN_KAT: convert categories to numeric values
    if "f_hbn_kat" in f_data.columns:
        hbn_mapping = {
            "AL": 1.0,
            "GL": 0.5,
            "DK": 0.0,
        }  # AL good for agroforestry, GL moderate, DK poor
        f_data["f_hbn_kat"] = f_data["f_hbn_kat"].map(hbn_mapping)

        # Handle any unmapped values with median
        if f_data["f_hbn_kat"].isnull().any():
            median_val = f_data["f_hbn_kat"].median()
            unmapped = f_data["f_hbn_kat"].isnull().sum()
            f_data["f_hbn_kat"].fillna(median_val, inplace=True)
            log.info(
                f"  Mapped HBN_KAT categories (AL=1.0, GL=0.5, DK=0.0), filled {unmapped} unmapped values"
            )
        else:
            log.info("  Mapped HBN_KAT categories (AL=1.0, GL=0.5, DK=0.0)")

    # Step 1: Median imputation
    log.info("Step 1: Imputing missing values with median...")
    for col in f_data.columns:
        if f_data[col].isnull().any():
            median_val = f_data[col].median()
            f_data[col].fillna(median_val, inplace=True)

    # Step 2: Winsorizing (1st-99th percentile)
    log.info("Step 2: Winsorizing outliers (1st-99th percentile)...")
    for col in f_data.columns:
        # Skip categorical/ordinal features since they're already properly scaled
        if col in ["f_hbn_kat", "f_soil"]:
            log.info(
                f"  Skipped {col}: already properly scaled categorical/ordinal data"
            )
            continue

        p1, p99 = np.percentile(f_data[col], [1, 99])
        old_vals = f_data[col].copy()
        f_data[col] = np.clip(f_data[col], p1, p99)

        # Count outliers
        low_outliers = (old_vals < p1).sum()
        high_outliers = (old_vals > p99).sum()
        log.info(f"  Clipped {col}: {low_outliers} low, {high_outliers} high outliers")

    # Store original cleaned features
    X = f_data.copy()

    # Step 3A: Quantile features (for potential index)
    log.info("Step 3A: Creating quantile-based features (Variant A)...")
    Q = f_data.copy()

    # Transform to quantiles (0-1 scale where 1 = best)
    for col in Q.columns:
        Q[col] = Q[col].rank(pct=True)

    # Invert cost features in quantile space
    for cost_feature in cost_features:
        if cost_feature in Q.columns:
            Q[cost_feature] = 1.0 - Q[cost_feature]
            log.info(f"  Inverted cost feature {cost_feature} in quantile space")

    # Step 3B: Standardized features (for clustering)
    log.info("Step 3B: Creating standardized features (Variant B)...")
    Xs = f_data.copy()

    # Handle ordinal soil class (0=unknown, 1=good ... 4=poor) - convert to 0..1 benefit scale
    if "f_soil" in Xs.columns and Xs["f_soil"].between(0, 4).all():
        # Map: 0→0.5 (neutral), 1→1.0 (best), 2→0.67, 3→0.33, 4→0.0 (worst)
        soil_mapping = {0: 0.5, 1: 1.0, 2: 0.67, 3: 0.33, 4: 0.0}
        Xs["f_soil"] = Xs["f_soil"].map(soil_mapping)
        log.info(
            "  Converted ordinal soil class to 0-1 benefit scale (0→0.5, 1→1.0, 2→0.67, 3→0.33, 4→0.0)"
        )
    else:
        # Fallback: generic sign flip for cost features
        for cost_feature in cost_features:
            if cost_feature in Xs.columns:
                Xs[cost_feature] = -Xs[cost_feature]
                log.info(f"  Flipped sign for cost feature {cost_feature}")

    # Standardize to mean=0, std=1
    scaler = StandardScaler()
    Xs_array = scaler.fit_transform(Xs.values)
    Xs = pd.DataFrame(Xs_array, columns=Xs.columns, index=Xs.index)

    # Log feature preparation summary
    log.info("Feature preparation completed")
    log.info(f"  Original (X): {X.shape}, range examples: {dict(X.iloc[0])}")
    log.info(f"  Quantile (Q): {Q.shape}, all values 0-1")
    log.info(f"  Standardized (Xs): {Xs.shape}, normalized ~N(0,1)")

    return X, Q, Xs


def calculate_potential_index(Q: pd.DataFrame) -> pd.Series:
    """
    Calculate agroforestry potential index from quantile features.

    Args:
        Q: DataFrame with quantile-normalized features (all 0-1, high=good)

    Returns:
        Series with potential index per field
    """
    log.info("Calculating potential index (mean of quantile features)...")

    # Simple mean across all quantile features
    potential = Q.mean(axis=1)

    log.info(f"Potential index calculated:")
    log.info(f"  Range: {potential.min():.3f} - {potential.max():.3f}")
    log.info(f"  Mean: {potential.mean():.3f} ± {potential.std():.3f}")

    return potential


def analyze_results(
    df: pd.DataFrame,
    X: pd.DataFrame,
    Q: pd.DataFrame,
    potential: pd.Series,
    output_dir: Path,
) -> None:
    """
    Analyze and visualize results.

    Args:
        df: Original DataFrame
        X: Original features (clipped & imputed)
        Q: Quantile features
        potential: Potential index
        output_dir: Output directory
    """
    log.info("Analyzing results...")

    # Handle optional ID columns
    id_cols = [c for c in ["gml_id", "FB_ID"] if c in df.columns]
    results = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    results["potential_index"] = potential

    # Use individual field-based classification with tertiles
    results["potential_class"] = pd.qcut(
        results["potential_index"], 3, labels=["low", "medium", "high"]
    )
    log.info("Using individual field-based tertile classification")

    # Add original features for interpretation
    for col in X.columns:
        results[f"original_{col}"] = X[col]

    # Add quantile features for interpretation
    for col in Q.columns:
        results[f"quantile_{col}"] = Q[col]

    # Save results
    results_path = output_dir / "agroforest_analysis.csv"
    results.to_csv(results_path, index=False)
    log.info(f"Saved complete results to {results_path}")

    # Save top candidates (handle optional ID columns)
    top_candidates_cols = ["potential_index"] + [
        c for c in ["gml_id", "FB_ID"] if c in results.columns
    ]
    top_candidates = results.nlargest(50, "potential_index")[top_candidates_cols]
    top_path = output_dir / "top_candidates.csv"
    top_candidates.to_csv(top_path, index=False)
    log.info(f"Saved top 50 candidates to {top_path}")

    # Create visualizations
    create_visualizations(potential, X, output_dir)

    # Print summary
    print_summary(potential, results)


def create_visualizations(
    potential: pd.Series,
    X: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create analysis visualizations."""

    try:
        import seaborn as sns

        has_seaborn = True
    except ImportError:
        has_seaborn = False
        log.warning("Seaborn not available, using matplotlib fallbacks")

    plt.style.use("default")

    # 1. Potential distribution
    plt.figure(figsize=(10, 6))
    plt.hist(potential, bins=30, alpha=0.7, edgecolor="black")
    plt.axvline(
        potential.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {potential.mean():.3f}",
    )
    plt.xlabel("Agroforestry Potential Index")
    plt.ylabel("Number of Fields")
    plt.title("Distribution of Potential Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "potential_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Feature correlation heatmap
    if has_seaborn:
        plt.figure(figsize=(10, 8))
        correlation_matrix = X.corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            fmt=".2f",
            square=True,
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            output_dir / "feature_correlation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    log.info("Visualizations created")


def print_summary(
    potential: pd.Series,
    results: pd.DataFrame = None,
) -> None:
    """Print analysis summary."""

    high_potential = (potential > 0.7).sum()
    medium_potential = ((potential > 0.5) & (potential <= 0.7)).sum()
    low_potential = (potential <= 0.5).sum()
    total = len(potential)

    # Calculate both old threshold-based and new tertile-based categories
    high_potential = (potential > 0.7).sum()
    medium_potential = ((potential >= 0.5) & (potential <= 0.7)).sum()
    low_potential = (potential < 0.5).sum()

    # Get tertile category counts from results if available
    tertile_counts = None
    if results is not None and "potential_class" in results.columns:
        tertile_counts = results["potential_class"].value_counts()

    # Summary output
    log.info("=" * 60)
    log.info("AGROFORESTRY ANALYSIS SUMMARY")
    log.info("=" * 60)
    log.info(f"Total fields analyzed: {total}")
    log.info(f"Potential index: {potential.mean():.3f} ± {potential.std():.3f}")
    log.info(f"Range: {potential.min():.3f} - {potential.max():.3f}")
    log.info("")
    log.info("Potential categories (threshold-based):")
    log.info(
        f"  High potential (>0.7): {high_potential} ({high_potential/total*100:.1f}%)"
    )
    log.info(
        f"  Medium potential (0.5-0.7): {medium_potential} ({medium_potential/total*100:.1f}%)"
    )
    log.info(
        f"  Low potential (≤0.5): {low_potential} ({low_potential/total*100:.1f}%)"
    )

    if tertile_counts is not None:
        log.info("")
        log.info("Potential categories (data-driven tertiles):")
        for category in ["low", "medium", "high"]:
            if category in tertile_counts:
                count = tertile_counts[category]
                log.info(
                    f"  {category.capitalize()} potential: {count} ({count/total*100:.1f}%)"
                )

    log.info("")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help="Path to field features CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="agroforest_analysis_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log.info("🌳 Agroforestry Potential Analysis (Individual Field Assessment)")
    log.info("=" * 60)
    log.info(f"Input file: {args.input}")
    log.info(f"Output directory: {args.output_dir}")

    try:
        # Load data
        log.info("Loading field data...")
        df = pd.read_csv(args.input)
        log.info(f"Loaded {len(df)} field blocks with {len(df.columns)} columns")

        # Map columns to features
        available_mappings = map_columns_to_features(df)
        log.info(f"Available feature mappings: {available_mappings}")

        # Create feature columns by renaming
        df_features = df.copy()
        for original_col, feature_name in available_mappings.items():
            if original_col in df.columns:
                df_features[feature_name] = df[original_col]

        # Define benefit and cost features according to specification
        benefit = [
            "f_dry",
            "f_eros_wasser",
            "f_eros_wind",
            "f_tof_gap",
            "f_size",
            "f_shape",
            "f_hbn_kat",
        ]
        cost = ["f_nfk", "f_soil"]

        # Filter to only include features that are available
        available_features = [f for f in benefit + cost if f in df_features.columns]
        benefit = [f for f in benefit if f in df_features.columns]
        cost = [f for f in cost if f in df_features.columns]

        log.info(
            f"Working with {len(available_features)} features: {available_features}"
        )
        log.info(f"Benefit features: {benefit}")
        log.info(f"Cost features: {cost}")

        # Prepare features
        X, Q, Xs = prepare_features(df_features, benefit, cost)

        # Calculate quantile-based potential index
        log.info("Calculating individual field potential index")
        potential = calculate_potential_index(Q)

        # Analyze results
        analyze_results(df, X, Q, potential, output_dir)

        log.info("✅ Analysis completed successfully!")
        log.info(f"📂 Results saved in: {output_dir}")

    except Exception as e:
        log.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
