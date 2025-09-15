#!/usr/bin/env python3
"""
Create GIS Visualization of Agroforestry Potential Analysis Results

Merges Variant A results with spatial data and creates 5 potential classes
for visualization in GIS software.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def create_potential_classes(potential_series, n_classes=5):
    """
    Create n classes from potential index using quantiles.

    Args:
        potential_series: Series with potential index values
        n_classes: Number of classes to create (default: 5)

    Returns:
        Series with class labels and class info dict
    """
    # Create quantile-based classes with robust handling of duplicates
    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"][:n_classes]

    # Calculate quantile boundaries
    quantiles = np.linspace(0, 1, n_classes + 1)
    class_boundaries = potential_series.quantile(quantiles).values

    # Create classes using qcut for equal-sized groups, handling duplicates
    try:
        classes = pd.qcut(
            potential_series, q=n_classes, labels=class_labels, precision=3
        )
    except ValueError as e:
        if "duplicate" in str(e).lower():
            # Handle duplicate values by using fewer classes or manual binning
            log.warning(
                f"Duplicate boundaries detected, adjusting classification approach"
            )

            # Try with fewer classes first
            unique_values = len(potential_series.unique())
            if unique_values < n_classes:
                log.warning(
                    f"Only {unique_values} unique values, reducing to {unique_values} classes"
                )
                n_classes = unique_values
                class_labels = class_labels[:n_classes]

            # Use manual percentile-based binning
            if n_classes == 1:
                classes = pd.Series(
                    ["All Fields"] * len(potential_series), index=potential_series.index
                )
                class_labels = ["All Fields"]
            else:
                percentiles = np.linspace(0, 100, n_classes + 1)
                boundaries = np.percentile(potential_series, percentiles)
                # Ensure boundaries are unique by adding small increments
                for i in range(1, len(boundaries)):
                    if boundaries[i] <= boundaries[i - 1]:
                        boundaries[i] = boundaries[i - 1] + 0.001
                classes = pd.cut(
                    potential_series,
                    bins=boundaries,
                    labels=class_labels,
                    include_lowest=True,
                )
        else:
            raise e

    # Create class info
    class_info = {}
    for i, label in enumerate(class_labels):
        mask = classes == label
        class_info[label] = {
            "count": mask.sum(),
            "percentage": mask.sum() / len(potential_series) * 100,
            "min_value": potential_series[mask].min(),
            "max_value": potential_series[mask].max(),
            "mean_value": potential_series[mask].mean(),
            "class_number": i + 1,
        }

    return classes, class_info, class_boundaries


def merge_with_spatial_data(results_csv, spatial_gpkg, output_path):
    """
    Merge analysis results with spatial data and create GIS-ready output.

    Args:
        results_csv: Path to CSV with analysis results
        spatial_gpkg: Path to GeoPackage with spatial field data
        output_path: Path for output GeoPackage
    """
    log.info("Loading analysis results...")
    results_df = pd.read_csv(
        "agroforest_analysis_som_updated/agroforest_analysis_variant_A.csv"
    )
    log.info(f"Loaded {len(results_df)} analysis results")

    log.info("Loading spatial field data...")
    spatial_gdf = gpd.read_file(spatial_gpkg)
    log.info(f"Loaded {len(spatial_gdf)} spatial features")

    # Create 5 potential classes
    log.info("Creating 5 potential classes (quintiles)...")
    classes, class_info, boundaries = create_potential_classes(
        results_df["potential_index"], n_classes=5
    )
    results_df["potential_class_5"] = classes
    results_df["potential_class_numeric"] = classes.cat.codes + 1  # 1-5 instead of 0-4

    # Log class information
    log.info("Potential Index Classification (5 Classes):")
    log.info("=" * 60)
    for label, info in class_info.items():
        log.info(f"{label} (Class {info['class_number']}):")
        log.info(f"  Count: {info['count']} fields ({info['percentage']:.1f}%)")
        log.info(f"  Range: {info['min_value']:.3f} - {info['max_value']:.3f}")
        log.info(f"  Mean: {info['mean_value']:.3f}")

    # Merge with spatial data
    log.info("Merging results with spatial data...")

    # Try different join strategies
    if "gml_id" in spatial_gdf.columns and "gml_id" in results_df.columns:
        merged_gdf = spatial_gdf.merge(results_df, on="gml_id", how="inner")
        log.info(f"Merged on 'gml_id': {len(merged_gdf)} features")
    elif "FB_ID" in spatial_gdf.columns and "FB_ID" in results_df.columns:
        merged_gdf = spatial_gdf.merge(results_df, on="FB_ID", how="inner")
        log.info(f"Merged on 'FB_ID': {len(merged_gdf)} features")
    else:
        # Try index-based merge as fallback
        log.warning("No common ID columns found, trying index merge...")
        merged_gdf = spatial_gdf.copy()
        for col in results_df.columns:
            if col not in ["gml_id", "FB_ID"]:
                merged_gdf[col] = results_df[col].values[: len(merged_gdf)]
        log.info(f"Index-based merge: {len(merged_gdf)} features")

    # Select key columns for GIS visualization
    essential_cols = [
        "geometry",
        "potential_index",
        "potential_class_5",
        "potential_class_numeric",
    ]
    id_cols = [col for col in ["gml_id", "FB_ID"] if col in merged_gdf.columns]
    feature_cols = [col for col in merged_gdf.columns if col.startswith("original_")]

    gis_cols = (
        essential_cols + id_cols + feature_cols[:8]
    )  # Limit to avoid too many columns
    gis_gdf = merged_gdf[gis_cols].copy()

    # Add class boundaries as metadata
    gis_gdf.attrs["class_boundaries"] = boundaries.tolist()
    gis_gdf.attrs["classification_method"] = "quintiles"
    gis_gdf.attrs["n_classes"] = 5

    # Save to GeoPackage
    log.info(f"Saving to GeoPackage: {output_path}")
    gis_gdf.to_file(output_path, driver="GPKG", layer="agroforest_potential")

    # Also create a simplified version with just essential columns
    simple_gdf = merged_gdf[essential_cols + id_cols].copy()
    simple_gdf.to_file(output_path, driver="GPKG", layer="agroforest_potential_simple")

    # Create a summary table for the legend
    summary_data = []
    for label, info in class_info.items():
        summary_data.append(
            {
                "class_name": label,
                "class_number": info["class_number"],
                "count": info["count"],
                "percentage": round(info["percentage"], 1),
                "min_potential": round(info["min_value"], 3),
                "max_potential": round(info["max_value"], 3),
                "mean_potential": round(info["mean_value"], 3),
            }
        )

    summary_df = pd.DataFrame(summary_data)

    # Save summary as additional layer (convert to geodataframe with dummy geometry for GPKG)
    summary_gdf = gpd.GeoDataFrame(summary_df, geometry=[None] * len(summary_df))
    summary_gdf.to_file(output_path, driver="GPKG", layer="class_summary")

    log.info("=" * 60)
    log.info("GIS VISUALIZATION FILES CREATED")
    log.info("=" * 60)
    log.info(f"Main output: {output_path}")
    log.info("Layers created:")
    log.info("  - agroforest_potential: Full data with all features")
    log.info("  - agroforest_potential_simple: Essential columns only")
    log.info("  - class_summary: Classification legend/summary")
    log.info("")
    log.info("GIS Styling Recommendations:")
    log.info("  - Use 'potential_class_numeric' for graduated symbology (1-5)")
    log.info("  - Color scheme: Red (Low) → Yellow (Medium) → Green (High)")
    log.info("  - Alternative: Use 'potential_index' for continuous color ramp")

    return gis_gdf, class_info


def main():
    # Define paths
    results_csv_a = Path("pipeline_full_run_analysis/agroforest_analysis_variant_A.csv")
    results_csv_b = Path("pipeline_full_run_analysis/agroforest_analysis_variant_B.csv")
    cluster_summary = Path("pipeline_full_run_analysis/cluster_summary_variant_B.csv")
    spatial_gpkg = Path("pipeline_full_run/feldblocks_with_features.gpkg")
    output_gpkg = Path("pipeline_full_run_gis/agroforest_potential_final_gis.gpkg")

    # Check if files exist
    if not results_csv_a.exists():
        log.error(f"Variant A results CSV not found: {results_csv_a}")
        return

    if not results_csv_b.exists():
        log.error(f"Variant B results CSV not found: {results_csv_b}")
        return

    if not spatial_gpkg.exists():
        log.error(f"Spatial GPKG not found: {spatial_gpkg}")
        return

    # Create the merged GIS output with both variants
    try:
        # Load both analysis results
        log.info("Loading Variant A results...")
        results_a = pd.read_csv(results_csv_a)
        log.info(f"Loaded {len(results_a)} Variant A analysis results")

        log.info("Loading Variant B results...")
        results_b = pd.read_csv(results_csv_b)
        log.info(f"Loaded {len(results_b)} Variant B analysis results")

        # Load cluster summary if available
        cluster_info = None
        if cluster_summary.exists():
            cluster_info = pd.read_csv(cluster_summary)
            log.info(f"Loaded cluster summary with {len(cluster_info)} clusters")

        # Load spatial data
        log.info("Loading spatial field data...")
        spatial_gdf = gpd.read_file(spatial_gpkg)
        log.info(f"Loaded {len(spatial_gdf)} spatial features")

        # Create combined results
        log.info("Merging Variant A and B results...")

        # Merge Variant A and B on common identifier
        if "gml_id" in results_a.columns and "gml_id" in results_b.columns:
            merged_results = results_a.merge(
                results_b,
                on="gml_id",
                suffixes=("_variant_a", "_variant_b"),
                how="inner",
            )
            join_column = "gml_id"
        elif "FB_ID" in results_a.columns and "FB_ID" in results_b.columns:
            merged_results = results_a.merge(
                results_b,
                on="FB_ID",
                suffixes=("_variant_a", "_variant_b"),
                how="inner",
            )
            join_column = "FB_ID"
        else:
            log.error("No common identifier found between variants")
            return

        log.info(f"Merged {len(merged_results)} records from both variants")

        # Create 5 classes for Variant A
        log.info("Creating 5 potential classes for Variant A (quintiles)...")
        classes_a, class_info_a, boundaries_a = create_potential_classes(
            merged_results["potential_index_variant_a"], n_classes=5
        )
        merged_results["potential_class_5_variant_a"] = classes_a
        merged_results["potential_class_numeric_variant_a"] = (
            classes_a.cat.codes + 1
        )  # 1-5 instead of 0-4

        # Create 5 classes for Variant B
        log.info("Creating 5 potential classes for Variant B (quintiles)...")
        classes_b, class_info_b, boundaries_b = create_potential_classes(
            merged_results["potential_index_variant_b"], n_classes=5
        )
        merged_results["potential_class_5_variant_b"] = classes_b
        merged_results["potential_class_numeric_variant_b"] = (
            classes_b.cat.codes + 1
        )  # 1-5 instead of 0-4

        # Log classification info for both variants
        log.info("Potential Index Classification - VARIANT A (5 Classes):")
        log.info("=" * 60)
        for i, (class_name, info) in enumerate(class_info_a.items(), 1):
            log.info(f"{class_name}:")
            log.info(f"  Count: {info['count']} fields ({info['percentage']:.1f}%)")
            log.info(f"  Range: {info['min_value']:.3f} - {info['max_value']:.3f}")
            log.info(f"  Mean: {info['mean_value']:.3f}")

        log.info("Potential Index Classification - VARIANT B (5 Classes):")
        log.info("=" * 60)
        for i, (class_name, info) in enumerate(class_info_b.items(), 1):
            log.info(f"{class_name}:")
            log.info(f"  Count: {info['count']} fields ({info['percentage']:.1f}%)")
            log.info(f"  Range: {info['min_value']:.3f} - {info['max_value']:.3f}")
            log.info(f"  Mean: {info['mean_value']:.3f}")

        # Merge with spatial data
        log.info("Merging combined results with spatial data...")
        if join_column in spatial_gdf.columns:
            gis_gdf = spatial_gdf.merge(merged_results, on=join_column, how="inner")
            log.info(f"Merged on '{join_column}': {len(gis_gdf)} features")
        else:
            log.error(f"Join column '{join_column}' not found in spatial data")
            return

        # Save to GeoPackage with multiple layers
        log.info(f"Saving to GeoPackage: {output_gpkg}")
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)

        # Layer 1: Complete data with all variants
        gis_gdf.to_file(output_gpkg, layer="agroforest_complete", driver="GPKG")
        log.info(f"Created {len(gis_gdf)} records")

        # Layer 2: Simplified view with essential columns only
        cluster_col = (
            "cluster_variant_b" if "cluster_variant_b" in gis_gdf.columns else "cluster"
        )
        ap_cluster_col = (
            "AP_cluster_variant_b"
            if "AP_cluster_variant_b" in gis_gdf.columns
            else "AP_cluster"
        )

        essential_cols = [
            "geometry",
            join_column,
            "FB_ID" if "FB_ID" in gis_gdf.columns else None,
            "HBN_KAT",
            "bodenertrag_class",
            "potential_index_variant_a",
            "potential_class_5_variant_a",
            "potential_class_numeric_variant_a",
            "potential_index_variant_b",
            "potential_class_5_variant_b",
            "potential_class_numeric_variant_b",
            cluster_col,
            ap_cluster_col,
        ]
        essential_cols = [
            col for col in essential_cols if col is not None and col in gis_gdf.columns
        ]
        gis_gdf[essential_cols].to_file(
            output_gpkg, layer="agroforest_simple", driver="GPKG"
        )
        log.info(f"Created {len(gis_gdf)} records")

        # Layer 3: Variant A classification summary (as CSV, not spatial)
        class_summary_a = pd.DataFrame(
            [
                {"class_name": name, "class_number": i, **info}
                for i, (name, info) in enumerate(class_info_a.items(), 1)
            ]
        )
        summary_csv_a = output_gpkg.parent / "variant_a_class_summary.csv"
        class_summary_a.to_csv(summary_csv_a, index=False)
        log.info(f"Created Variant A class summary: {summary_csv_a}")

        # Layer 4: Variant B classification summary (as CSV, not spatial)
        class_summary_b = pd.DataFrame(
            [
                {"class_name": name, "class_number": i, **info}
                for i, (name, info) in enumerate(class_info_b.items(), 1)
            ]
        )
        summary_csv_b = output_gpkg.parent / "variant_b_class_summary.csv"
        class_summary_b.to_csv(summary_csv_b, index=False)
        log.info(f"Created Variant B class summary: {summary_csv_b}")

        # Layer 5: SOM cluster summary (if available)
        if cluster_info is not None:
            # Add some basic geometry for the cluster summary (centroid of cluster fields)
            cluster_geoms = []
            cluster_data = []

            # Determine correct cluster column name
            cluster_col = (
                "cluster_variant_b"
                if "cluster_variant_b" in gis_gdf.columns
                else "cluster"
            )

            for _, cluster_row in cluster_info.iterrows():
                cluster_id = cluster_row["cluster"]
                cluster_fields = gis_gdf[gis_gdf[cluster_col] == cluster_id]
                if len(cluster_fields) > 0:
                    centroid = cluster_fields.geometry.centroid.unary_union.centroid
                    cluster_geoms.append(centroid)

                    # Add enhanced cluster information
                    cluster_data.append(
                        {
                            "cluster_id": cluster_id,
                            "cluster_potential": cluster_row.get("AP_cluster", "N/A"),
                            "field_count": len(cluster_fields),
                            "avg_potential_a": cluster_fields[
                                "potential_index_variant_a"
                            ].mean(),
                            "avg_potential_b": cluster_fields[
                                "potential_index_variant_b"
                            ].mean(),
                            **{
                                col: cluster_row[col]
                                for col in cluster_info.columns
                                if col not in ["cluster"]
                            },
                        }
                    )
                else:
                    cluster_geoms.append(None)
                    cluster_data.append(
                        {
                            "cluster_id": cluster_id,
                            "cluster_potential": cluster_row.get("AP_cluster", "N/A"),
                            "field_count": 0,
                            "avg_potential_a": None,
                            "avg_potential_b": None,
                            **{
                                col: cluster_row[col]
                                for col in cluster_info.columns
                                if col not in ["cluster"]
                            },
                        }
                    )

            cluster_gdf = gpd.GeoDataFrame(
                cluster_data, geometry=cluster_geoms, crs=gis_gdf.crs
            )
            cluster_gdf.to_file(output_gpkg, layer="som_clusters", driver="GPKG")
            log.info(f"Created {len(cluster_gdf)} SOM cluster records with centroids")

        # Print summary
        log.info("=" * 60)
        log.info("COMPLETE GIS VISUALIZATION FILES CREATED")
        log.info("=" * 60)
        log.info(f"Main output: {output_gpkg}")
        log.info("Layers created:")
        log.info("  - agroforest_complete: Full data with both variants")
        log.info("  - agroforest_simple: Essential columns only")
        log.info("  - variant_a_classes: Variant A classification legend")
        log.info("  - variant_b_classes: Variant B classification legend")
        if cluster_info is not None:
            log.info("  - som_clusters: SOM cluster summary with centroids")
        log.info("")
        log.info("GIS Styling Recommendations:")
        log.info(
            "  - Variant A: Use 'potential_class_numeric_variant_a' for graduated symbology (1-5)"
        )
        log.info(
            "  - Variant B: Use 'potential_class_numeric_variant_b' for graduated symbology (1-5)"
        )
        cluster_col = (
            "cluster_variant_b" if "cluster_variant_b" in gis_gdf.columns else "cluster"
        )
        log.info(f"  - SOM Clusters: Use '{cluster_col}' for categorical symbology")
        log.info("  - Color scheme: Red (Low) → Yellow (Medium) → Green (High)")
        log.info(
            "  - Alternative: Use potential_index fields for continuous color ramps"
        )

        log.info("✅ Complete GIS visualization files created successfully!")
        log.info(
            f"Final dataset: {len(gis_gdf)} features with both Variant A and B data"
        )
        log.info(f"Coordinate system: {gis_gdf.crs}")

    except Exception as e:
        log.error(f"❌ Error creating GIS visualization: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
