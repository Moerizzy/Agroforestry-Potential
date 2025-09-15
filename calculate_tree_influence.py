#!/usr/bin/env python3
"""
Berechnung des Baumeinflusses (Tree Influence Calculation)

Calculates tree influence values based on:
- TOF (Trees Outside Forest) classification data
- Height values from nDOM rasters
- Porosity values by class
- Range calculation: R(H) = α * H (α=10)
- Tree influence: B = R(H) * (1 - Pc)

Usage:
    python calculate_tree_influence.py
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge
import rasterstats

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def create_height_vrt(ndom_path: str, output_vrt: str) -> str:
    """
    Create a Virtual Raster Table (VRT) from all nDOM TIF files using GDAL.
    This is more memory-efficient than rasterio merge for large datasets.

    Args:
        ndom_path: Path to directory containing nDOM TIF files
        output_vrt: Output VRT file path

    Returns:
        Path to created VRT file
    """
    log.info(f"Creating VRT from nDOM rasters in {ndom_path}")

    # Find all TIF files
    tif_files = list(Path(ndom_path).glob("*.tif"))
    if not tif_files:
        tif_files = list(Path(ndom_path).glob("*.TIF"))

    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {ndom_path}")

    log.info(f"Found {len(tif_files)} TIF files")

    # Use GDAL to build VRT (more efficient than rasterio merge)
    import subprocess

    # Create list of file paths as strings
    file_list = [str(f) for f in tif_files]

    # Build VRT using gdalbuildvrt
    cmd = ["gdalbuildvrt", "-resolution", "highest", output_vrt] + file_list

    log.info("Building VRT with gdalbuildvrt...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info(f"✓ Created VRT: {output_vrt}")
    except subprocess.CalledProcessError as e:
        log.error(f"GDAL error: {e.stderr}")
        # Fallback to simple VRT creation
        log.info("Falling back to simple VRT creation...")
        create_simple_vrt(file_list, output_vrt)
    except FileNotFoundError:
        log.warning("gdalbuildvrt not found, using simple VRT creation...")
        create_simple_vrt(file_list, output_vrt)

    return output_vrt


def create_simple_vrt(file_list: List[str], output_vrt: str):
    """
    Create a simple VRT file manually when GDAL tools are not available.
    """
    # Get reference information from first file
    with rasterio.open(file_list[0]) as src:
        crs = src.crs
        dtype = src.dtypes[0]

    # Create VRT XML content
    vrt_content = f"""<VRTDataset>
    <SRS>{crs}</SRS>
    <VRTRasterBand dataType="{dtype}" band="1">
"""

    for i, file_path in enumerate(file_list):
        vrt_content += f"""        <SimpleSource>
            <SourceFilename relativeToVRT="0">{file_path}</SourceFilename>
            <SourceBand>1</SourceBand>
        </SimpleSource>
"""

    vrt_content += """    </VRTRasterBand>
</VRTDataset>"""

    # Write VRT file
    with open(output_vrt, "w") as f:
        f.write(vrt_content)

    log.info(f"✓ Created simple VRT: {output_vrt}")


def get_porosity_values() -> Dict[str, float]:
    """
    Get porosity values (Pc) for each TOF class.
    Maps classvalue numbers to class names and porosity values.

    Returns:
        Dictionary mapping class values to porosity values
    """
    # Map classvalue to class names and porosity
    return {
        "1": 0.20,  # Forest: Pc = 0.20
        "2": 0.35,  # Patch: Pc = 0.35
        "3": 0.45,  # Linear: Pc = 0.45
        "4": 0.65,  # Tree: Pc = 0.65
        1: 0.20,  # Handle numeric values too
        2: 0.35,
        3: 0.45,
        4: 0.65,
    }


def get_class_names() -> Dict[str, str]:
    """
    Get human-readable class names for classvalue codes.

    Returns:
        Dictionary mapping class values to class names
    """
    return {
        "1": "Forest",
        "2": "Patch",
        "3": "Linear",
        "4": "Tree",
        1: "Forest",
        2: "Patch",
        3: "Linear",
        4: "Tree",
    }


def calculate_tree_influence(
    height: float, porosity: float, alpha: float = 7.0
) -> float:
    """
    Calculate tree influence value.

    Args:
        height: Tree height (H) in meters
        porosity: Porosity value (Pc) for the class
        alpha: Range factor (default: 7)

    Returns:
        Tree influence value B = R(H) * (1 - Pc) where R(H) = α * H
    """
    if np.isnan(height) or height <= 0:
        return 0.0

    range_value = alpha * height  # R(H) = α * H
    influence = range_value * (1 - porosity)  # B = R(H) * (1 - Pc)

    return influence


def process_tof_objects(
    tof_gpkg: str, height_vrt: str, output_path: str
) -> gpd.GeoDataFrame:
    """
    Process TOF objects to calculate tree influence values.

    Args:
        tof_gpkg: Path to TOF GPKG file
        height_vrt: Path to height VRT file
        output_path: Output file path

    Returns:
        GeoDataFrame with calculated influence values
    """
    log.info("Loading TOF data...")
    tof_gdf = gpd.read_file(tof_gpkg)
    log.info(f"Loaded {len(tof_gdf)} TOF objects")

    # Check available columns
    log.info(f"Available columns: {list(tof_gdf.columns)}")

    # Use classvalue column which contains the class codes (1-4)
    class_column = "classvalue"

    if class_column not in tof_gdf.columns:
        raise ValueError(
            f"Expected column '{class_column}' not found. Available columns: {list(tof_gdf.columns)}"
        )

    log.info(f"Using class column: {class_column}")
    log.info(f"Unique class values: {sorted(tof_gdf[class_column].dropna().unique())}")

    # Get porosity and class name mappings
    porosity_map = get_porosity_values()
    class_names = get_class_names()

    # Calculate height statistics using rasterstats
    log.info("Calculating height statistics (mean height)...")

    # Calculate mean height for each object (changed from 95th percentile)
    height_stats = rasterstats.zonal_stats(
        tof_gdf.geometry,
        height_vrt,
        stats=["mean", "max", "count"],
        nodata=np.nan,
        all_touched=True,
    )

    # Add height statistics to GeoDataFrame
    tof_gdf["height_mean"] = [
        stats["mean"] if stats["mean"] is not None else np.nan for stats in height_stats
    ]
    tof_gdf["height_max"] = [
        stats["max"] if stats["max"] is not None else np.nan for stats in height_stats
    ]
    tof_gdf["pixel_count"] = [
        stats["count"] if stats["count"] is not None else 0 for stats in height_stats
    ]

    # Add porosity values and class names based on classvalue
    log.info("Adding porosity values and class names...")
    tof_gdf["porosity"] = tof_gdf[class_column].map(porosity_map)
    tof_gdf["class_name"] = tof_gdf[class_column].map(class_names)

    # Handle unmapped classes
    unmapped = tof_gdf["porosity"].isna()
    if unmapped.any():
        log.warning(f"Found {unmapped.sum()} objects with unmapped class values:")
        for cls in tof_gdf.loc[unmapped, class_column].unique():
            log.warning(f"  - {cls}")
        # Set default porosity for unmapped classes
        tof_gdf.loc[unmapped, "porosity"] = 0.5
        tof_gdf.loc[unmapped, "class_name"] = "Unknown"

    # Calculate tree influence
    log.info("Calculating tree influence values...")
    tof_gdf["range_m"] = 7.0 * tof_gdf["height_mean"]  # R(H) = α * H, α=7
    tof_gdf["tree_influence"] = tof_gdf["range_m"] * (
        1 - tof_gdf["porosity"]
    )  # B = R(H) * (1 - Pc)

    # Handle NaN values
    tof_gdf["range_m"] = tof_gdf["range_m"].fillna(0.0)
    tof_gdf["tree_influence"] = tof_gdf["tree_influence"].fillna(0.0)

    # Add area calculation for reference
    if tof_gdf.crs != "EPSG:25833":
        tof_utm = tof_gdf.to_crs("EPSG:25833")
        tof_gdf["area_m2"] = tof_utm.geometry.area
    else:
        tof_gdf["area_m2"] = tof_gdf.geometry.area

    # Create summary statistics
    log.info("\n" + "=" * 50)
    log.info("TREE INFLUENCE CALCULATION RESULTS")
    log.info("=" * 50)

    for class_val in sorted(tof_gdf[class_column].dropna().unique()):
        mask = tof_gdf[class_column] == class_val
        count = mask.sum()
        class_name = class_names.get(class_val, f"Unknown({class_val})")
        avg_height = tof_gdf.loc[mask, "height_mean"].mean()
        avg_influence = tof_gdf.loc[mask, "tree_influence"].mean()
        porosity = tof_gdf.loc[mask, "porosity"].iloc[0] if count > 0 else np.nan

        log.info(f"{class_name} (Class {class_val}):")
        log.info(f"  Objects: {count}")
        log.info(f"  Porosity (Pc): {porosity:.2f}")
        log.info(f"  Avg Height (mean): {avg_height:.1f} m")
        log.info(f"  Avg Influence (B): {avg_influence:.1f} m")
        log.info("")

    # Overall statistics
    valid_height = tof_gdf["height_mean"].notna()
    log.info(f"Overall Statistics:")
    log.info(f"  Total objects: {len(tof_gdf)}")
    log.info(f"  Objects with height data: {valid_height.sum()}")
    log.info(
        f"  Height range: {tof_gdf.loc[valid_height, 'height_mean'].min():.1f} - {tof_gdf.loc[valid_height, 'height_mean'].max():.1f} m"
    )
    log.info(
        f"  Influence range: {tof_gdf['tree_influence'].min():.1f} - {tof_gdf['tree_influence'].max():.1f} m"
    )

    # Save results
    log.info(f"Saving results to {output_path}")
    tof_gdf.to_file(output_path, driver="GPKG")

    # Also save as CSV for analysis
    csv_path = output_path.replace(".gpkg", ".csv")
    tof_csv = tof_gdf.drop(columns=["geometry"])
    tof_csv.to_csv(csv_path, index=False)
    log.info(f"✓ Saved CSV: {csv_path}")

    return tof_gdf


def create_influence_raster(
    tof_gdf: gpd.GeoDataFrame,
    height_vrt: str,
    output_raster: str,
    pixel_size: float = 25.0,
) -> str:
    """
    Create a raster map of tree influence values.

    Args:
        tof_gdf: GeoDataFrame with tree influence values
        height_vrt: Height VRT for reference
        output_raster: Output raster path
        pixel_size: Pixel size in meters

    Returns:
        Path to created raster
    """
    log.info("Creating tree influence raster...")

    # Get reference information from height VRT
    with rasterio.open(height_vrt) as src:
        bounds = src.bounds
        crs = src.crs

    # Calculate raster dimensions
    width = int((bounds.right - bounds.left) / pixel_size)
    height = int((bounds.top - bounds.bottom) / pixel_size)

    # Create transform
    transform = rasterio.transform.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, width, height
    )

    # Rasterize tree influence values
    from rasterio.features import rasterize

    # Create shapes with influence values
    shapes = [
        (geom, influence)
        for geom, influence in zip(tof_gdf.geometry, tof_gdf["tree_influence"])
    ]

    # Rasterize
    influence_array = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.float32
    )

    # Save raster
    profile = {
        "driver": "GTiff",
        "dtype": np.float32,
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(influence_array, 1)

    log.info(f"✓ Created influence raster: {output_raster}")
    return output_raster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tof-data",
        default="/home/morizzi/git/Agroforest_Potential/data/TOF/BB_TOF.gpkg",
        help="Path to TOF GPKG file",
    )
    parser.add_argument(
        "--ndom-path",
        default="/home/morizzi/git/Agroforest_Potential/data/nDOM",
        help="Path to nDOM raster directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/morizzi/git/Agroforest_Potential/tree_influence_results",
        help="Output directory",
    )
    parser.add_argument(
        "--create-raster", action="store_true", help="Also create influence raster map"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=25.0,
        help="Pixel size for raster output (default: 25m)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log.info("🌳 Tree Influence Calculation")
    log.info("=" * 40)
    log.info(f"TOF data: {args.tof_data}")
    log.info(f"nDOM path: {args.ndom_path}")
    log.info(f"Output dir: {args.output_dir}")

    try:
        # Step 1: Create VRT from nDOM rasters
        vrt_path = output_dir / "height_mosaic.vrt"
        create_height_vrt(args.ndom_path, str(vrt_path))

        # Step 2: Process TOF objects
        output_gpkg = output_dir / "tof_with_influence.gpkg"
        tof_result = process_tof_objects(args.tof_data, str(vrt_path), str(output_gpkg))

        # Step 3: Create influence raster (optional)
        if args.create_raster:
            output_raster = output_dir / "tree_influence_raster.tif"
            create_influence_raster(
                tof_result, str(vrt_path), str(output_raster), args.pixel_size
            )

        log.info("✅ Tree influence calculation completed successfully!")
        log.info(f"📂 Results saved in: {output_dir}")

    except Exception as e:
        log.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
