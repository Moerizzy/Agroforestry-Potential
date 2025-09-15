#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Agroforestry Data Aggregation Pipeline

This script combines WFS and WMS data sources to create a comprehensive
agroforestry analysis dataset. It:

1. Downloads Feldblöcke (field blocks) via WFS
2. Downloads various WMS raster layers
3. Downloads and rasterizes Bodenertrag (soil yield) data with classification
4. Performs zonal statistics to aggregate all data into field blocks
5. Outputs enhanced Feldblöcke with all environmental variables

Usage:
  python agroforest_aggregator.py --config-wfs wfs_services.yaml --config-wms services.yaml --aoi data/extent_bb.geojson --output output/

  # List WMS layers for debugging:
  python agroforest_aggregator.py --list-wms "https://inspire.brandenburg.de/services/boerosion_wms?language=ger"
"""

import os
import sys
import io
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import rasterio.mask as rio_mask
from ruamel.yaml import YAML
import requests
from requests.adapters import HTTPAdapter, Retry
from rasterstats import zonal_stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer

# Import our WFS downloader functions (defined below)
# from wfs_downloader import wfs_download, probe_wfs, bbox_from_vector

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("agroforest")


# ---- WFS Download Functions ----
def wfs_download(
    url: str,
    typename: str,
    out_path: Optional[str] = None,
    crs: str = "EPSG:25833",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    output_format: Optional[str] = None,
    chunk_size: int = 5000,
) -> gpd.GeoDataFrame:
    """
    Download data from WFS service using direct URL construction.

    Args:
        url: WFS service URL
        typename: Feature type name
        out_path: Optional output file path
        crs: Target CRS
        bbox: Bounding box as (minx, miny, maxx, maxy)
        output_format: Output format
        chunk_size: Chunk size for pagination

    Returns:
        GeoDataFrame with downloaded features
    """
    try:
        # Construct WFS GetFeature URL
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typename": typename,
            "srsName": crs,
            "outputFormat": output_format or "application/gml+xml; version=3.2",
        }

        # Add bbox if provided
        if bbox:
            params["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{crs}"

        # Add pagination if chunk_size specified
        if chunk_size > 0:
            params["count"] = chunk_size

        # Construct full URL
        import urllib.parse

        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        log.debug(f"WFS URL: {full_url}")

        # Use requests session for better error handling
        response = SESSION.get(url, params=params, timeout=120)
        response.raise_for_status()

        # Check if we got an exception report
        if (
            b"ExceptionReport" in response.content
            or b"ServiceException" in response.content
        ):
            error_msg = response.content.decode("utf-8", errors="ignore")
            raise Exception(f"WFS service error: {error_msg[:500]}...")

        # Save response to temporary file and read with geopandas
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".gml", delete=False
        ) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        try:
            # Read with geopandas
            gdf = gpd.read_file(tmp_path)

            if len(gdf) == 0:
                log.warning(f"No features returned from WFS: {typename}")
                return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)
            elif gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            # Save to file if requested
            if out_path:
                gdf.to_file(out_path, driver="GPKG")
                log.info(f"✓ Saved {len(gdf)} features to {out_path}")

            return gdf

        finally:
            # Clean up temporary file
            import os

            try:
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        log.error(f"WFS download failed for {typename}: {e}")
        # Return empty GeoDataFrame on error
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)


# ---- HTTP session with retries ----
def make_session():
    """Create HTTP session with retry strategy."""
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "agroforest-pipeline/2.0"})
    return s


SESSION = make_session()


def load_yaml(path: str) -> Dict:
    """Load YAML configuration file."""
    y = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        return y.load(f)


def list_wms_layers(url: str) -> List[str]:
    """List available layers from WMS service."""
    try:
        r = SESSION.get(
            url,
            params={"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"},
            timeout=(10, 60),
        )
        r.raise_for_status()
        text = r.text
        names = re.findall(r"<Layer>.*?<Name>([^<]+)</Name>", text, flags=re.S)
        return [n.strip() for n in names if n.strip()]
    except Exception as e:
        log.error(f"Failed to list WMS layers: {e}")
        return []


def read_aoi(path: str) -> gpd.GeoDataFrame:
    """Read Area of Interest from file."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("AOI has no CRS. Please set one (e.g., EPSG:25833)")
    if len(gdf) > 1:
        # Dissolve multiple geometries into one
        gdf = gdf[[gdf.geometry.name]].dissolve().reset_index(drop=True)
    return gdf


def add_geometry_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add geometric features like area and circumference to field blocks."""

    # Ensure we're in a projected CRS for accurate measurements
    if gdf.crs.is_geographic:
        # Transform to UTM zone 33N (EPSG:25833) for Brandenburg
        gdf = gdf.to_crs("EPSG:25833")

    # Calculate area in hectares
    gdf["area_ha"] = gdf.geometry.area / 10000

    # Calculate circumference/perimeter in meters
    gdf["circumference_m"] = gdf.geometry.length

    # Calculate shape compactness (4π × area / perimeter²)
    # Values close to 1 indicate circular shapes, lower values indicate elongated shapes
    gdf["compactness"] = (4 * np.pi * gdf.geometry.area) / (gdf.geometry.length**2)

    log.info(f"✓ Added geometry features: area_ha, circumference_m, compactness")

    return gdf


def download_feldblocks_wfs(
    aoi: gpd.GeoDataFrame, wfs_config: Dict, output_dir: str
) -> gpd.GeoDataFrame:
    """Download Feldblöcke using WFS service from configuration."""

    # Get Feldblöcke service config
    if "feldbloecke" not in wfs_config["wfs_services"]:
        raise ValueError("No 'feldbloecke' service found in WFS configuration")

    service = wfs_config["wfs_services"]["feldbloecke"]
    defaults = wfs_config.get("defaults", {})

    # Get bounding box from AOI
    target_crs = service.get("target_crs", "EPSG:25833")
    aoi_transformed = aoi.to_crs(target_crs)
    bbox = tuple(aoi_transformed.total_bounds)

    # Create output path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download using our WFS downloader
    log.info(f"Downloading Feldblöcke from {service['url']}")
    gdf = wfs_download(
        url=service["url"],
        typename=service["typename"],
        out_path=str(output_path / "feldbloecke_raw.gpkg"),
        crs=target_crs,
        bbox=bbox,
        output_format=service.get("output_format"),
        chunk_size=defaults.get("chunk_size", 5000),
    )

    # Clip to AOI if needed
    if len(gdf) > 0:
        log.info(f"Clipping {len(gdf)} features to AOI...")

        # First filter by intersection to reduce processing
        gdf_intersecting = gdf[gdf.intersects(aoi_transformed.union_all())]

        if len(gdf_intersecting) == 0:
            log.warning("No features intersect with AOI after clipping")
            return gpd.GeoDataFrame()

        # Now actually clip geometries to AOI boundaries
        log.info(
            f"Properly clipping {len(gdf_intersecting)} Feldblöcke geometries to study area..."
        )
        aoi_union = aoi_transformed.union_all()

        clipped_geoms = []
        valid_indices = []

        for idx, field in gdf_intersecting.iterrows():
            try:
                clipped_geom = field.geometry.intersection(aoi_union)

                # Only keep if resulting geometry is valid and has area
                if (
                    not clipped_geom.is_empty and clipped_geom.area > 1.0
                ):  # Minimum 1 m² area
                    clipped_geoms.append(clipped_geom)
                    valid_indices.append(idx)
            except Exception as e:
                log.warning(f"Failed to clip field {idx}: {e}")
                continue

        if len(clipped_geoms) == 0:
            log.warning("No valid fields remain after clipping to AOI")
            return gpd.GeoDataFrame()

        # Create new GeoDataFrame with clipped geometries
        gdf_clipped = gdf_intersecting.loc[valid_indices].copy()
        gdf_clipped.geometry = clipped_geoms
        gdf_clipped = gdf_clipped.reset_index(drop=True)

        log.info(
            f"Clipped {len(gdf_intersecting)} → {len(gdf_clipped)} valid Feldblöcke"
        )

        # Add geometry features (area, circumference, compactness) - recalculate after clipping
        gdf_clipped = add_geometry_features(gdf_clipped)

        # Create output file path
        output_file = (
            output_path / "feldbloecke.geojson"
        )  # Use GeoJSON to avoid GPKG FID issues

        # Clean and save
        gdf_clipped = gdf_clipped.reset_index(drop=True)
        gdf_clipped.to_file(output_file, driver="GeoJSON")
        log.info(f"✓ Saved {len(gdf_clipped)} Feldblöcke to {output_file}")
        return gdf_clipped
    else:
        raise RuntimeError("No Feldblöcke found in AOI")


def download_bodenertrag_wfs(
    aoi: gpd.GeoDataFrame, wfs_config: Dict, output_dir: str
) -> gpd.GeoDataFrame:
    """Download Bodenertrag (soil yield) data via WFS."""

    if "bodenertrag" not in wfs_config["wfs_services"]:
        raise ValueError("No 'bodenertrag' service found in WFS configuration")

    service = wfs_config["wfs_services"]["bodenertrag"]
    defaults = wfs_config.get("defaults", {})

    target_crs = service.get("target_crs", "EPSG:25833")
    aoi_transformed = aoi.to_crs(target_crs)
    bbox = tuple(aoi_transformed.total_bounds)

    output_path = Path(output_dir)

    log.info(f"Downloading Bodenertrag from {service['url']}")
    gdf = wfs_download(
        url=service["url"],
        typename=service["typename"],
        out_path=str(output_path / "bodenertrag_raw.gpkg"),
        crs=target_crs,
        bbox=bbox,
        output_format=service.get("output_format"),
        chunk_size=defaults.get("chunk_size", 5000),
    )

    if len(gdf) > 0:
        # Use intersection instead of overlay to avoid issues
        gdf_clipped = gdf[gdf.intersects(aoi_transformed.union_all())]

        if len(gdf_clipped) == 0:
            log.warning("No Bodenertrag features intersect with AOI")
            return gpd.GeoDataFrame()

        # Create output file and reset index
        gdf_clipped = gdf_clipped.reset_index(drop=True)
        output_file = output_path / "bodenertrag.geojson"  # Use GeoJSON
        gdf_clipped.to_file(output_file, driver="GeoJSON")
        log.info(f"✓ Saved {len(gdf_clipped)} Bodenertrag features to {output_file}")
        return gdf_clipped
    else:
        log.warning("No Bodenertrag data found in AOI")
        return gpd.GeoDataFrame()


def classify_bodenertrag(value: Any) -> int:
    """
    Classify Bodenertrag values according to scheme:
    1 = g/r (good/red)
    2 = g (good)
    3 = n/g (neutral/good)
    4 = g/n (good/neutral)
    """
    if pd.isna(value):
        return 0

    # Convert to string and check for classification patterns
    val_str = str(value).lower().strip()

    if "g/r" in val_str or "gut/rot" in val_str:
        return 1
    elif val_str == "g" or val_str == "gut":
        return 2
    elif "n/g" in val_str or "neutral/gut" in val_str:
        return 3
    elif "g/n" in val_str or "gut/neutral" in val_str:
        return 4
    else:
        # Try to map other common values
        if "gut" in val_str:
            return 2
        elif "neutral" in val_str:
            return 3
        else:
            return 0  # Unknown/unclassified


def rasterize_bodenertrag(
    gdf: gpd.GeoDataFrame, reference_bounds: Tuple, crs: str, resolution: float = 25.0
) -> np.ndarray:
    """
    Rasterize Bodenertrag data with classification.

    Args:
        gdf: Bodenertrag GeoDataFrame
        reference_bounds: (minx, miny, maxx, maxy) for raster extent
        crs: Target CRS
        resolution: Raster resolution in meters

    Returns:
        Tuple of (raster_array, transform, profile)
    """
    if len(gdf) == 0:
        log.warning("No Bodenertrag data to rasterize")
        return None, None, None

    # Ensure consistent CRS
    gdf = gdf.to_crs(crs)

    # Find classification column (look for common field names)
    class_column = None
    possible_columns = [
        "ertrag_kurz",  # Brandenburg Bodenertrag column name
        "etrag_kurz",  # Alternative spelling
        "bewertung",
        "klasse",
        "classification",
        "class",
        "value",
        "boertrag",
    ]

    for col in possible_columns:
        if col in gdf.columns:
            class_column = col
            break

    if class_column is None:
        # Use first non-geometry column as fallback
        non_geom_cols = [c for c in gdf.columns if c != gdf.geometry.name]
        if non_geom_cols:
            class_column = non_geom_cols[0]
            log.warning(f"Using column '{class_column}' for Bodenertrag classification")
        else:
            raise ValueError("No suitable column found for Bodenertrag classification")

    # Apply classification
    gdf["class_code"] = gdf[class_column].apply(classify_bodenertrag)

    # Calculate raster dimensions
    minx, miny, maxx, maxy = reference_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Rasterize
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf["class_code"])]
    raster = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
    )

    # Create profile
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    log.info(
        f"✓ Rasterized Bodenertrag: {width}x{height} pixels, resolution {resolution}m"
    )
    return raster, transform, profile


def request_wms_getmap(
    layer: Dict, bbox: Tuple, width: int, height: int, out_dir: str
) -> Tuple[bytes, str]:
    """Request WMS GetMap and return content and content type."""

    crs = layer.get("crs", "EPSG:3857")

    # Determine WMS version from URL
    url = layer["url"]
    if "VERSION=1.1" in url.upper():
        version = "1.1.1"
        crs_param = "SRS"  # WMS 1.1.1 uses SRS instead of CRS
    else:
        version = "1.3.0"
        crs_param = "CRS"

    # Handle axis order - WMS 1.1.1 always uses x,y order
    if version == "1.1.1" or not crs.upper().startswith("EPSG:4326"):
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    else:
        # WMS 1.3.0 with EPSG:4326 uses lat,lon order
        bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"

    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": version,
        "LAYERS": layer["name"],
        crs_param: crs,
        "BBOX": bbox_str,
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": layer.get("fmt", "image/png"),
        "TRANSPARENT": "TRUE" if layer.get("transparent", False) else "FALSE",
    }

    if layer.get("styles"):
        params["STYLES"] = layer["styles"]

    log.info(f"WMS GetMap {layer['id']} ({layer['name']})")

    r = SESSION.get(layer["url"], params=params, timeout=(10, 180))
    r.raise_for_status()

    ctype = r.headers.get("Content-Type", "").lower()

    # Check for error responses
    if "xml" in ctype or "html" in ctype:
        error_file = Path(out_dir) / f"{layer['id']}_error.xml"
        with open(error_file, "wb") as f:
            f.write(r.content)
        raise RuntimeError(f"WMS returned {ctype}. See {error_file}")

    return r.content, ctype


def georef_bytes_to_dataset(
    content: bytes, ctype: str, bbox: Tuple, crs: str, width: int, height: int
):
    """Convert image bytes to georeferenced rasterio dataset."""

    # Check if it's already a GeoTIFF
    if "tif" in ctype or content[:4] in (b"II*\x00", b"MM\x00*"):
        mem = MemoryFile(content)
        return mem.open()

    # Convert PNG/JPEG to georeferenced raster
    from PIL import Image

    img = Image.open(io.BytesIO(content)).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height))

    arr = np.array(img, dtype=np.float32)  # Use float32 instead of uint8
    transform = from_bounds(*bbox, width=width, height=height)

    mem = MemoryFile()
    with mem.open(
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",  # Use float32 instead of uint8
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=-9999,  # Use proper nodata value
    ) as dst:
        dst.write(arr, 1)

    return mem.open()


def pixels_for_bbox(bbox: Tuple, resolution: float) -> Tuple[int, int]:
    """Calculate pixel dimensions for given bbox and resolution."""
    minx, miny, maxx, maxy = bbox
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    return width, height


def process_local_raster(
    raster_config: Dict, aoi: gpd.GeoDataFrame, output_dir: str
) -> str:
    """Process a local raster file by clipping it to AOI and copying to output directory."""

    input_file = Path(raster_config["file"])
    if not input_file.exists():
        raise FileNotFoundError(f"Local raster file not found: {input_file}")

    # Create output directory
    output_raster_dir = Path(output_dir) / "rasters"
    output_raster_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_raster_dir / f"{raster_config['id']}.tif"

    log.info(f"Processing local raster {raster_config['id']} from {input_file}")

    # Read and clip raster to AOI
    with rasterio.open(input_file) as src:
        # Transform AOI to raster CRS
        aoi_transformed = aoi.to_crs(src.crs)

        # Mask the raster to AOI bounds
        out_image, out_transform = rio_mask.mask(
            src, aoi_transformed.geometry.values, crop=True, nodata=src.nodata or -9999
        )

        # Update metadata
        meta = src.meta.copy()
        meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src.nodata or -9999,
            }
        )

        # Write clipped raster
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(out_image)

    log.info(f"✓ Saved local raster {raster_config['id']} to {output_file}")
    return str(output_file)


def download_wms_layer(layer: Dict, aoi: gpd.GeoDataFrame, output_dir: str) -> str:
    """Download and process a single WMS layer."""

    # Transform AOI to layer CRS
    layer_crs = layer.get("crs", "EPSG:3857")
    aoi_transformed = aoi.to_crs(layer_crs)
    bbox = aoi_transformed.total_bounds

    # Calculate dimensions
    resolution = float(layer.get("res", 100.0))
    width, height = pixels_for_bbox(bbox, resolution)

    # Request WMS data
    content, ctype = request_wms_getmap(layer, bbox, width, height, output_dir)

    # Convert to georeferenced dataset
    ds = georef_bytes_to_dataset(content, ctype, bbox, layer_crs, width, height)

    # Mask to AOI and save
    output_file = Path(output_dir) / "rasters" / f"{layer['id']}.tif"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with ds as src:
        out_image, out_transform = rio_mask.mask(
            src, aoi_transformed.geometry.values, crop=True, nodata=np.nan
        )

        meta = src.meta.copy()
        meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": -9999,  # Use consistent nodata value
            }
        )

    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(out_image)

    log.info(f"✓ Saved WMS layer {layer['id']} to {output_file}")
    return str(output_file)


def aggregate_features_to_fields(
    fields: gpd.GeoDataFrame,
    raster_files: List[str],
    bodenertrag_raster: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Perform zonal statistics to aggregate raster values to field polygons."""

    result_gdf = fields.copy()

    # Process each raster file
    for raster_file in raster_files:
        layer_id = Path(raster_file).stem

        log.info(f"Aggregating {layer_id} to fields...")

        with rasterio.open(raster_file) as src:
            # Transform fields to raster CRS if needed
            fields_transformed = fields.to_crs(src.crs)

            # Calculate zonal statistics - only mean, with all_touched=True for better coverage
            stats = zonal_stats(
                fields_transformed,
                raster_file,
                stats=["mean"],
                nodata=src.nodata,
                all_touched=True,  # Include pixels that touch the polygon boundary
            )

            # Add mean as new column
            col_name = f"{layer_id}_mean"
            values = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
            result_gdf[col_name] = values

    # Process Bodenertrag raster if available
    if bodenertrag_raster and Path(bodenertrag_raster).exists():
        log.info("Aggregating Bodenertrag classification to fields...")

        with rasterio.open(bodenertrag_raster) as src:
            fields_transformed = fields.to_crs(src.crs)

            # For classification data, use mode (most common value)
            stats = zonal_stats(
                fields_transformed,
                bodenertrag_raster,
                stats=["majority", "count"],
                nodata=src.nodata,
                categorical=True,
                all_touched=False,
            )

            # Add Bodenertrag classification
            result_gdf["bodenertrag_class"] = [
                s["majority"] if s["majority"] is not None else 0 for s in stats
            ]

            # Add class descriptions
            class_map = {0: "unclassified", 1: "g/r", 2: "g", 3: "n/g", 4: "g/n"}
            result_gdf["bodenertrag_desc"] = result_gdf["bodenertrag_class"].map(
                class_map
            )

    return result_gdf


def interpolate_missing_values(
    gdf: gpd.GeoDataFrame, columns_to_interpolate: List[str]
) -> gpd.GeoDataFrame:
    """
    Interpolate missing values in specified columns using spatial nearest neighbor interpolation.

    Args:
        gdf: GeoDataFrame with missing values
        columns_to_interpolate: List of column names to interpolate

    Returns:
        GeoDataFrame with interpolated values
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.impute import KNNImputer

    result_gdf = gdf.copy()

    for col in columns_to_interpolate:
        if col not in gdf.columns:
            continue

        # Get coordinates for spatial interpolation
        coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
        values = gdf[col].values

        # Find missing values
        missing_mask = pd.isna(values)
        valid_mask = ~missing_mask

        if missing_mask.sum() == 0:
            log.info(f"No missing values in {col}")
            continue

        if valid_mask.sum() == 0:
            log.warning(f"All values missing in {col}, using median fallback")
            # If all values are missing, use the overall median (could be from other data sources)
            overall_median = np.nanmedian(gdf[col].values)
            if np.isnan(overall_median):
                overall_median = 0  # Ultimate fallback
            result_gdf[col] = overall_median
            continue

        # Spatial interpolation using k-nearest neighbors
        valid_coords = coords[valid_mask]
        valid_values = values[valid_mask]
        missing_coords = coords[missing_mask]

        # Use fewer neighbors if we don't have many valid points
        n_neighbors = min(5, len(valid_values))

        if n_neighbors >= 1:
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
            knn.fit(valid_coords, valid_values)
            interpolated_values = knn.predict(missing_coords)

            # Update the result
            new_values = values.copy()
            new_values[missing_mask] = interpolated_values
            result_gdf[col] = new_values

            log.info(
                f"Interpolated {missing_mask.sum()} missing values in {col} using {n_neighbors} neighbors"
            )
        else:
            log.warning(f"Not enough valid values for spatial interpolation in {col}")

    return result_gdf


def exclude_protected_areas(
    fields_gdf: gpd.GeoDataFrame, wfs_config: Dict, output_dir: Path = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Clip fields by removing areas that intersect with protected areas (nature conservation zones).
    Fields are geometrically clipped rather than completely excluded.

    Args:
        fields_gdf: Field blocks GeoDataFrame
        wfs_config: WFS configuration containing protected areas
        output_dir: Output directory for saving protected areas

    Returns:
        Tuple of (GeoDataFrame with fields clipped to exclude protected areas, protected areas GeoDataFrame)
    """
    protected_areas_config = wfs_config.get("protected_areas", {})
    areas_to_merge = protected_areas_config.get("merge_these", [])

    if not areas_to_merge:
        log.info("No protected areas configured for exclusion")
        # Return empty protected areas GeoDataFrame
        empty_protected = gpd.GeoDataFrame(columns=["geometry"], crs=fields_gdf.crs)
        return fields_gdf, empty_protected

    log.info("Downloading and merging protected areas...")

    all_protected = []
    protected_by_type = {}  # Store each type separately
    target_crs = "EPSG:25833"

    for area_type in areas_to_merge:
        if area_type not in wfs_config["wfs_services"]:
            log.warning(f"Protected area type {area_type} not found in WFS services")
            continue

        service = wfs_config["wfs_services"][area_type]

        try:
            # Get bounding box from fields
            fields_bounds = tuple(fields_gdf.to_crs(target_crs).total_bounds)

            # Download protected area
            protected_gdf = wfs_download(
                url=service["url"],
                typename=service["typename"],
                out_path=None,  # Don't save to file
                crs=target_crs,
                bbox=fields_bounds,
                output_format=service.get("output_format"),
                chunk_size=wfs_config.get("defaults", {}).get("chunk_size", 5000),
            )

            if len(protected_gdf) > 0:
                # Add area type column for identification
                protected_gdf["area_type"] = area_type
                all_protected.append(protected_gdf)
                protected_by_type[area_type] = protected_gdf
                log.info(f"Downloaded {len(protected_gdf)} {area_type} features")

        except Exception as e:
            log.warning(f"Failed to download {area_type}: {e}")

    if not all_protected:
        log.info("No protected areas downloaded, keeping all fields")
        # Return empty protected areas GeoDataFrame
        empty_protected = gpd.GeoDataFrame(columns=["geometry"], crs=fields_gdf.crs)
        return fields_gdf, empty_protected

    # Merge all protected areas
    merged_protected = pd.concat(all_protected, ignore_index=True)
    merged_protected = merged_protected.to_crs(fields_gdf.crs)

    # Save protected areas if output directory provided
    if output_dir:
        # Save individual protected area types
        for area_type, area_gdf in protected_by_type.items():
            area_gdf_corrected = area_gdf.to_crs(fields_gdf.crs)
            individual_path = output_dir / f"protected_areas_{area_type}.gpkg"
            area_gdf_corrected.to_file(individual_path, driver="GPKG")
            log.info(f"✓ Saved {area_type} to {individual_path}")

        # Also save combined file
        protected_areas_path = output_dir / "protected_areas_combined.gpkg"
        merged_protected.to_file(protected_areas_path, driver="GPKG")
        log.info(f"✓ Saved combined protected areas to {protected_areas_path}")

    # Create a union of all protected geometries for efficient clipping
    protected_union = merged_protected.geometry.union_all()

    # Clip fields by subtracting protected areas (difference operation)
    log.info("Clipping fields to remove protected area intersections...")

    clipped_fields = []
    original_count = len(fields_gdf)
    removed_count = 0

    for idx, field in fields_gdf.iterrows():
        if field.geometry.intersects(protected_union):
            # Calculate the difference (field minus protected areas)
            try:
                clipped_geom = field.geometry.difference(protected_union)

                # Check if there's remaining geometry after clipping
                if not clipped_geom.is_empty and clipped_geom.area > 0:
                    # Create new field record with clipped geometry
                    field_copy = field.copy()
                    field_copy.geometry = clipped_geom

                    # Recalculate area and circumference for clipped field
                    if field_copy.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                        # Convert to UTM for accurate area calculation
                        utm_geom = (
                            gpd.GeoSeries([field_copy.geometry], crs=fields_gdf.crs)
                            .to_crs("EPSG:25833")
                            .iloc[0]
                        )
                        field_copy["area_ha"] = (
                            utm_geom.area / 10000
                        )  # Convert m² to hectares
                        field_copy["circumference_m"] = utm_geom.length
                        # Recalculate compactness: 4π×area/perimeter²
                        if field_copy["circumference_m"] > 0:
                            field_copy["compactness"] = (
                                4 * 3.14159 * utm_geom.area
                            ) / (field_copy["circumference_m"] ** 2)
                        else:
                            field_copy["compactness"] = 0

                    clipped_fields.append(field_copy)
                else:
                    # Field is completely within protected area
                    removed_count += 1
            except Exception as e:
                log.warning(f"Failed to clip field {idx}: {e}. Keeping original field.")
                clipped_fields.append(field)
        else:
            # Field doesn't intersect protected areas, keep as-is
            clipped_fields.append(field)

    if not clipped_fields:
        log.warning("No fields remaining after clipping protected areas!")
        return gpd.GeoDataFrame(columns=fields_gdf.columns, crs=fields_gdf.crs)

    # Create result GeoDataFrame
    fields_clipped = gpd.GeoDataFrame(clipped_fields, crs=fields_gdf.crs)

    # Remove fields that are now too small (less than 0.01 ha)
    min_area = 0.01
    before_size_filter = len(fields_clipped)
    fields_clipped = fields_clipped[fields_clipped["area_ha"] >= min_area]
    size_filtered_count = before_size_filter - len(fields_clipped)

    clipped_count = original_count - len(fields_clipped)
    log.info(f"Field clipping results:")
    log.info(f"  Original fields: {original_count}")
    log.info(f"  Completely removed (within protected areas): {removed_count}")
    log.info(f"  Removed (too small after clipping): {size_filtered_count}")
    log.info(f"  Final fields for agroforestry: {len(fields_clipped)}")
    log.info(
        f"  Total reduction: {clipped_count} fields ({clipped_count/original_count*100:.1f}%)"
    )

    return fields_clipped, merged_protected


def calculate_tree_influence_for_fields(
    fields_gdf: gpd.GeoDataFrame, tof_file: str = None, output_dir: Path = None
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Calculate tree influence coverage for field blocks.

    Args:
        fields_gdf: Field blocks GeoDataFrame
        tof_file: Path to TOF with influence data (optional, will look for default)
        output_dir: Output directory for saving tree influence zones

    Returns:
        Tuple of (GeoDataFrame with tree influence columns added, tree influence zones GeoDataFrame)
    """
    if tof_file is None:
        # Look for TOF results in common locations
        possible_paths = [
            "tree_influence_results/tof_with_influence.gpkg",
            "tree_influence_test/tof_with_influence.gpkg",
            "tree_influence_mean_height/tof_with_influence.gpkg",
            "data/TOF/tof_with_influence.gpkg",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                tof_file = path
                break

        if tof_file is None:
            log.warning(
                "No TOF influence data found. Skipping tree influence calculation."
            )
            # Add empty columns
            fields_gdf["tree_influence_mean"] = 0.0
            fields_gdf["tree_influence_max"] = 0.0
            fields_gdf["tree_coverage_pct"] = 0.0
            # Return empty influence zones
            empty_zones = gpd.GeoDataFrame(columns=["geometry"], crs=fields_gdf.crs)
            return fields_gdf, empty_zones

    log.info(f"Loading TOF influence data from {tof_file}")

    try:
        # Load TOF data with influence values
        tof_gdf = gpd.read_file(tof_file)

        if "tree_influence" not in tof_gdf.columns:
            log.warning(
                "TOF file missing 'tree_influence' column. Skipping tree influence calculation."
            )
            fields_gdf["tree_influence_mean"] = 0.0
            fields_gdf["tree_influence_max"] = 0.0
            fields_gdf["tree_coverage_pct"] = 0.0
            # Return empty influence zones
            empty_zones = gpd.GeoDataFrame(columns=["geometry"], crs=fields_gdf.crs)
            return fields_gdf, empty_zones

        log.info(f"Loaded {len(tof_gdf)} TOF objects with influence data")

        # Ensure same CRS
        if tof_gdf.crs != fields_gdf.crs:
            tof_gdf = tof_gdf.to_crs(fields_gdf.crs)

        # Create influence zones (buffers around each TOF object)
        log.info("Creating tree influence zones...")
        tof_influence_zones = tof_gdf.copy()
        tof_influence_zones["geometry"] = tof_gdf.buffer(tof_gdf["tree_influence"])

        # Calculate overlaps with field blocks
        log.info("Calculating tree influence for field blocks...")

        tree_influence_stats = []

        for idx, field in fields_gdf.iterrows():
            field_geom = field.geometry

            # Find overlapping influence zones
            overlapping = tof_influence_zones[
                tof_influence_zones.intersects(field_geom)
            ]

            if len(overlapping) == 0:
                # No tree influence
                stats = {
                    "tree_influence_mean": 0.0,
                    "tree_influence_max": 0.0,
                    "tree_coverage_pct": 0.0,
                }
            else:
                # Calculate influence statistics
                influence_values = overlapping["tree_influence"].values

                # Calculate coverage percentage
                # Union all overlapping influence zones and intersect with field
                influence_union = overlapping.geometry.union_all()
                intersection = field_geom.intersection(influence_union)

                if intersection.is_empty:
                    coverage_pct = 0.0
                else:
                    coverage_pct = (intersection.area / field_geom.area) * 100

                stats = {
                    "tree_influence_mean": np.mean(influence_values),
                    "tree_influence_max": np.max(influence_values),
                    "tree_coverage_pct": min(coverage_pct, 100.0),  # Cap at 100%
                }

            tree_influence_stats.append(stats)

        # Add results to fields GeoDataFrame
        influence_df = pd.DataFrame(tree_influence_stats)
        for col in influence_df.columns:
            fields_gdf[col] = influence_df[col].values

        # Log summary statistics
        log.info(f"Tree influence summary:")
        log.info(
            f"  Fields with tree influence: {(fields_gdf['tree_coverage_pct'] > 0).sum()}"
        )
        log.info(f"  Average coverage: {fields_gdf['tree_coverage_pct'].mean():.1f}%")
        log.info(f"  Maximum coverage: {fields_gdf['tree_coverage_pct'].max():.1f}%")
        log.info(
            f"  Average influence value: {fields_gdf['tree_influence_mean'].mean():.1f}m"
        )

        # Save tree influence zones if output directory provided
        if output_dir:
            influence_zones_path = output_dir / "tree_influence_zones.gpkg"

            # Dissolve overlapping influence zones to create a clean coverage map
            log.info("Dissolving overlapping tree influence zones...")

            try:
                # Create a dissolved version for cleaner visualization
                dissolved_zones = tof_influence_zones.dissolve()
                dissolved_zones = dissolved_zones.reset_index(drop=True)

                # Add some basic statistics to dissolved zones
                dissolved_zones["total_tof_objects"] = len(tof_influence_zones)
                dissolved_zones["area_ha"] = dissolved_zones.geometry.area / 10000

                # Save dissolved version
                dissolved_zones.to_file(influence_zones_path, driver="GPKG")
                log.info(
                    f"✓ Saved dissolved tree influence zones to {influence_zones_path}"
                )
                log.info(f"  Original TOF objects: {len(tof_influence_zones)}")
                log.info(f"  Dissolved zones: {len(dissolved_zones)}")
                log.info(
                    f"  Total influence area: {dissolved_zones['area_ha'].sum():.1f} ha"
                )

                # Also save individual zones for detailed analysis if needed
                individual_zones_path = (
                    output_dir / "tree_influence_zones_individual.gpkg"
                )
                tof_influence_zones.to_file(individual_zones_path, driver="GPKG")
                log.info(
                    f"✓ Saved individual tree influence zones to {individual_zones_path}"
                )

            except Exception as e:
                log.warning(
                    f"Could not dissolve influence zones ({e}), saving individual zones only"
                )
                tof_influence_zones.to_file(influence_zones_path, driver="GPKG")
                log.info(f"✓ Saved tree influence zones to {influence_zones_path}")

        return fields_gdf, tof_influence_zones

    except Exception as e:
        log.error(f"Error calculating tree influence: {e}")
        # Add empty columns on error
        fields_gdf["tree_influence_mean"] = 0.0
        fields_gdf["tree_influence_max"] = 0.0
        fields_gdf["tree_coverage_pct"] = 0.0
        # Return empty influence zones
        empty_zones = gpd.GeoDataFrame(columns=["geometry"], crs=fields_gdf.crs)
        return fields_gdf, empty_zones


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-wfs", help="WFS services configuration file")
    parser.add_argument("--config-wms", help="WMS services configuration file")
    parser.add_argument("--aoi", help="Area of Interest file (GeoJSON/GPKG)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--list-wms", help="List WMS layers from URL")
    parser.add_argument(
        "--resolution",
        type=float,
        default=25.0,
        help="Raster resolution for Bodenertrag (default: 25m)",
    )

    args = parser.parse_args()

    # Handle --list-wms option
    if args.list_wms:
        layers = list_wms_layers(args.list_wms)
        print(f"\nAvailable layers from {args.list_wms}:")
        for layer in layers:
            print(f"  - {layer}")
        return

    # Check required arguments for main functionality
    if not all([args.config_wfs, args.config_wms, args.aoi, args.output]):
        parser.error(
            "--config-wfs, --config-wms, --aoi, and --output are required for main functionality"
        )

    # Load configurations
    log.info("Loading configurations...")
    wfs_config = load_yaml(args.config_wfs)
    wms_config = load_yaml(args.config_wms)

    # Read AOI
    log.info(f"Reading AOI from {args.aoi}")
    aoi = read_aoi(args.aoi)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download Feldblöcke
    log.info("Step 1: Downloading Feldblöcke...")
    fields = download_feldblocks_wfs(aoi, wfs_config, args.output)

    # Step 2: Download Bodenertrag and rasterize
    log.info("Step 2: Processing Bodenertrag data...")
    bodenertrag_raster = None
    try:
        bodenertrag_gdf = download_bodenertrag_wfs(aoi, wfs_config, args.output)
        if len(bodenertrag_gdf) > 0:
            # Rasterize Bodenertrag
            target_crs = "EPSG:25833"
            fields_bounds = fields.to_crs(target_crs).total_bounds

            raster_array, transform, profile = rasterize_bodenertrag(
                bodenertrag_gdf, fields_bounds, target_crs, args.resolution
            )

            if raster_array is not None:
                bodenertrag_raster = (
                    output_dir / "rasters" / "bodenertrag_classified.tif"
                )
                bodenertrag_raster.parent.mkdir(parents=True, exist_ok=True)

                with rasterio.open(bodenertrag_raster, "w", **profile) as dst:
                    dst.write(raster_array, 1)

                log.info(f"✓ Saved Bodenertrag raster to {bodenertrag_raster}")
    except Exception as e:
        log.warning(f"Could not process Bodenertrag data: {e}")

    # Step 3: Download WMS layers
    log.info("Step 3: Downloading WMS layers...")
    wms_layers = wms_config.get("wms", [])
    raster_files = []

    for layer in wms_layers:
        try:
            raster_file = download_wms_layer(layer, aoi, args.output)
            raster_files.append(raster_file)
        except Exception as e:
            log.error(f"Failed to download WMS layer {layer.get('id', 'unknown')}: {e}")

    # Step 3.5: Process local raster files
    log.info("Step 3.5: Processing local raster files...")
    local_rasters = wms_config.get("local_rasters", [])

    for raster_config in local_rasters:
        try:
            raster_file = process_local_raster(raster_config, aoi, args.output)
            raster_files.append(raster_file)
        except Exception as e:
            log.error(
                f"Failed to process local raster {raster_config.get('id', 'unknown')}: {e}"
            )

    if not raster_files and not bodenertrag_raster:
        log.warning(
            "No raster data was successfully downloaded. Proceeding with Feldblöcke only."
        )
        # Save Feldblöcke without additional features
        output_gpkg = output_dir / "feldblocks_only.gpkg"
        output_csv = output_dir / "feldblocks_only.csv"

        fields.to_file(output_gpkg, driver="GPKG")
        csv_data = fields.drop(columns=[fields.geometry.name])
        csv_data.to_csv(output_csv, index=False)

        log.info(f"✓ Saved Feldblöcke only:")
        log.info(f"  - GPKG: {output_gpkg}")
        log.info(f"  - CSV: {output_csv}")
        return

    # Step 4: Aggregate features
    log.info("Step 4: Aggregating features to fields...")
    final_gdf = aggregate_features_to_fields(
        fields, raster_files, str(bodenertrag_raster) if bodenertrag_raster else None
    )

    # Step 4.5: Interpolate missing environmental values
    log.info("Step 4.5: Interpolating missing environmental values...")
    environmental_columns = ["dry_mean", "nfk_mean"]
    final_gdf = interpolate_missing_values(final_gdf, environmental_columns)

    # Step 4.6: Calculate tree influence for fields
    log.info("Step 4.6: Calculating tree influence for fields...")
    final_gdf, tree_influence_zones = calculate_tree_influence_for_fields(
        final_gdf, output_dir=output_dir
    )

    # Step 4.7: Clip protected areas (remove conservation zones from fields)
    log.info("Step 4.7: Clipping fields to exclude protected areas...")
    try:
        final_gdf, protected_areas = exclude_protected_areas(
            final_gdf, wfs_config, output_dir=output_dir
        )
    except Exception as e:
        log.warning(f"Could not process protected areas clipping: {e}")
        log.info("Continuing with all field blocks...")
        # Create empty protected areas GeoDataFrame for consistency
        protected_areas = gpd.GeoDataFrame(columns=["geometry"], crs=final_gdf.crs)

    # Step 5: Save results
    log.info("Step 5: Saving results...")
    output_gpkg = output_dir / "feldblocks_with_features.gpkg"
    output_csv = output_dir / "feldblocks_with_features.csv"

    final_gdf.to_file(output_gpkg, driver="GPKG")

    # Save CSV without geometry
    csv_data = final_gdf.drop(columns=[final_gdf.geometry.name])
    csv_data.to_csv(output_csv, index=False)

    log.info(f"✓ Results saved:")
    log.info(f"  - GPKG: {output_gpkg}")
    log.info(f"  - CSV: {output_csv}")
    log.info(f"  - Rasters: {output_dir / 'rasters'}")
    if len(tree_influence_zones) > 0:
        log.info(
            f"  - Tree influence zones: {output_dir / 'tree_influence_zones.gpkg'}"
        )
    if len(protected_areas) > 0:
        log.info(f"  - Protected areas: {output_dir / 'protected_areas.gpkg'}")
    log.info(
        f"✓ Processed {len(final_gdf)} field blocks with {len(final_gdf.columns) - 1} attributes"
    )


if __name__ == "__main__":
    main()
