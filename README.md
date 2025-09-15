# Agroforest Potential Analysis Pipeline

This project provides a comprehensive analysis framework for agroforestry potential in agricultural areas by combining environmental data, soil characteristics, and existing tree coverage. The pipeline includes automated data aggregation from web services and local data processing with agroforestry potential modeling.

## 🌳 Project Overview

The comprehensive analysis pipeline combines:
- **Agricultural field boundaries** (Feldblöcke) downloaded from Brandenburg WFS services
- **Environmental risk factors** (erosion, drought stress) from WMS and local rasters
- **Soil characteristics** (field capacity, yield potential, soil classes)
- **Habitat biotope categories** (HBN_KAT with agroforestry suitability mapping)
- **Existing tree influence** (TOF analysis with mathematical modeling)
- **Agroforestry potential modeling** (Individual field assessment)
- **GIS visualization** (Comprehensive spatial outputs)

## 🚀 Main Components

### 1. Data Aggregation Pipeline (`agroforest_aggregator.py`)
**Enhanced agroforestry data collection with tree influence integration**

**Processing Steps:**
1. **Field Block Download**: Downloads Feldblöcke via WFS from Brandenburg
2. **Soil Yield Processing**: Downloads and rasterizes Bodenertrag data with classification
3. **Environmental Data**: Downloads WMS layers (water/wind erosion) and processes local rasters
4. **Data Aggregation**: Performs zonal statistics to aggregate all environmental data to field level
5. **Missing Value Interpolation**: K-nearest neighbors interpolation for data gaps
6. **🌳 Tree Influence Calculation**: Calculates tree coverage and influence for each field
7. **Protected Area Processing**: Geometric clipping of fields by conservation zones
8. **Comprehensive Output**: Enhanced field blocks with all environmental parameters

### 2. Agroforestry Potential Analysis (`agroforest_analysis.py`)
**Specification-based potential modeling for agricultural fields**

**Core Features:**
- **Specification-based approach**: Uses domain-expert defined benefit/cost features
- **HBN_KAT integration**: Habitat biotope categories mapped to agroforestry suitability
- **Categorical data protection**: Proper handling of ordinal/categorical features
- **Individual field analysis**: Field-by-field potential assessment
- **Robust preprocessing**: Winsorizing, imputation, and feature engineering
- **Quantile-based features**: 0-1 normalization with cost feature inversion
- **Tertile classification**: Low/medium/high potential categories

**Analysis Process:**
- Quantile-based feature transformation (0-1 normalization)
- Cost feature inversion (NFK, soil class)
- Individual field potential calculation using mean of quantile features
- Tertile-based classification for practical decision making
- Standardized feature preprocessing
- Cluster-level potential aggregation
- Cluster-based classification using AP_cluster values
- Spatial pattern identification

### 3. GIS Visualization (`create_gis_visualization.py`)
**Comprehensive spatial visualization of analysis results**

**Features:**
- **Spatial integration**: Combines analysis results with field geometries
- **Classification visualization**: 5-class potential rating system (Very Low to Very High)
- **Multiple output layers**: Complete dataset and simplified views
- **GeoPackage format**: Professional GIS-ready spatial data
- **Styling guidance**: Color scheme recommendations for mapping

### 4. Tree Influence Calculation (`calculate_tree_influence.py`)
**Quantifies existing tree coverage impact using scientific modeling**

**Mathematical Model:**
- **Influence Radius**: R(H) = α × H (where α = 10, H = tree height)
- **Effective Influence**: B = R(H) × (1 - Pc) (where Pc = porosity factor)

**Porosity Factors by Tree Class:**
- **Forest (Class 1)**: Pc = 0.20 (dense coverage)
- **Patch (Class 2)**: Pc = 0.35 (moderate density)  
- **Linear (Class 3)**: Pc = 0.45 (hedgerows, tree lines)
- **Tree (Class 4)**: Pc = 0.65 (individual trees)

##  Feature Engineering

### Feature Processing Pipeline

#### Input Features (9 core features):
1. **`f_dry`** (Benefit): Drought stress index - higher values indicate more drought stress (benefit for agroforestry)
2. **`f_eros_wasser`** (Benefit): Water erosion risk - higher values need more protection
3. **`f_eros_wind`** (Benefit): Wind erosion risk - higher values need more protection  
4. **`f_tof_gap`** (Benefit): Tree coverage gap - calculated as (1 - tree_coverage_pct) - higher gaps have more potential
5. **`f_size`** (Benefit): Field size in hectares - larger fields more suitable for agroforestry
6. **`f_shape`** (Benefit): Compactness index - more compact fields are better
7. **`f_hbn_kat`** (Benefit): Habitat biotope category - mapped as AL=1.0 (good), GL=0.5 (moderate), DK=0.0 (poor)
8. **`f_nfk`** (Cost): Soil field capacity - **inverted** as lower NFK values are more favorable for agroforestry
9. **`f_soil`** (Cost): Soil class (0-4) - **inverted** as lower classes indicate better soil productivity

#### Preprocessing Steps:
1. **Missing Value Imputation**: Median-based imputation
2. **Outlier Treatment**: Winsorizing to 1st-99th percentiles (excluding categorical features)
3. **Categorical Protection**: HBN_KAT and soil class excluded from winsorizing
## Data Requirements & Sources

### Automated Data (Downloaded by Pipeline)
The pipeline automatically downloads the following data via web services:

#### WFS Services (Vector Data)
| Service | Source | Description | Auto-Download |
|---------|--------|-------------|---------------|
| **Feldblöcke** | Brandenburg GeoBasis | Agricultural field boundaries | ✅ Yes |
| **Bodenertrag** | Brandenburg INSPIRE | Soil yield classification | ✅ Yes |
| **Protected Areas** | BfN Federal | Nature conservation zones | ✅ Yes |

#### WMS Services (Raster Data)  
| Layer | Source | Description | Auto-Download |
|-------|--------|-------------|---------------|
| **Water Erosion** | Brandenburg INSPIRE | Erosion risk classification | ✅ Yes |
| **Wind Erosion** | Brandenburg INSPIRE | Wind erosion susceptibility | ✅ Yes |

### 📁 Manual Data Requirements
**⚠️ IMPORTANT: These datasets must be downloaded manually before running the pipeline**

#### Required Local Data Files
| Dataset | Required Path | Source | Download Instructions |
|---------|---------------|--------|----------------------|
| **Drought Index 2024** | `data/Trockenheitsindex/trockenheits_index_2024.tif` | DWD/Brandenburg | Download from regional drought monitoring |
| **Soil Field Capacity** | `data/Feldkapazitaet/NFKWe1000_250.tif` | BGR/Brandenburg | Download from soil survey data portal |
| **TOF Data** | `data/TOF/BB_TOF.gpkg` | Brandenburg Forest Service | Request from forestry administration |
| **nDOM Height Data** | `data/nDOM/*.tif` | Brandenburg LGB | Download elevation model tiles |
| **Study Area Extent** | `data/extent_bb.geojson` | User-defined | Create study area boundary |

#### Detailed Download Instructions

**1. Drought Index (Trockenheitsindex)**
- **Source**: Deutsche Wetterdienst (DWD) / Brandenburg Climate Portal
- **URL**: https://opendata.dwd.de/climate_environment/CDC/grids_germany/annual/drought_index/
- **Format**: GeoTIFF raster file
- **Required**: 2024 drought stress index for Brandenburg
- **Place in**: `data/Trockenheitsindex/trockenheits_index_2024.tif`

**2. Soil Field Capacity (Feldkapazität)**  
- **Source**: Bundesanstalt für Geowissenschaften und Rohstoffe (BGR)
- **URL**: https://numis.niedersachsen.de/trefferanzeige?docuuid=8e3f001c-9c6e-4eeb-8d0d-988456a20486
- **Format**: GeoTIFF raster (250m resolution)
- **Required**: Soil water holding capacity data
- **Place in**: `data/Feldkapazitaet/NFKWe1000_250.tif`

**3. Trees Outside Forest (TOF)**
- **Source**: Research Project of Joint Lab
- **RePo**: https://github.com/Moerizzy/TOFMapper
- **Format**: GeoPackage with tree polygons and height information
- **Required**: All TOF objects with tree class and height attributes
- **Place in**: `data/TOF/BB_TOF.gpkg`

**4. Normalized Digital Elevation Model (nDOM)**
- **Source**: Landesvermessung und Geobasisinformation Brandenburg (LGB)
- **URL**: https://data.geobasis-bb.de
- **Format**: Multiple GeoTIFF tiles (*.tif)
- **Required**: Height data for tree influence calculations
- **Place in**: `data/nDOM/` (all TIF files)

**5. Study Area Boundary**
- **Source**: User-created or official administrative boundaries
- **Format**: GeoJSON polygon
- **Required**: Defines the analysis extent
- **Place in**: `data/extent_bb.geojson`

### 📊 Expected Input Data Structure
After running the data aggregation pipeline, the analysis requires a CSV file with these columns:

#### Required Columns for Analysis:
- **`gml_id`**: Unique field identifier
- **`FB_ID`**: Field block ID  
- **`area_ha`**: Field area in hectares
- **`compactness`**: Shape compactness index
- **`eros_wasser_mean`**: Water erosion risk (mean)
- **`eros_wind_mean`**: Wind erosion risk (mean)
- **`dry_mean`**: Drought stress index (mean)
- **`nfk_mean`**: Soil field capacity (mean)
- **`bodenertrag_class`**: Soil yield class (0-4, ordinal)
- **`tree_coverage_pct`**: Tree coverage percentage (0-100)
- **`HBN_KAT`**: Habitat biotope category (AL/GL/DK)

#### Example Input Format:
```csv
gml_id,FB_ID,area_ha,compactness,eros_wasser_mean,eros_wind_mean,dry_mean,nfk_mean,bodenertrag_class,tree_coverage_pct,HBN_KAT
FB_1484060,DEBBLI0265002971,5.2,0.65,151.3,101.7,26.0,120.4,3,25.5,GL
FB_1459110,DEBBLI0365030354,12.8,0.18,176.6,208.0,27.0,198.2,2,2.2,AL
```
| **Field Capacity** | `data/Feldkapazitaet/` | Soil water holding capacity |
| **TOF Data** | `data/TOF/BB_TOF.gpkg` | Trees Outside Forest inventory |
| **Height Data** | `data/nDOM/*.tif` | Normalized height model (100 files) |

### Tree Influence Calculation (`calculate_tree_influence.py`)
**Quantifies existing tree coverage impact using scientific modeling**

#### Mathematical Model:
- **Influence Radius**: R(H) = α × H (where α = 10, H = tree height)
- **Effective Influence**: B = R(H) × (1 - Pc) (where Pc = porosity factor)

#### Porosity Factors by Tree Class:
- **Forest (Class 1)**: Pc = 0.20 (dense coverage)
- **Patch (Class 2)**: Pc = 0.35 (moderate density)
- **Linear (Class 3)**: Pc = 0.45 (hedgerows, tree lines)
- **Tree (Class 4)**: Pc = 0.65 (individual trees)

## 🔧 Usage & Script Execution

### Prerequisites
1. **Install required Python packages** (see Environment Setup section)
2. **Download manual data files** (see Data Requirements section above)
3. **Create study area boundary** (`data/extent_bb.geojson`)
4. **Configure data paths** in `agroforest_config.yaml`

### Step 1: Data Aggregation Pipeline
```bash
# Complete data collection and aggregation
python agroforest_aggregator.py \
    --config-wfs agroforest_config.yaml \
    --config-wms agroforest_config.yaml \
    --aoi data/extent_bb.geojson \
    --output collected_data/

# Standalone tree influence calculation (optional)
python calculate_tree_influence.py \
    --output-dir tree_influence_results/
```

### Step 2: Agroforestry Potential Analysis
```bash
# Run agroforestry potential analysis
python agroforest_analysis.py \
    --input collected_data/feldblocks_with_features.csv \
    --output analysis_results/

# Alternative with custom parameters
python agroforest_analysis.py \
    --input data.csv \
    --output results/
```

#### Analysis Parameters:
- `--input`: Path to CSV file with field data (from aggregation pipeline)
- `--output-dir`: Output directory for results

### Step 3: GIS Visualization
```bash
# Create GIS visualization from analysis results
python create_gis_visualization.py \
    --analysis-results analysis_results/agroforest_analysis.csv \
    --output gis_visualization/

### Configuration and Utilities
```bash
# List available WMS layers
python agroforest_aggregator.py \
    --list-wms "https://inspire.brandenburg.de/services/boerosion_wms"

# Custom resolution for rasterization
python agroforest_aggregator.py \
    --resolution 10 \
    --config-wfs config.yaml \
    --aoi area.geojson \
    --output results/
```
python agroforest_aggregator.py \
    --list-wms "https://inspire.brandenburg.de/services/boerosion_wms"

# Custom resolution for rasterization
python agroforest_aggregator.py \
    --resolution 10 \
    --config-wfs config.yaml \
    --aoi area.geojson \
    --output results/
```

## 📈 Output Files & Analysis Results

### Agroforestry Analysis Outputs

#### Primary Result Files:
- **`agroforest_analysis.csv`**: Individual field analysis results
- **`top_candidates.csv`**: Top 50 fields with highest potential

#### Visualization Files:
- **`potential_distribution.png`**: Histogram of potential index values
- **`feature_correlation.png`**: Feature correlation matrix

### Key Output Columns

#### Analysis Results:
```csv
gml_id,FB_ID,potential_index,potential_class,original_f_dry,...,quantile_f_dry,...
FB_1484060,DEBBLI0265002971,0.385,low,26.0,...,0.098,...
```

#### Column Descriptions:
- **`potential_index`**: Calculated agroforestry potential (0-1 scale)
- **`potential_class`**: Classification (low/medium/high)
- **`original_*`**: Original feature values before preprocessing
- **`quantile_*`**: Quantile-transformed features

### GIS Visualization Outputs:
- **`agroforestry_potential.gpkg`**: Spatial layer with potential classification
- **`analysis_metadata.json`**: Processing metadata and statistics

### Data Aggregation Pipeline Outputs:
- **`feldblocks_with_features.gpkg`**: Enhanced field data with all parameters
- **`feldblocks_with_features.csv`**: Tabular export for analysis
- **`rasters/`**: Processed environmental raster layers

#### Environmental Parameters:
- `eros_wasser_mean/max/std`: Water erosion risk statistics
- `eros_wind_mean/max/std`: Wind erosion risk statistics  
- `dry_mean/max/std`: Drought stress index
- `nfk_mean/max/std`: Soil field capacity
- `bodenertrag_*`: Comprehensive yield class statistics

#### Tree Influence Metrics:
- **`tree_influence_mean`**: Average tree influence radius (meters)
- **`tree_influence_max`**: Maximum tree influence radius (meters)
- **`tree_coverage_pct`**: Percentage of field under tree influence (0-100%)

#### Geometric Properties:
- `area_ha`: Field area in hectares
- `circumference_m`: Field perimeter in meters
- `compactness`: Shape complexity index

### Analysis Summary Example:
```
Total fields analyzed: 302
Potential index: 0.501 ± 0.087
Range: 0.291 - 0.826

Potential categories (data-driven tertiles):
  Low potential: 101 (33.4%)
  Medium potential: 101 (33.4%) 
  High potential: 100 (33.1%)
```

## ⚙️ Configuration & Environment Setup

### Python Environment Setup
```bash
# Create conda environment with required packages
conda create -n agroforest python=3.11
conda activate agroforest

# Install core packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install geopandas rasterio rasterstats requests

# Alternative: Install from requirements
pip install -r requirements.txt
```

### Required Dependencies:
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
geopandas >= 0.13.0
rasterio >= 1.3.0
rasterstats >= 0.19.0
requests >= 2.31.0
pathlib (built-in)
argparse (built-in)
logging (built-in)
```
```

### Configuration Files

#### Unified Configuration (`agroforest_config.yaml`)
Single comprehensive file containing:
- **WFS services**: Vector data endpoints and parameters
- **WMS services**: Raster data sources and layer specifications
- **Local rasters**: File paths and processing parameters
- **Service URLs**: Official Brandenburg and federal data sources

#### Example Configuration Structure:
```yaml
wfs_services:
  feldblocks:
    url: "https://geobroker.geobasis-bb.de/gbss.php"
    layer: "inspire:AU.AdministrativeUnit"
    
wms_services:
  erosion_water:
    url: "https://inspire.brandenburg.de/services/boerosion_wms"
    layers: ["bb_eros_wasser"]
    
local_rasters:
  drought:
    path: "data/Trockenheitsindex/"
    pattern: "*.tif"
```