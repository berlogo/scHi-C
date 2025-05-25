# scHi-C

**A Python toolkit for single-cell Hi-C data normalization and analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

scHi-C is a study project developed at the Bioinformatics Institute for exploring single-cell Hi-C data in Python. It provides a lightweight codebase and an example notebook to illustrate basic workflows in loading, processing and visualizing sparse single-cell Hi-C contact matrices.

## Table of Contents

- [About Single-Cell Hi-C Data](#about-single-cell-hi-c-data)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Normalization Methods](#normalization-methods)
- [Example Usage](#example-usage)
- [Data Sources](#data-sources)
- [References](#references)
- [Contact](#contact)

## About Single-Cell Hi-C Data

Single-cell Hi-C (scHi-C) captures chromosome conformation in individual cells, revealing cell-to-cell variability in 3D genome organization. Unlike bulk Hi-C, scHi-C data presents unique analytical challenges:

- **Ultra-sparse contact matrices** (>99.9% zeros)
- **Variable sequencing depth** across cells  
- **High technical noise** requiring specialized normalization
- **Limited statistical power** per individual cell

This toolkit implements normalization methods adapted for these challenges, based on established algorithms from the Hi-C analysis ecosystem.

## Features

**Normalization Methods:**
- **ICE** - Iterative Correction and Eigenvector decomposition (for high-coverage cells)
- **VC** - Vanilla Coverage normalization  
- **Coverage-based** - Simple coverage normalization for sparse data
- **SCALE** - Total count scaling (optimal for ultra-sparse data)

**Data Processing:**
- Automatic method selection based on data sparsity
- Parallel processing for large cell collections
- Memory-efficient sparse matrix operations
- Integration with cooler/cooltools ecosystem

**Quality Control:**
- Sparsity pattern diagnosis
- Coverage analysis and filtering
- Cell quality assessment
- Normalization validation

**Visualization:**
- Contact map plotting
- Quality control plots  
- Before/after normalization comparisons
- Group-based analysis

## Installation

### Requirements
- Python 3.8 or higher
- 16GB+ RAM recommended for large datasets

### Install from Source
```bash
git clone https://github.com/berlogo/scHi-C
cd scHi-C
pip install -e .
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```python
# Test basic imports
from schic_normalization import scHiCNormConfig
from schic_class import scHiC
from scHiC_vis import plot_hic_maps
print("scHiCTools imported successfully!")
```


## Quick start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/berlogo/scHi-C.git
   cd scHi-C
   ```
   
2. **Open the example notebook**

Launch Jupyter and open Example.ipynb to walk through basic data loading, metadata annotation, grouping and visualization steps:
   ```bash
   jupyter lab Example.ipynb
   ```

## Normalization Methods

| Method | Best for | Description |
|--------|----------|-------------|
| **SCALE** | Ultra-sparse (<10K contacts) | Simple total count scaling |
| **Coverage** | Sparse (10K-100K contacts) | Coverage-based normalization |
| **VC** | Medium-sparse (100K-1M contacts) | Vanilla Coverage (square root) |
| **ICE** | Dense (>1M contacts) | Iterative correction (computationally intensive) |

The toolkit automatically selects the optimal method based on your data's sparsity patterns.


## Data Sources

This toolkit has been tested on single-cell Hi-C datasets from:

### Public Datasets
- **GEO: GSE117876** - Human prefrontal cortex (Ramani et al., 2017)  

### Supported Data Formats
- **Cool files** (.cool) - Primary format, created by cooler

## References

### Key Publications

**Single-cell Hi-C Methods:**
- Tian, D. et al. (2023). Single-cell Hi-C reveals cell-to-cell variability in chromosome structure. *Nature Genetics*. [DOI: 10.1038/s41588-023-01389-1](https://doi.org/10.1038/s41588-023-01389-1)
- Ramani, V. et al. (2017). Massively multiplex single-cell Hi-C. *Nature Methods*, 14, 263–266. [DOI: 10.1038/nmeth.4155](https://doi.org/10.1038/nmeth.4155)
- Nagano, T. et al. (2017). Cell-cycle dynamics of chromosomal organization at single-cell resolution. *Nature*, 547, 61–67. [DOI: 10.1038/nature23001](https://doi.org/10.1038/nature23001)

**Hi-C Analysis Tools:**
- Abdennur, N. & Mirny, L.A. (2020). Cooler: scalable storage for Hi-C data and other genomically labeled arrays. *Bioinformatics*, 36, 311-316. [DOI: 10.1093/bioinformatics/btz540](https://doi.org/10.1093/bioinformatics/btz540)
- Open2C et al. (2023). Cooltools: enabling high-resolution Hi-C analysis in Python. *PLOS Computational Biology*. [DOI: 10.1371/journal.pcbi.1011214](https://doi.org/10.1371/journal.pcbi.1011214)

**Normalization Methods:**
- Imakaev, M. et al. (2012). Iterative correction of Hi-C data reveals hallmarks of chromosome organization. *Nature Methods*, 9, 999–1003. [DOI: 10.1038/nmeth.2148](https://doi.org/10.1038/nmeth.2148)
- Lieberman-Aiden, E. et al. (2009). Comprehensive mapping of long-range interactions reveals folding principles of the human genome. *Science*, 326(5950), 289-293. [DOI: 10.1126/science.1181369](https://doi.org/10.1016/j.plantsci.2025.112557)
- Cournac, A. et al. (2012). Normalization of a chromosomal contact map. *BMC Genomics*, 13, 436. [DOI: 10.1186/1471-2164-13-436](https://doi.org/10.1186/1471-2164-13-436)

## Troubleshooting

### Common Issues

**Import errors**
```python
# Make sure your Python files are in the correct location
# and adjust import paths as needed
import sys
sys.path.append('/path/to/your/code/')
from schic_normalization import scHiCNormConfig
```

**"No cells were successfully normalized"**
```python
# Try ultra-sparse configuration
config = scHiCNormConfig.for_ultra_sparse_data()
config.method = "SCALE"
# Then run normalize_schic with this config
```

**Memory errors with large datasets**
```python
# Process fewer cells at once
config = scHiCNormConfig()
config.chunk_size = 50  # Reduce from default
config.use_float32 = True  # Use less memory
```

**ICE convergence issues**
```python
# Use simpler methods for sparse data
config = scHiCNormConfig(method="coverage")  # Instead of ICE
```