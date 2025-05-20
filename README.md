# scHi-C

**scHi-C** is a study project developed at the **Bioinformatics Institute** for exploring single-cell Hi-C data in Python. It provides a lightweight codebase and an example notebook to illustrate basic workflows in loading, processing and visualizing sparse single-cell Hi-C contact matrices.

---

## Repository structure

- **`schic/`**  
  The Python package directory. Contains the modules that implement data I/O, sparse-matrix operations and plotting utilities for single-cell Hi-C.

- **`Example.ipynb`**  
  A self-contained notebook showing how to:
  1. Load a `.cool`‐format Hi-C dataset and cell‐level metadata
  2. Explore main statistics of the dataset
  3. Compute and visualize average contact maps for cell subsets or genomic regions  

---

## Quick start

1. **Clone the repository**  
   ```bash
   git clone https://github.com/berlogo/scHi-C.git
   cd scHi-C

2. **Open the example notebook**

Launch Jupyter and open Example.ipynb to walk through basic data loading, metadata annotation, grouping and visualization steps:
   ```bash
   jupyter lab Example.ipynb


