import cooler
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import coo_matrix, issparse
from pathlib import Path
import os
from typing import Dict, Any, List
import logging
import gc
import psutil
import matplotlib.pyplot as plt
import matspy 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scHiC")

class scHiC:
    def __init__(self):
        self.hic_data = []  # Store COO matrices for memory efficiency
        self.metadata = pd.DataFrame()
        self.bins = None
        self.chroms = None
        self.cell_names = []
        self._dtype = np.float32

    def _mem_usage(self):
        return psutil.Process().memory_info().rss / 1024**3  # in GB

    def load_cells(self, cool_files_dir: str, balance: bool = False) -> None:
        """
        Load .cool files from a directory into the scHiC object.

        Parameters
        ----------
        cool_files_dir : str
            Path to directory containing .cool files.
        balance : bool
            If True, apply balancing weights from .cool files.
        """
        import cooler

        cool_files_dir = Path(cool_files_dir)
        if not cool_files_dir.exists():
            raise FileNotFoundError(f"Directory {cool_files_dir} does not exist.")

        cool_files = list(cool_files_dir.glob("*.cool"))
        if not cool_files:
            logger.warning(f"No .cool files found in {cool_files_dir}.")
            return

        expected_shape = None

        for i, cool_file in enumerate(cool_files):
            try:
                cell_name = cool_file.stem
                logger.info(f"[{i+1}/{len(cool_files)}] Loading: {cool_file.name} (Cell: {cell_name})")

                clr = cooler.Cooler(str(cool_file))

                if expected_shape is None:
                    expected_shape = clr.shape
                    self.chroms = clr.chroms()[:]  # store chroms dataframe
                    self.bins = clr.bins()[:][['chrom', 'start', 'end']]  # store only needed columns

                elif clr.shape != expected_shape:
                    raise ValueError(f"Matrix shape mismatch in {cool_file.name}: {clr.shape} vs {expected_shape}")

                pixels = clr.pixels()[:]
                row = pixels['bin1_id'].astype(np.int32)
                col = pixels['bin2_id'].astype(np.int32)
                values = pixels['count'].astype(np.float32)

                if balance:
                    try:
                        weights = clr.bins()[:]['weight'].astype(np.float32).fillna(1.0)
                        values = values * weights[row].values * weights[col].values
                    except Exception as e:
                        logger.warning(f"Balancing failed for {cool_file.name}: {e}")

                sparse_matrix = coo_matrix((values, (row, col)), shape=expected_shape, dtype=self._dtype)
                self.hic_data.append(sparse_matrix)
                self.cell_names.append(cell_name)

                logger.debug(f"{cell_name}: Matrix loaded with shape {sparse_matrix.shape} and nnz={sparse_matrix.nnz}")

                del pixels, row, col, values
                gc.collect()

            except Exception as e:
                logger.error(f"Error loading {cool_file.name}: {e}")

        if not self.hic_data:
            raise ValueError("No valid .cool files were loaded.")
        logger.info(f"Successfully loaded {len(self.cell_names)} .cool files.")


    def load_metadata(self, metadata_path: str) -> None:
        """
        Load metadata from a CSV file and match it with the loaded cells.

        Parameters:
        -----------
        metadata_path : str
            Path to the metadata CSV file.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")

        try:
            metadata = pd.read_csv(metadata_path)
            logger.info(f"Metadata loaded from {metadata_path}")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise

        # Ensure metadata has a 'Cool_name' column
        if 'Cool_name' not in metadata.columns:
            raise ValueError("Metadata CSV must contain a 'Cool_name' column matching .cool file names.")

        metadata['Cool_name'] = metadata['Cool_name'].str.replace('.cool', '', regex=False)

        # Filter metadata to include only processed cells
        self.metadata = metadata[metadata['Cool_name'].isin(self.cell_names)]
        logger.info(f"Metadata filtered to {len(self.metadata)} cells.")

        
    def subset_from_self(self, selected_names: list) -> "scHiC":
        """
        Create a new scHiC object containing a subset of the current object
        based on provided cell names.

        Parameters:
        -----------
        selected_names : list
            List of cell names to include in the subset (without .cool extension).

        Returns:
        --------
        scHiC
            A new scHiC object with filtered metadata and hic_data.
        """
        selected_indices = [self.cell_names.index(name) for name in selected_names if name in self.cell_names]
        filtered_hic_data = [self.hic_data[i] for i in selected_indices]
        filtered_cell_names = [self.cell_names[i] for i in selected_indices]
        filtered_metadata = self.metadata.set_index('Cool_name').loc[selected_names].reset_index()

        subset = scHiC()
        subset.hic_data = filtered_hic_data
        subset.metadata = filtered_metadata
        subset.cell_names = filtered_cell_names
        subset.bins = self.bins
        subset.chroms = self.chroms

        logger.info(f"Created subset with {len(filtered_cell_names)} cells.")

        return subset

              
    def describe(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for the loaded Hi-C data using sparse matrices.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing detailed statistics for each cell.
        """
        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        stats = []

        for i, matrix in enumerate(self.hic_data):
            non_zero_values = matrix.data

            cell_stats = {
                "Cell Name": self.cell_names[i],
                "Matrix Shape": matrix.shape,
                "Total Contacts": matrix.sum(),
                "Non-zero Contacts": matrix.nnz,
                "Sparsity (%)": 100 * (1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))),
                "Mean Contact Value": np.mean(non_zero_values) if non_zero_values.size > 0 else 0,
                "Median Contact Value": np.median(non_zero_values) if non_zero_values.size > 0 else 0,
                "Min Contact Value": np.min(non_zero_values) if non_zero_values.size > 0 else 0,
                "Max Contact Value": np.max(non_zero_values) if non_zero_values.size > 0 else 0,
                "Std Contact Value": np.std(non_zero_values) if non_zero_values.size > 0 else 0,
            }
            stats.append(cell_stats)

        stats_df = pd.DataFrame(stats)
        return stats_df

    def describe_all(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics for all Hi-C maps using sparse matrices.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing aggregated statistics for all cells.
        """
        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        all_stats = self.describe()

        aggregated_stats = {
            "Total Cells": len(self.hic_data),
            "Average Matrix Shape": all_stats["Matrix Shape"].mode().iloc[0],  # Most common shape
            "Total Contacts (All Cells)": all_stats["Total Contacts"].sum(),
            "Average Non-zero Contacts": all_stats["Non-zero Contacts"].mean(),
            "Average Sparsity (%)": all_stats["Sparsity (%)"].mean(),
            "Average Mean Contact Value": all_stats["Mean Contact Value"].mean(),
            "Median Mean Contact Value": all_stats["Mean Contact Value"].median(),
            "Min Mean Contact Value": all_stats["Mean Contact Value"].min(),
            "Max Mean Contact Value": all_stats["Mean Contact Value"].max(),
            "Average Std Contact Value": all_stats["Std Contact Value"].mean(),
        }
        print("\nAggregated Statistics for All Cells:")
        print("=" * 40)
        for key, value in aggregated_stats.items():
            print(f"{key:<30}: {value}")
        print("=" * 40)

        return aggregated_stats

    def save_to_hdf(self, output_path: str, compression: str = "gzip", compression_level: int = 4) -> None:
        """
        Save HiC data and metadata to an HDF5 file.

        Parameters:
        -----------
        output_path : str
            Directory to save the HDF5 file and metadata.
        compression : str, optional
            Compression method for HDF5 (default: "gzip").
        compression_level : int, optional
            Compression level for HDF5 (default: 4).
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save HiC data as a compressed 3D matrix in HDF5 format
        hdf5_file = output_path / "hic_data.h5"
        with h5py.File(hdf5_file, 'w') as f:
            for i, matrix in enumerate(self.hic_data):
                f.create_dataset(
                    f'data_{i}',
                    data=matrix.toarray(),
                    compression=compression,
                    compression_opts=compression_level
                )
            if self.bins is not None:
                f.create_dataset('bins', data=self.bins.to_records(index=False))
            if self.chroms is not None:
                f.create_dataset('chroms', data=self.chroms.to_records(index=False))

        # Save metadata to CSV
        metadata_file = output_path / "metadata.csv"
        self.metadata.to_csv(metadata_file, index=False)
        logger.info(f"Data saved to {output_path}")

    def save_to_cool(self, output_dir: str) -> None:
        """
        Save Hi-C data to .cool files.

        Parameters:
        -----------
        output_dir : str
            Directory to save the .cool files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        if self.bins is None or self.chroms is None:
            raise ValueError("Bin or chromosome information is missing.")

        for i, (matrix, cell_name) in enumerate(zip(self.hic_data, self.cell_names)):
            try:
                coo = matrix.tocoo()
                pixels = pd.DataFrame({
                    'bin1_id': coo.row,
                    'bin2_id': coo.col,
                    'count': coo.data
                })

                # Create .cool file
                cool_file = output_dir / f"{cell_name}.cool"
                cooler.create_cooler(
                    str(cool_file),
                    bins=self.bins,
                    pixels=pixels,
                    chroms=self.chroms,
                    ordered=True,
                    dtypes={'count': np.float32}
                )
                logger.info(f"Saved {cool_file.name}")
            except Exception as e:
                logger.error(f"Error saving {cell_name}.cool: {e}")
                
         
