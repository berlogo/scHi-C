import cooler
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import coo_matrix
from typing import Dict, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scHiC")

class scHiC:
    def __init__(self):
        """
        Initialize the scHiC object.
        """
        self.hic_data = []  # List of sparse matrices for HiC data
        self.metadata = pd.DataFrame()  # Metadata for each cell
        self.bins = None  # Bin information (chromosome, start, end)
        self.chroms = None  # Chromosome information
        self.cell_names = []  # List of cell names (from .cool file names)

    def load_cells(self, cool_files_dir: str, balance: bool = False) -> None:
        """
        Load all .cool files from the specified directory into a list of sparse matrices.
        Uses COO format for memory efficiency.

        Parameters:
        -----------
        cool_files_dir : str
            Directory containing .cool files.
        balance : bool, optional
            Whether to apply balancing to the Hi-C matrices (default: False).
        """
        cool_files_dir = Path(cool_files_dir)
        if not cool_files_dir.exists():
            raise FileNotFoundError(f"Directory {cool_files_dir} does not exist.")

        cool_files = [f for f in cool_files_dir.glob("*.cool") if f.is_file()]
        if not cool_files:
            logger.warning(f"No .cool files found in {cool_files_dir}.")
            return

        # Initialize variables to check matrix sizes
        expected_shape = None

        for cool_file in cool_files:
            try:
                cell_name = cool_file.stem  # Use file name (without extension) as cell name
                logger.info(f"Processing file: {cool_file.name} (Cell: {cell_name})")

                h5cool = cooler.Cooler(str(cool_file))

                # Check matrix size consistency
                if expected_shape is None:
                    expected_shape = h5cool.shape
                    # Read chromosome and bin information from the first file
                    self.chroms = h5cool.chroms()[:]
                    self.bins = h5cool.bins()[:]
                elif h5cool.shape != expected_shape:
                    raise ValueError(f"Matrix size mismatch in file {cool_file.name}. Expected {expected_shape}, got {h5cool.shape}.")

                # Read pixels
                pixels = h5cool.pixels()[:]  # Read all pixels
                row, col, values = pixels['bin1_id'], pixels['bin2_id'], pixels['count']

                # Debugging: Check lengths of row, col, and values
                logger.debug(f"Lengths - row: {len(row)}, col: {len(col)}, values: {len(values)}")
                if len(row) != len(col) or len(row) != len(values):
                    raise ValueError(f"Length mismatch in file {cool_file.name}. Row: {len(row)}, Col: {len(col)}, Values: {len(values)}")

                # Apply balancing if required
                if balance:
                    try:
                        weights = h5cool.bins()[:]['weight']
                        row_weights = weights[row]
                        col_weights = weights[col]
                        values = values * row_weights * col_weights
                    except Exception as e:
                        logger.warning(f"Balancing failed for file {cool_file.name}: {e}. Proceeding without balancing.")
                        balance = False  # Disable balancing for this file

                # Create sparse matrix in COO format
                sparse_matrix = coo_matrix((values, (row, col)), shape=h5cool.shape)
                self.hic_data.append(sparse_matrix)
                self.cell_names.append(cell_name)

            except Exception as e:
                logger.error(f"Error processing file {cool_file.name}: {e}")

        if not self.hic_data:
            raise ValueError("No valid .cool files processed.")

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

        # Filter metadata to include only processed cells
        self.metadata = metadata[metadata['Cool_name'].isin(self.cell_names)]
        logger.info(f"Metadata filtered to {len(self.metadata)} cells.")

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
            # Extract non-zero values from the sparse matrix
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

        # Convert to DataFrame for pretty printing
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

        # Collect statistics for all cells
        all_stats = self.describe()

        # Compute aggregated statistics
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
                # Convert sparse matrix to pixels table
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
