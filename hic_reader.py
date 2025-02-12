"""
hic_reader.py
=============

This module provides an optimized implementation for reading Hi‑C data from .cool files.
Hi‑C contact matrices are built from the raw pixels and bins tables and stored as sparse
matrices. Additional metadata (including patient_id, cell_type, sample_date) is extracted.
The module supports three output modes:

    - "default": Save individual sparse matrices and cell metadata in an HDF5 file.
    - "3d": Load all Hi‑C matrices into a single 3D xarray.DataArray (cells × n_bins × n_bins)
            with coordinates for both bin axes, and save as netCDF.
    - "anndata": Create an AnnData object that stores the full 3D Hi‑C data in obsm["hic"]
                 (without flattening) and saves bin coordinates in uns["hic_coords"].
                 
If the flag --cooltools is set, cooltools’ advanced ICE normalization is used.
The executor for parallel file reading is chosen automatically based on the average file size.
Additionally, files with total contacts below a specified threshold (--min-contacts) are skipped,
and all warnings can be logged to a file (--log-file).

References:
    - Lieberman-Aiden, E., et al. "Comprehensive Mapping of Long-Range Interactions Reveals Folding Principles of the Human Genome." Science, 2009.
    - Abdennur, F. & Mirny, L.A. "Cooler: scalable storage for Hi‑C data and other genomically-labeled arrays." Bioinformatics, 2020.
    - cooltools (https://github.com/mirnylab/cooltools)
    - Scanpy/AnnData (https://scanpy.readthedocs.io)
    - HDF5 (https://docs.h5py.org/en/stable/))
    - xarray (https://xarray.pydata.org)
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np
import pandas as pd
import sparse
import cooler
import xarray as xr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HiCReader:
    """
    HiCReader monitors a directory for new .cool files, builds Hi‑C contact matrices from
    the pixels and bins data, and extracts cell metadata (patient_id, cell_type, sample_date).
    Matrices are initially stored as sparse.COO objects and cell metadata are stored as an
    xarray.Dataset. The module supports export in three modes:
    
      - "default": Save individual sparse matrices and metadata to an HDF5 file.
      - "3d": Combine all Hi‑C matrices into a single 3D xarray.DataArray (cells × n_bins × n_bins)
              with proper bin coordinates, and save as netCDF.
      - "anndata": Create an AnnData object that stores the full 3D Hi‑C data in obsm["hic"]
                   (without flattening) and stores bin coordinates in uns["hic_coords"].
    """

    def __init__(self, use_cooltools: bool = False, min_contacts: float = 0, log_file: Optional[str] = None) -> None:
        """
        Initialize an empty HiCReader instance.
        
        Args:
            use_cooltools (bool): If True, use cooltools for ICE normalization. Default is False.
            min_contacts (float): Minimum total contacts required to process a file.
            log_file (Optional[str]): Path to a log file where warnings and errors will be saved.
        """
        self.sparse_data: List[sparse.COO] = []
        self.metadata: Optional[xr.Dataset] = None  # xarray.Dataset for cell metadata
        self.hic_3d: Optional[xr.DataArray] = None   # For 3D mode data
        self.use_cooltools = use_cooltools
        self.min_contacts = min_contacts
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def read_cool_file_safe(self, file_path: str, use_cooltools: bool = False) -> Tuple[Optional[sparse.COO], Optional[np.ndarray], Optional[Dict[str, any]]]:
        """
        Reads a .cool file by accessing its pixels and bins tables and builds a sparse Hi‑C contact matrix.
        If use_cooltools is True, attempts to use cooltools’ ICE normalization; falls back to cooler otherwise.
        
        The pixels table must contain:
            - 'bin1_id' and 'bin2_id' for indices.
            - Either 'balanced' (preferred) or 'count' for contact values.
        
        Additionally, extra metadata is extracted from the file's info:
            - patient_id, cell_type, sample_date.
        Files with total contacts below self.min_contacts are skipped (with a warning).
        
        Args:
            file_path (str): Path to the .cool file.
            use_cooltools (bool): Whether to use cooltools for normalization.
        
        Returns:
            tuple: (matrix, bin_positions, metadata) where:
                - matrix (sparse.COO): Sparse Hi‑C contact matrix.
                - bin_positions (np.ndarray): Array of bin start positions (int64).
                - metadata (dict): Dictionary with keys: file, total_contacts, contact_density,
                  patient_id, cell_type, sample_date.
            Returns (None, None, None) if an error occurs or if total_contacts < self.min_contacts.
        """
        try:
            c = cooler.Cooler(file_path)
            # Attempt to use cooltools if requested.
            if use_cooltools:
                try:
                    from cooltools.lib import ice
                    balanced_clr = ice.balance_cooler(file_path)
                    raw_matrix = balanced_clr.matrix(sparse=True)[:]
                    logger.info(f"Used cooltools for balancing file: {file_path}")
                except Exception:
                    logger.exception(f"Error using cooltools for balancing file '{file_path}'. Falling back to cooler.")
                    raw_matrix = c.matrix(balance=True, sparse=True)[:]
            else:
                raw_matrix = c.matrix(balance=True, sparse=True)[:]
            
            bins = c.bins()[:]
            n_bins = len(bins)
            shape = (n_bins, n_bins)
            
            pixels = c.pixels()[:]
            if "balanced" in pixels.columns:
                data = pixels["balanced"].values
                if not np.issubdtype(data.dtype, np.floating):
                    data = data.astype(np.float32)
            else:
                data = pixels["count"].values
                if data.dtype != np.int32:
                    max_val = data.max()
                    if max_val <= np.iinfo(np.int32).max:
                        data = data.astype(np.int32, casting="safe")
                    else:
                        logger.warning(f"Data in {file_path} exceeds int32 range; using unsafe cast.")
                        data = data.astype(np.int32, casting="unsafe")
            
            row = pixels["bin1_id"].values
            col = pixels["bin2_id"].values
            coords = np.vstack([row, col])
            
            matrix = sparse.COO(coords, data, shape=shape)
            total_contacts = float(matrix.sum())
            contact_density = float((matrix > 0).sum()) / (n_bins * n_bins)
            
            if total_contacts < self.min_contacts:
                logger.warning(f"File {file_path} skipped due to low total contacts ({total_contacts} < {self.min_contacts}).")
                return None, None, None
            
            cool_metadata = c.info
            if cool_metadata:
                patient_id = cool_metadata.get("patient_id", "NA")
                cell_type = cool_metadata.get("cell_type", "NA")
                sample_date = cool_metadata.get("sample_date", "NA")
            else:
                logger.warning(f"No metadata found in file {file_path}.")
                patient_id = cell_type = sample_date = "NA"
            
            metadata = {
                "file": file_path,
                "total_contacts": total_contacts,
                "contact_density": contact_density,
                "patient_id": patient_id,
                "cell_type": cell_type,
                "sample_date": sample_date
            }
            
            bin_positions = bins["start"].values.astype(np.int64)
            logger.info(f"Successfully read file: {file_path}")
            return matrix, bin_positions, metadata

        except Exception:
            logger.exception(f"Error reading file '{file_path}'")
            return None, None, None

    def _load_dense_and_meta(self, file_path: str) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Helper function that loads a .cool file and returns its dense Hi‑C matrix and metadata.
        
        Args:
            file_path (str): Path to the .cool file.
        
        Returns:
            tuple: (dense_matrix, metadata)
        """
        matrix, bin_positions, meta = self.read_cool_file_safe(file_path, self.use_cooltools)
        if matrix is None:
            raise ValueError(f"Failed to load file {file_path}")
        return matrix.todense(), meta

    async def continuous_folder_scan(self, folder_path: str, scan_interval: int = 10, output_file: Optional[str] = None) -> None:
        """
        Continuously scans the specified folder for new .cool files and processes them.
        The scan loop exits if no new files are found and at least one file has been processed.
        After exiting, if output_file is provided, the data is automatically saved to HDF5.
        
        Args:
            folder_path (str): Directory containing .cool files.
            scan_interval (int): Seconds between scans.
            output_file (Optional[str]): Output HDF5 file path.
        """
        processed_files = set()
        logger.info(f"Starting folder scan in: {folder_path}")

        while True:
            try:
                file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                              if f.lower().endswith(".cool")]
                new_files = [fp for fp in file_paths if fp not in processed_files]

                if new_files:
                    logger.info(f"Found {len(new_files)} new file(s) to process.")
                    await self._read_cool_files_async(new_files)
                    processed_files.update(new_files)
                else:
                    logger.debug("No new files found.")
                    if processed_files:
                        logger.info("No new files found and processing is complete. Saving data to HDF5.")
                        if output_file:
                            self.save_to_hdf5(output_file)
                        break

                await asyncio.sleep(scan_interval)
            except Exception as e:
                logger.error(f"Error during folder scanning: {e}")
                await asyncio.sleep(scan_interval)

    async def _read_cool_files_async(self, file_paths: List[str]) -> None:
        """
        Asynchronously reads multiple .cool files using an executor.
        The executor is automatically chosen based on average file size:
            - Files > 1 MB use ProcessPoolExecutor.
            - Otherwise, ThreadPoolExecutor is used.
        
        Args:
            file_paths (List[str]): List of file paths to process.
        """
        loop = asyncio.get_running_loop()
        tasks = []
        results: List[Tuple[sparse.COO, np.ndarray, Dict[str, any]]] = []

        sizes = [os.path.getsize(fp) for fp in file_paths]
        avg_size = np.mean(sizes)
        threshold = 1e6  # 1 MB threshold
        if avg_size > threshold:
            Executor = concurrent.futures.ProcessPoolExecutor
            logger.info("Using ProcessPoolExecutor based on average file size.")
        else:
            Executor = concurrent.futures.ThreadPoolExecutor
            logger.info("Using ThreadPoolExecutor based on average file size.")

        max_workers = min(4, os.cpu_count() or 2)
        with Executor(max_workers=max_workers) as pool:
            for fp in file_paths:
                task = loop.run_in_executor(pool, self.read_cool_file_safe, fp, self.use_cooltools)
                tasks.append(task)

            for completed_task in asyncio.as_completed(tasks):
                matrix_coo, bin_positions, meta = await completed_task
                if matrix_coo is not None:
                    results.append((matrix_coo, bin_positions, meta))
                else:
                    logger.warning("A file was skipped due to an error during reading.")

        self._store_results(results)

    def _store_results(self, results: List[Tuple[sparse.COO, np.ndarray, Dict[str, any]]]) -> None:
        """
        Stores newly read Hi‑C matrices and cell metadata.
        New metadata is converted to an xarray.Dataset along the "cell" dimension.
        
        Args:
            results (List[tuple]): List of (matrix, bin_positions, metadata) for each file.
        """
        if not results:
            logger.info("No new data to store.")
            return

        for matrix, _, meta in results:
            self.sparse_data.append(matrix)
        new_meta_df = pd.DataFrame([meta for _, _, meta in results])
        new_meta_xr = xr.Dataset.from_dataframe(new_meta_df).rename({"index": "cell"})
        if self.metadata is None:
            self.metadata = new_meta_xr
        else:
            self.metadata = xr.concat([self.metadata, new_meta_xr], dim="cell")

        logger.info(f"Stored {len(results)} new Hi‑C datasets. Total datasets: {len(self.sparse_data)}.")

    def save_to_hdf5(self, output_file: str) -> None:
        """
        Saves the Hi‑C data (sparse matrices) and cell metadata to an HDF5 file.
        In default mode, individual sparse matrices are saved along with cell metadata.
        Cell metadata (stored as an xarray.Dataset) is converted to a structured NumPy array.
        
        Args:
            output_file (str): Path to the output HDF5 (or netCDF) file.
        """
        try:
            with h5py.File(output_file, "w") as hdf:
                group = hdf.create_group("sparse_matrices")
                for i, matrix in enumerate(self.sparse_data):
                    mat_group = group.create_group(f"matrix_{i}")
                    mat_group.create_dataset("data", data=matrix.data, compression="gzip")
                    mat_group.create_dataset("coords", data=matrix.coords, compression="gzip")
                    mat_group.attrs["shape"] = matrix.shape
                # Save cell metadata from the xarray.Dataset.
                meta_df = self.metadata.to_dataframe()
                for col in meta_df.select_dtypes(include=[object]).columns:
                    max_len = meta_df[col].str.len().max()
                    meta_df[col] = meta_df[col].astype(f"S{max_len}")
                meta_array = meta_df.to_records(index=False)
                # Set chunks=(1,) since meta_array is 1D.
                hdf.create_dataset("metadata", data=meta_array, compression="lzf", chunks=(1,))
            logger.info(f"Data successfully saved to '{output_file}'.")
        except Exception:
            logger.exception(f"Failed to save data to HDF5 file '{output_file}'")

    def load_from_hdf5(self, input_file: str) -> None:
        """
        Loads Hi‑C data and cell metadata from an HDF5 file.
        
        Args:
            input_file (str): Path to the input HDF5 file.
        """
        try:
            with h5py.File(input_file, "r") as hdf:
                group = hdf["sparse_matrices"]
                self.sparse_data = []
                for key in group.keys():
                    mat_group = group[key]
                    data = mat_group["data"][:]
                    coords = mat_group["coords"][:]
                    shape = tuple(mat_group.attrs["shape"])
                    matrix = sparse.COO(coords, data, shape=shape)
                    self.sparse_data.append(matrix)
                meta_array = hdf["metadata"][:]
                meta_df = pd.DataFrame(meta_array)
                self.metadata = xr.Dataset.from_dataframe(meta_df).rename({"index": "cell"})
            logger.info(f"Data successfully loaded from '{input_file}'.")
        except Exception:
            logger.exception(f"Failed to load data from HDF5 file '{input_file}'")

    def read_cool_to_3d(self, folder_path: str, output_file: str) -> None:
        """
        Loads all .cool files in the specified folder and constructs a 3D xarray.DataArray
        (cells × n_bins × n_bins), where each slice along the "cell" dimension corresponds
        to one cell's Hi‑C matrix. The bin coordinates (bin_x and bin_y) are obtained from
        the first file. Cell identifiers are derived from the file names.
        The combined xarray.Dataset (with variable "hic_data" for the 3D array and "cell_meta" for cell metadata)
        is saved to disk in netCDF format (HDF5‑based) using LZF compression and one cell per chunk.
        
        Args:
            folder_path (str): Directory containing .cool files.
            output_file (str): Path to the output netCDF file.
        """
        import dask.array as da
        from dask import delayed, compute

        file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                            if f.lower().endswith(".cool")])
        if not file_list:
            logger.info("No .cool files found in the folder.")
            return

        # Load first file to determine shape and dtype.
        first_dense, _ = self._load_dense_and_meta(file_list[0])
        shape_mat = first_dense.shape  # (n_bins, n_bins)
        dtype = first_dense.dtype
        # Get bin positions from the first file.
        first_bins = cooler.Cooler(file_list[0]).bins()[:]["start"].values

        delayed_results = [delayed(self._load_dense_and_meta)(fp) for fp in file_list]
        dask_arrays = [da.from_delayed(result[0], shape=shape_mat, dtype=dtype) for result in delayed_results]
        stacked = da.stack(dask_arrays, axis=0)  # shape = (num_cells, n_bins, n_bins)

        # Compute metadata for each cell.
        meta_list = [result[1] for result in delayed_results]
        meta_list = compute(*meta_list)
        meta_df = pd.DataFrame(meta_list)
        # Derive cell IDs from file names.
        cell_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in file_list]

        logger.info(f"Constructed 3D array with shape {stacked.shape} from {len(file_list)} cells.")
        chunk_size = (1, stacked.shape[1], stacked.shape[2])
        data_np = stacked.compute()
        data_xr = xr.DataArray(
            data_np,
            dims=["cell", "bin_x", "bin_y"],
            coords={"cell": cell_ids, "bin_x": first_bins, "bin_y": first_bins},
            name="hic_data"
        )
        cell_meta = xr.Dataset.from_dataframe(meta_df).rename({"index": "cell"})
        ds = xr.Dataset({"hic_data": data_xr, "cell_meta": cell_meta})
        ds = ds.assign_coords({"cell": ("cell", cell_ids),
                                 "bin_x": ("bin_x", first_bins),
                                 "bin_y": ("bin_y", first_bins)})
        encoding = {"hic_data": {"chunksizes": (1, data_np.shape[1], data_np.shape[2]), "complevel": 1, "zlib": True}}
        ds.to_netcdf(output_file, mode="w", encoding=encoding)
        logger.info(f"3D Hi‑C data and cell metadata successfully saved to '{output_file}'.")
        self.hic_3d = ds["hic_data"]
        self.metadata = ds["cell_meta"]

    def load_from_netcdf(self, input_file: str) -> None:
        """
        Loads a netCDF file (HDF5‑based) containing 3D Hi‑C data and cell metadata into the HiCReader.
        
        Args:
            input_file (str): Path to the input netCDF file.
        """
        try:
            ds = xr.open_dataset(input_file)
            self.hic_3d = ds["hic_data"]
            self.metadata = ds["cell_meta"]
            logger.info(f"NetCDF file '{input_file}' successfully loaded into HiCReader.")
        except Exception:
            logger.exception(f"Failed to load netCDF file '{input_file}'")

    def save_to_h5ad(self, output_file: str) -> None:
        """
        Converts the loaded 3D Hi‑C data (stored as an xarray.Dataset in netCDF format)
        into an AnnData object for compatibility with Scanpy. The Hi‑C data are stored as 3D matrices
        in obsm["hic"] (cells × n_bins × n_bins) without flattening.
        Bin coordinates (bin_x and bin_y) are stored in adata.uns["hic_coords"].
        
        Args:
            output_file (str): Path to the input netCDF file containing 3D Hi‑C data.
                            The output AnnData object is saved as .h5ad.
        """
        import anndata as ad

        ds = xr.open_dataset(output_file)
        data_3d = ds["hic_data"].values  # shape = (num_cells, n_bins, n_bins)
        cell_meta = ds["cell_meta"].to_dataframe().reset_index(drop=True)
        bin_x = ds["bin_x"].values
        bin_y = ds["bin_y"].values
        num_cells, n_bins, _ = data_3d.shape

        adata = ad.AnnData(X=np.empty((num_cells, 0)), obs=cell_meta)
        adata.obsm["hic"] = data_3d
        adata.uns["hic_coords"] = {"bin_x": bin_x, "bin_y": bin_y}
        out_file = output_file.replace(".nc", ".h5ad")
        adata.write_h5ad(out_file, compression="gzip")
        logger.info(f"AnnData successfully saved to '{out_file}'.")


if __name__ == "__main__":
    import argparse
    import multiprocessing

    # Set multiprocessing start method to 'spawn' for macOS/M1 compatibility.
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Continuously monitor a directory for .cool files and process Hi‑C data."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the directory containing .cool files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hic_data.nc",
        help="Path to the output netCDF file for saving processed data (or HDF5 in default mode)."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Time interval (in seconds) between folder scans."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "3d", "anndata"],
        default="default",
        help="Processing mode: 'default' uses sparse storage, '3d' builds a 3D xarray, 'anndata' creates an AnnData object."
    )
    parser.add_argument(
        "--cooltools",
        action="store_true",
        help="If set, use cooltools for advanced ICE normalization."
    )
    parser.add_argument(
        "--min-contacts",
        type=float,
        default=0,
        help="Minimum total contacts required to process a file. Files below this threshold are skipped."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a log file where warnings and errors will be saved."
    )
    args = parser.parse_args()

    hic_reader = HiCReader(use_cooltools=args.cooltools, min_contacts=args.min_contacts, log_file=args.log_file)

    try:
        if args.mode == "default":
            asyncio.run(hic_reader.continuous_folder_scan(args.folder, scan_interval=args.interval, output_file=args.output))
        elif args.mode == "3d":
            hic_reader.read_cool_to_3d(args.folder, args.output)
        elif args.mode == "anndata":
            hic_reader.read_cool_to_3d(args.folder, args.output)
            hic_reader.save_to_h5ad(args.output)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
        if hic_reader.sparse_data:
            hic_reader.save_to_hdf5(args.output)
        exit(0)
