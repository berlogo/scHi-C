#!/usr/bin/env python3
"""
schic_class.py - Single-cell Hi-C data container with integrated normalization.

This class provides functionality to:
- Load and manage scHiC data from .cool files
- Apply various normalization methods
- Visualize and analyze normalized/unnormalized data
- Save and export processed data

Follows PEP8, Flake8, and SciPy conventions.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import cooler
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import shutil
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.sparse import coo_matrix, csr_matrix, issparse
import gc
import tempfile

warnings.filterwarnings('ignore')

# Import normalization module
try:
    from schic_normalization import (
        normalize_schic,
        scHiCNormConfig,
        diagnose_sparsity,
        NormalizationResult
    )
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False
    warnings.warn(
        "schic_normalization module not found. "
        "Normalization features will be disabled."
    )

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scHiC")


class scHiC:
    """
    Single-cell Hi-C data container with integrated normalization support.
    
    Attributes
    ----------
    hic_data : list of coo_matrix
        Sparse matrices containing Hi-C contact data
    metadata : pd.DataFrame
        Cell metadata
    bins : pd.DataFrame
        Genomic bin information
    chroms : pd.DataFrame
        Chromosome information
    cell_names : list of str
        Names of loaded cells
    normalization_weights : dict
        Normalization weights for each cell
    normalization_stats : dict
        Statistics from normalization
    normalization_method : str
        Method used for normalization
    is_normalized : bool
        Whether data has been normalized
    """
    
    def __init__(self):
        """Initialize empty scHiC object."""
        self.hic_data = []  # Store COO matrices for memory efficiency
        self.metadata = pd.DataFrame()
        self.bins = None
        self.chroms = None
        self.cell_names = []
        self._dtype = np.float32
        
        # Normalization-related attributes
        self.normalization_weights = {}
        self.normalization_stats = None
        self.normalization_method = None
        self.is_normalized = False

    def _mem_usage(self) -> float:
        """
        Get current memory usage in GB.
        
        Returns
        -------
        float
            Memory usage in gigabytes
        """
        return psutil.Process().memory_info().rss / 1024**3

    def load_cells(self, cool_files_dir: str, balance: bool = False) -> None:
        """
        Load .cool files from a directory into the scHiC object.

        Parameters
        ----------
        cool_files_dir : str
            Path to directory containing .cool files.
        balance : bool, default=False
            If True, apply balancing weights from .cool files.
        """
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
                logger.info(
                    f"[{i+1}/{len(cool_files)}] Loading: {cool_file.name} "
                    f"(Cell: {cell_name})"
                )

                clr = cooler.Cooler(str(cool_file))

                if expected_shape is None:
                    expected_shape = clr.shape
                    self.chroms = clr.chroms()[:]
                    self.bins = clr.bins()[:][['chrom', 'start', 'end']]

                elif clr.shape != expected_shape:
                    raise ValueError(
                        f"Matrix shape mismatch in {cool_file.name}: "
                        f"{clr.shape} vs {expected_shape}"
                    )

                pixels = clr.pixels()[:]
                row = pixels['bin1_id'].astype(np.int32)
                col = pixels['bin2_id'].astype(np.int32)
                values = pixels['count'].astype(np.float32)

                if balance:
                    try:
                        weights = clr.bins()[:]['weight'].astype(np.float32)
                        weights = weights.fillna(1.0)
                        values = values * weights[row].values * weights[col].values
                        self.normalization_weights[cell_name] = weights.values
                        self.is_normalized = True
                    except Exception as e:
                        logger.warning(f"Balancing failed for {cool_file.name}: {e}")

                sparse_matrix = coo_matrix(
                    (values, (row, col)),
                    shape=expected_shape,
                    dtype=self._dtype
                )
                self.hic_data.append(sparse_matrix)
                self.cell_names.append(cell_name)

                logger.debug(
                    f"{cell_name}: Matrix loaded with shape {sparse_matrix.shape} "
                    f"and nnz={sparse_matrix.nnz}"
                )

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

        Parameters
        ----------
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

        if 'Cool_name' not in metadata.columns:
            raise ValueError(
                "Metadata CSV must contain a 'Cool_name' column "
                "matching .cool file names."
            )

        metadata['Cool_name'] = metadata['Cool_name'].str.replace(
            '.cool', '', regex=False
        )
        self.metadata = metadata[metadata['Cool_name'].isin(self.cell_names)]
        logger.info(f"Metadata filtered to {len(self.metadata)} cells.")

    def normalize(self,
                  method: str = "auto",
                  output_dir: Optional[Union[str, Path]] = None,
                  config: Optional['scHiCNormConfig'] = None,
                  n_workers: Optional[int] = None,
                  diagnose_first: bool = True,
                  show_progress: bool = True,
                  apply_to_data: bool = True) -> 'NormalizationResult':
        """
        Normalize the Hi-C data using various methods.
        
        Parameters
        ----------
        method : str, default="auto"
            Normalization method: "auto", "coverage", "ICE", "KR", "VC", "SCALE"
        output_dir : str or Path, optional
            Directory to save normalized .cool files. If None, uses temp directory
        config : scHiCNormConfig, optional
            Configuration object. If None, will auto-configure
        n_workers : int, optional
            Number of parallel workers
        diagnose_first : bool, default=True
            Whether to run sparsity diagnosis first
        show_progress : bool, default=True
            Show progress bars
        apply_to_data : bool, default=True
            If True, apply normalization weights to loaded data
            
        Returns
        -------
        NormalizationResult
            Result object containing paths, weights, and statistics
        """
        if not NORMALIZATION_AVAILABLE:
            raise ImportError(
                "Normalization module not available. "
                "Please ensure schic_normalization.py is in your path."
            )
        
        if not self.hic_data:
            raise ValueError("No Hi-C data loaded. Please load data first.")
        
        # Save current data to temporary cool files for normalization
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Save to temporary cool files
            logger.info("Preparing data for normalization...")
            self.save_to_cool(temp_dir)
            cool_paths = list(temp_dir.glob("*.cool"))
            
            # Set output directory
            if output_dir is None:
                output_dir = temp_dir / "normalized"
            else:
                output_dir = Path(output_dir)
            
            # Run normalization
            logger.info(f"Running {method} normalization...")
            result = normalize_schic(
                cool_paths=cool_paths,
                output_dir=output_dir,
                method=method,
                config=config,
                n_workers=n_workers,
                diagnose_first=diagnose_first,
                show_progress=show_progress
            )
            
            # Store normalization info
            self.normalization_method = (
                method if method != "auto"
                else result.stats.get('method', method)
            )
            self.normalization_stats = result.stats
            
            # Check if any cells were normalized
            if len(result.cool_paths) == 0:
                logger.error("No cells were successfully normalized!")
                logger.error(
                    "Try using scHiCNormConfig.for_ultra_sparse_data() "
                    "or method='SCALE'"
                )
                return result
            
            # Apply weights to current data if requested
            if apply_to_data and result.weights:
                logger.info("Applying normalization weights to loaded data...")
                self._apply_normalization_weights(result.weights)
                self.is_normalized = True
            elif apply_to_data and len(result.cool_paths) > 0:
                # Load weights from normalized files if not in result
                logger.info("Loading weights from normalized files...")
                self._load_weights_from_files(result.cool_paths)
                if self.normalization_weights:
                    self.is_normalized = True
            
            return result
            
        finally:
            # Clean up temp files if using temp directory
            if output_dir is None or str(output_dir).startswith(str(temp_dir)):

                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _apply_normalization_weights(
        self,
        weights: Dict[str, np.ndarray]
    ) -> None:
        """Apply normalization weights to loaded matrices."""
        for i, cell_name in enumerate(self.cell_names):
            if cell_name in weights:
                weight_vec = weights[cell_name]
                matrix = self.hic_data[i]
                
                # Apply weights: w[i] * w[j] * data
                # Handle NaN weights properly
                row_weights = weight_vec[matrix.row]
                col_weights = weight_vec[matrix.col]
                
                # Create mask for valid weights
                valid_mask = ~(np.isnan(row_weights) | np.isnan(col_weights))
                
                if valid_mask.sum() == 0:
                    logger.warning(
                        f"Cell {cell_name}: No valid weights after normalization"
                    )
                    continue
                
                # Apply weights only to valid entries
                new_data = matrix.data.copy()
                new_data[valid_mask] = (
                    matrix.data[valid_mask] *
                    row_weights[valid_mask] *
                    col_weights[valid_mask]
                )
                new_data[~valid_mask] = 0  # Set invalid entries to 0
                
                # Create new matrix keeping only valid entries
                valid_indices = np.where(valid_mask)[0]
                new_row = matrix.row[valid_indices]
                new_col = matrix.col[valid_indices]
                new_values = new_data[valid_indices]
                
                # Replace matrix
                self.hic_data[i] = coo_matrix(
                    (new_values, (new_row, new_col)),
                    shape=matrix.shape,
                    dtype=self._dtype
                )
                
                # Store weights
                self.normalization_weights[cell_name] = weight_vec
    
    def _load_weights_from_files(self, cool_paths: List[str]) -> None:
        """Load normalization weights from cool files."""
        for cool_path in cool_paths:
            try:
                clr = cooler.Cooler(str(cool_path))
                cell_name = Path(cool_path).stem.replace("_normalized", "")
                
                if 'weight' in clr.bins().columns:
                    weights = clr.bins()[:]['weight'].values
                    self.normalization_weights[cell_name] = weights
            except Exception as e:
                logger.warning(f"Could not load weights from {cool_path}: {e}")
    
    def diagnose(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Diagnose sparsity patterns in the loaded data.
        
        Parameters
        ----------
        sample_size : int, default=10
            Number of cells to sample for diagnosis
            
        Returns
        -------
        dict
            Diagnosis results with statistics and recommendations
        """
        if not NORMALIZATION_AVAILABLE:
            raise ImportError("Normalization module not available.")
        
        # Save sample to temp files
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Save subset
            sample_indices = np.random.choice(
                len(self.cell_names),
                min(sample_size, len(self.cell_names)),
                replace=False
            )
            
            for idx in sample_indices:
                cell_name = self.cell_names[idx]
                matrix = self.hic_data[idx]
                
                # Convert to pixels
                pixels = pd.DataFrame({
                    'bin1_id': matrix.row,
                    'bin2_id': matrix.col,
                    'count': matrix.data
                })
                
                # Save cool file
                cool_file = temp_dir / f"{cell_name}.cool"
                cooler.create_cooler(
                    str(cool_file),
                    bins=self.bins,
                    pixels=pixels,
                    chroms=self.chroms,
                    ordered=True
                )
            
            # Run diagnosis
            cool_paths = list(temp_dir.glob("*.cool"))
            diagnosis = diagnose_sparsity(cool_paths, sample_size=len(cool_paths))
            
            return diagnosis
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def plot_contact_maps(self,
                          cell_indices: Optional[List[int]] = None,
                          chromosome: Optional[str] = None,
                          vmin: float = 0,
                          vmax: Optional[float] = None,
                          cmap: str = 'Reds',
                          figsize: Tuple[int, int] = (15, 10),
                          show_comparison: bool = False) -> plt.Figure:
        """
        Plot contact maps for selected cells.
        
        Parameters
        ----------
        cell_indices : list of int, optional
            Indices of cells to plot. If None, plots first 4 cells
        chromosome : str, optional
            Specific chromosome to plot. If None, plots whole genome
        vmin, vmax : float
            Color scale limits
        cmap : str, default='Reds'
            Colormap name
        figsize : tuple, default=(15, 10)
            Figure size
        show_comparison : bool, default=False
            If True and data is normalized, show before/after comparison
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with contact maps
        """
        if cell_indices is None:
            cell_indices = list(range(min(4, len(self.hic_data))))
        
        n_cells = len(cell_indices)
        n_cols = 2 if show_comparison and self.is_normalized else 1
        n_rows = n_cells
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, cell_idx in enumerate(cell_indices):
            cell_name = self.cell_names[cell_idx]
            matrix = self.hic_data[cell_idx]
            
            # Get chromosome extent if specified
            if chromosome:
                chrom_bins = self.bins[self.bins['chrom'] == chromosome].index
                if len(chrom_bins) == 0:
                    logger.warning(f"Chromosome {chromosome} not found")
                    continue
                    
                start_bin = chrom_bins[0]
                end_bin = chrom_bins[-1] + 1
                
                # Extract submatrix
                matrix_dense = matrix.toarray()[start_bin:end_bin, start_bin:end_bin]
            else:
                matrix_dense = matrix.toarray()
            
            # Plot normalized data
            ax = axes[i, 0]
            im = ax.imshow(
                np.log1p(matrix_dense),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )
            
            title = f"{cell_name}"
            if self.is_normalized:
                title += f" (Normalized - {self.normalization_method})"
            if chromosome:
                title += f" - Chr{chromosome}"
            ax.set_title(title)
            ax.set_xlabel("Genomic bins")
            ax.set_ylabel("Genomic bins")
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot unnormalized comparison if requested
            if (show_comparison and self.is_normalized and
                    cell_name in self.normalization_weights):
                ax2 = axes[i, 1]
                
                # Recreate unnormalized matrix
                weights = self.normalization_weights[cell_name]
                valid_mask = ~np.isnan(weights[matrix.row]) & ~np.isnan(
                    weights[matrix.col]
                )
                if valid_mask.sum() > 0:
                    unnorm_data = matrix.data[valid_mask] / (
                        weights[matrix.row[valid_mask]] *
                        weights[matrix.col[valid_mask]]
                    )
                    unnorm_matrix = coo_matrix(
                        (unnorm_data,
                         (matrix.row[valid_mask], matrix.col[valid_mask])),
                        shape=matrix.shape
                    )
                    
                    if chromosome:
                        unnorm_dense = unnorm_matrix.toarray()[
                            start_bin:end_bin, start_bin:end_bin
                        ]
                    else:
                        unnorm_dense = unnorm_matrix.toarray()
                    
                    im2 = ax2.imshow(
                        np.log1p(unnorm_dense),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect='auto'
                    )
                    ax2.set_title(f"{cell_name} (Original)")
                    ax2.set_xlabel("Genomic bins")
                    ax2.set_ylabel("Genomic bins")
                    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
    
    def plot_normalization_stats(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot normalization statistics.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with normalization statistics
        """
        if not self.is_normalized or not self.normalization_weights:
            raise ValueError("No normalization has been applied yet.")
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Weight distribution
        ax1 = fig.add_subplot(gs[0, 0])
        all_weights = []
        for weights in self.normalization_weights.values():
            valid_weights = weights[~np.isnan(weights)]
            all_weights.extend(valid_weights)
        
        if all_weights:
            ax1.hist(np.log10(all_weights), bins=50, alpha=0.7, color='blue')
            ax1.set_xlabel("log10(Normalization weights)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Distribution of Normalization Weights")
        
        # 2. Coverage before/after
        ax2 = fig.add_subplot(gs[0, 1])
        coverage_before = []
        coverage_after = []
        
        for i, (cell_name, matrix) in enumerate(
            zip(self.cell_names, self.hic_data)
        ):
            # After normalization
            cov_after = np.array(matrix.tocsr().sum(axis=0)).flatten()
            coverage_after.append(np.sum(cov_after > 0))
            
            # Before normalization (if we have weights)
            if cell_name in self.normalization_weights:
                weights = self.normalization_weights[cell_name]
                # Approximate original coverage
                valid_mask = ~np.isnan(weights[matrix.row]) & ~np.isnan(
                    weights[matrix.col]
                )
                if valid_mask.sum() > 0:
                    orig_data = matrix.data[valid_mask] / (
                        weights[matrix.row[valid_mask]] *
                        weights[matrix.col[valid_mask]]
                    )
                    orig_matrix = coo_matrix(
                        (orig_data,
                         (matrix.row[valid_mask], matrix.col[valid_mask])),
                        shape=matrix.shape
                    ).tocsr()
                    cov_before = np.array(orig_matrix.sum(axis=0)).flatten()
                    coverage_before.append(np.sum(cov_before > 0))
        
        if coverage_before:
            x = np.arange(len(coverage_after))
            width = 0.35
            ax2.bar(
                x - width/2, coverage_before, width,
                label='Before', alpha=0.7
            )
            ax2.bar(
                x + width/2, coverage_after, width,
                label='After', alpha=0.7
            )
            ax2.set_xlabel("Cell index")
            ax2.set_ylabel("Number of non-zero bins")
            ax2.set_title("Coverage Before/After Normalization")
            ax2.legend()
        
        # 3. Sparsity per cell
        ax3 = fig.add_subplot(gs[1, 0])
        sparsities = []
        for matrix in self.hic_data:
            sparsity = 100 * (
                1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
            )
            sparsities.append(sparsity)
        
        ax3.scatter(range(len(sparsities)), sparsities, alpha=0.6)
        ax3.set_xlabel("Cell index")
        ax3.set_ylabel("Sparsity (%)")
        ax3.set_title("Sparsity per Cell")
        ax3.axhline(
            np.mean(sparsities), color='red', linestyle='--',
            label=f'Mean: {np.mean(sparsities):.2f}%'
        )
        ax3.legend()
        
        # 4. Method-specific stats
        ax4 = fig.add_subplot(gs[1, 1])
        if self.normalization_stats:
            stats_text = f"Normalization Method: {self.normalization_method}\n"
            stats_text += (
                f"Total Cells: "
                f"{self.normalization_stats.get('n_cells', len(self.cell_names))}\n"
            )
            
            if 'fraction_valid_bins' in self.normalization_stats:
                frac_stats = self.normalization_stats['fraction_valid_bins']
                stats_text += f"Mean Valid Bins: {frac_stats['mean']:.2%}\n"
                stats_text += f"Std Valid Bins: {frac_stats['std']:.2%}\n"
            
            if 'fraction_converged' in self.normalization_stats:
                stats_text += (
                    f"Converged: "
                    f"{self.normalization_stats['fraction_converged']:.2%}\n"
                )
            
            ax4.text(
                0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center'
            )
            ax4.axis('off')
            ax4.set_title("Normalization Summary")
        
        plt.tight_layout()
        return fig
    
    def subset_from_self(self, selected_names: list) -> "scHiC":
        """
        Create a new scHiC object containing a subset of the current object.

        Parameters
        ----------
        selected_names : list
            List of cell names to include in the subset

        Returns
        -------
        scHiC
            A new scHiC object with filtered data
        """
        selected_indices = [
            self.cell_names.index(name)
            for name in selected_names
            if name in self.cell_names
        ]

        filtered_hic_data = [self.hic_data[i] for i in selected_indices]
        filtered_cell_names = [self.cell_names[i] for i in selected_indices]
        filtered_metadata = self.metadata.set_index('Cool_name').loc[
            selected_names
        ].reset_index()

        subset = scHiC()
        subset.hic_data = filtered_hic_data
        subset.metadata = filtered_metadata
        subset.cell_names = filtered_cell_names
        subset.bins = self.bins
        subset.chroms = self.chroms
        
        # Copy normalization info if present
        if self.is_normalized:
            subset.is_normalized = True
            subset.normalization_method = self.normalization_method
            subset.normalization_weights = {
                name: self.normalization_weights[name]
                for name in filtered_cell_names
                if name in self.normalization_weights
            }

        logger.info(f"Created subset with {len(filtered_cell_names)} cells.")
        return subset

    def describe(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for the loaded Hi-C data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing detailed statistics for each cell
        """
        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        stats = []

        for i, matrix in enumerate(self.hic_data):
            # Extract non-zero values from the sparse matrix
            non_zero_values = matrix.data
            
            # Filter out any potential NaN or inf values
            valid_mask = np.isfinite(non_zero_values)
            valid_values = non_zero_values[valid_mask]

            cell_stats = {
                "Cell Name": self.cell_names[i],
                "Matrix Shape": matrix.shape,
                "Total Contacts": (
                    np.sum(valid_values) if valid_values.size > 0 else 0
                ),
                "Non-zero Contacts": matrix.nnz,
                "Sparsity (%)": 100 * (
                    1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
                ),
                "Mean Contact Value": (
                    np.mean(valid_values) if valid_values.size > 0 else 0
                ),
                "Median Contact Value": (
                    np.median(valid_values) if valid_values.size > 0 else 0
                ),
                "Min Contact Value": (
                    np.min(valid_values) if valid_values.size > 0 else 0
                ),
                "Max Contact Value": (
                    np.max(valid_values) if valid_values.size > 0 else 0
                ),
                "Std Contact Value": (
                    np.std(valid_values) if valid_values.size > 0 else 0
                ),
                "Is Normalized": self.cell_names[i] in self.normalization_weights
            }
            stats.append(cell_stats)

        stats_df = pd.DataFrame(stats)
        return stats_df

    def describe_all(self) -> Dict[str, Any]:
        """
        Compute aggregated statistics for all Hi-C maps.

        Returns
        -------
        dict
            Dictionary containing aggregated statistics for all cells
        """
        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        all_stats = self.describe()
        
        # Filter valid values for aggregation
        valid_means = all_stats[
            all_stats["Mean Contact Value"] > 0
        ]["Mean Contact Value"]
        valid_stds = all_stats[
            all_stats["Std Contact Value"] > 0
        ]["Std Contact Value"]

        aggregated_stats = {
            "Total Cells": len(self.hic_data),
            "Average Matrix Shape": all_stats["Matrix Shape"].mode().iloc[0],
            "Total Contacts (All Cells)": all_stats["Total Contacts"].sum(),
            "Average Non-zero Contacts": all_stats["Non-zero Contacts"].mean(),
            "Average Sparsity (%)": all_stats["Sparsity (%)"].mean(),
            "Average Mean Contact Value": (
                valid_means.mean() if len(valid_means) > 0 else 0
            ),
            "Median Mean Contact Value": (
                valid_means.median() if len(valid_means) > 0 else 0
            ),
            "Min Mean Contact Value": (
                valid_means.min() if len(valid_means) > 0 else 0
            ),
            "Max Mean Contact Value": (
                valid_means.max() if len(valid_means) > 0 else 0
            ),
            "Average Std Contact Value": (
                valid_stds.mean() if len(valid_stds) > 0 else 0
            ),
            "Normalization Status": (
                f"{sum(all_stats['Is Normalized'])}/{len(all_stats)} "
                "cells normalized"
            )
        }
        
        # Add normalization-specific stats if normalized
        if self.is_normalized and self.normalization_method:
            aggregated_stats["Normalization Method"] = self.normalization_method
        
        # Pretty print statistics
        print("\nAggregated Statistics for All Cells:")
        print("=" * 60)
        for key, value in aggregated_stats.items():
            if isinstance(value, float):
                print(f"{key:<40}: {value:.6f}")
            else:
                print(f"{key:<40}: {value}")
        print("=" * 60)

        return aggregated_stats

    def save_to_hdf(
        self,
        output_path: str,
        compression: str = "gzip",
        compression_level: int = 4
    ) -> None:
        """
        Save HiC data and metadata to an HDF5 file.

        Parameters
        ----------
        output_path : str
            Directory to save the HDF5 file and metadata
        compression : str, default="gzip"
            Compression method for HDF5
        compression_level : int, default=4
            Compression level for HDF5
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        hdf5_file = output_path / "hic_data.h5"
        with h5py.File(hdf5_file, 'w') as f:
            # Save matrices
            for i, matrix in enumerate(self.hic_data):
                f.create_dataset(
                    f'data_{i}',
                    data=matrix.toarray(),
                    compression=compression,
                    compression_opts=compression_level
                )
            
            # Save bins and chroms
            if self.bins is not None:
                f.create_dataset('bins', data=self.bins.to_records(index=False))
            if self.chroms is not None:
                f.create_dataset('chroms', data=self.chroms.to_records(index=False))
            
            # Save normalization weights if present
            if self.normalization_weights:
                norm_group = f.create_group('normalization')
                for cell_name, weights in self.normalization_weights.items():
                    norm_group.create_dataset(cell_name, data=weights)
                norm_group.attrs['method'] = self.normalization_method or 'unknown'
                norm_group.attrs['is_normalized'] = self.is_normalized

        # Save metadata
        metadata_file = output_path / "metadata.csv"
        self.metadata.to_csv(metadata_file, index=False)
        logger.info(f"Data saved to {output_path}")

    def save_to_cool(
        self,
        output_dir: str,
        save_weights: bool = True
    ) -> None:
        """
        Save Hi-C data to .cool files.

        Parameters
        ----------
        output_dir : str
            Directory to save the .cool files
        save_weights : bool, default=True
            If True and data is normalized, save weights in cool files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.hic_data:
            raise ValueError("No Hi-C data loaded.")

        if self.bins is None or self.chroms is None:
            raise ValueError("Bin or chromosome information is missing.")

        for i, (matrix, cell_name) in enumerate(
            zip(self.hic_data, self.cell_names)
        ):
            try:
                coo = matrix.tocoo()
                pixels = pd.DataFrame({
                    'bin1_id': coo.row,
                    'bin2_id': coo.col,
                    'count': coo.data
                })

                cool_file = output_dir / f"{cell_name}.cool"
                
                # Prepare bins dataframe
                bins_df = self.bins.copy()
                
                # Add weights if normalized
                if save_weights and cell_name in self.normalization_weights:
                    bins_df['weight'] = self.normalization_weights[cell_name]
                
                cooler.create_cooler(
                    str(cool_file),
                    bins=bins_df,
                    pixels=pixels,
                    ordered=True,
                    dtypes={'count': np.float32},
                    symmetric_upper=True,
                    mode='w'
                )
                logger.info(f"Saved {cool_file.name}")
            except Exception as e:
                logger.error(f"Error saving {cell_name}.cool: {e}")

    @classmethod
    def load_from_hdf(cls, hdf_path: str) -> 'scHiC':
        """
        Load scHiC object from HDF5 file.
        
        Parameters
        ----------
        hdf_path : str
            Path to HDF5 file
            
        Returns
        -------
        scHiC
            Loaded scHiC object
        """
        obj = cls()
        hdf_path = Path(hdf_path)
        
        if hdf_path.is_dir():
            hdf_file = hdf_path / "hic_data.h5"
            metadata_file = hdf_path / "metadata.csv"
        else:
            hdf_file = hdf_path
            metadata_file = hdf_path.parent / "metadata.csv"
        
        with h5py.File(hdf_file, 'r') as f:
            # Load matrices
            i = 0
            while f'data_{i}' in f:
                data = f[f'data_{i}'][:]
                sparse = coo_matrix(data, dtype=obj._dtype)
                obj.hic_data.append(sparse)
                obj.cell_names.append(f"cell_{i}")
                i += 1
            
            # Load bins and chroms
            if 'bins' in f:
                obj.bins = pd.DataFrame(f['bins'][:])
            if 'chroms' in f:
                obj.chroms = pd.DataFrame(f['chroms'][:])
            
            # Load normalization info if present
            if 'normalization' in f:
                norm_group = f['normalization']
                obj.normalization_method = norm_group.attrs.get('method', 'unknown')
                obj.is_normalized = norm_group.attrs.get('is_normalized', False)
                
                for cell_name in norm_group:
                    obj.normalization_weights[cell_name] = norm_group[cell_name][:]
        
        # Load metadata if exists
        if metadata_file.exists():
            obj.metadata = pd.read_csv(metadata_file)
            # Update cell names from metadata
            if 'Cool_name' in obj.metadata.columns:
                obj.cell_names = obj.metadata['Cool_name'].tolist()
        
        logger.info(f"Loaded {len(obj.hic_data)} cells from {hdf_path}")
        return obj


def quick_normalize(
    schic_obj: scHiC,
    method: str = "auto",
    **kwargs
) -> scHiC:
    """
    Quick normalization function for Jupyter notebooks.
    
    Parameters
    ----------
    schic_obj : scHiC
        scHiC object to normalize
    method : str, default="auto"
        Normalization method
    **kwargs
        Additional arguments passed to normalize()
        
    Returns
    -------
    scHiC
        The same object (normalized in-place)
        
    Example
    -------
    >>> schic = scHiC()
    >>> schic.load_cells("path/to/cool/files/")
    >>> schic = quick_normalize(schic, method="coverage")
    """
    schic_obj.normalize(method=method, apply_to_data=True, **kwargs)
    return schic_obj