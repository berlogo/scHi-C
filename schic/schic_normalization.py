#!/usr/bin/env python3
"""
schic_normalization.py - Single-cell Hi-C normalization module using cooltools.

Optimized for extremely sparse matrices with proper handling of low coverage regions.
Follows PEP8 and Flake8 standards.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import cooler
import cooltools
import h5py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("schic_normalization")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration Classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class scHiCNormConfig:
    """Configuration for single-cell Hi-C normalization."""
    
    # Normalization method
    method: str = "coverage"  # "coverage", "VC", "SCALE", "ICE" (ICE only for dense)
    
    # scHiC specific parameters
    min_nnz: int = 50  # minimum non-zero entries per chromosome
    min_coverage: float = 0.01  # minimum coverage fraction to keep bin
    coverage_threshold: int = 10  # minimum marginal sum for valid bins
    
    # ICE parameters (only for dense matrices)
    ice_max_iters: int = 200
    ice_tol: float = 1e-5
    ice_ignore_diags: int = 2  # diagonals to ignore in ICE
    ice_min_nnz_per_bin: int = 3  # min contacts per bin for ICE
    
    # Processing options
    cis_only: bool = True  # normalize only cis contacts
    log_transform: bool = False  # apply log1p transformation
    remove_outliers: bool = True  # remove outlier bins
    outlier_zscore: float = 4.0  # z-score threshold for outliers
    
    # Memory optimization
    chunk_size: int = 1000  # process matrices in chunks
    use_float32: bool = True  # use float32 to save memory

    @classmethod
    def for_ultra_sparse_data(cls):
        """Configuration optimized for ultra-sparse scHiC data."""
        return cls(
            method="SCALE",
            min_nnz=10,  # Very low threshold
            min_coverage=0.001,
            coverage_threshold=1,  # Accept any non-zero bin
            ice_min_nnz_per_bin=1,
            remove_outliers=False,  # Don't remove outliers in sparse data
            cis_only=True
        )


@dataclass
class NormalizationResult:
    """Result of normalization process."""
    
    cool_paths: List[str]
    weights: Dict[str, np.ndarray]
    stats: Dict[str, Any]
    failed_cells: List[int]
    converged: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Core Normalization Functions
# ──────────────────────────────────────────────────────────────────────────────


class scHiCNormalizer:
    """
    Single-cell Hi-C normalizer using cooltools.
    
    Handles extremely sparse matrices with proper coverage checks.
    """
    
    def __init__(self, config: scHiCNormConfig):
        """Initialize normalizer with configuration."""
        self.config = config
        
    def normalize_batch(
        self,
        cool_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        n_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> NormalizationResult:
        """Normalize a batch of single-cell Hi-C cool files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process in chunks to manage memory
        n_cells = len(cool_paths)
        chunk_size = self.config.chunk_size
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        
        all_results = []
        all_weights = {}
        all_stats = []
        failed_cells = []
        
        logger.info(f"Processing {n_cells} cells in {n_chunks} chunks")
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_cells)
            chunk_paths = cool_paths[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks}")
            
            # Process chunk
            if n_workers and n_workers > 1:
                chunk_results = self._process_chunk_parallel(
                    chunk_paths, output_dir, start_idx, n_workers, show_progress
                )
            else:
                chunk_results = self._process_chunk_serial(
                    chunk_paths, output_dir, start_idx, show_progress
                )
            
            # Collect results
            for result in chunk_results:
                if result['success']:
                    all_results.append(result['output_path'])
                    if result['weights'] is not None:
                        all_weights[result['cell_id']] = result['weights']
                    all_stats.append(result['stats'])
                else:
                    failed_cells.append(result['cell_idx'])
                    
        # Aggregate statistics
        aggregated_stats = self._aggregate_stats(all_stats)
        
        # Check if any cells were successfully normalized
        if len(all_results) == 0:
            logger.warning(
                f"No cells were successfully normalized out of {n_cells} cells."
            )
            logger.warning(
                "This may be due to insufficient coverage or too stringent parameters."
            )
            logger.warning("Consider:")
            logger.warning(
                "  1. Lowering coverage_threshold (current: %d)",
                self.config.coverage_threshold
            )
            logger.warning(
                "  2. Lowering min_nnz (current: %d)",
                self.config.min_nnz
            )
            logger.warning("  3. Using 'SCALE' method for very sparse data")
            logger.warning("  4. Running diagnose() to check data quality")
        
        return NormalizationResult(
            cool_paths=all_results,
            weights=all_weights,
            stats=aggregated_stats,
            failed_cells=failed_cells,
            converged=True
        )
    
    def _process_chunk_serial(
        self,
        chunk_paths: List[Union[str, Path]],
        output_dir: Path,
        start_idx: int,
        show_progress: bool
    ) -> List[Dict]:
        """Process chunk serially."""
        results = []
        iterator = enumerate(chunk_paths)
        
        if show_progress and TQDM_AVAILABLE:
            iterator = tqdm(
                iterator,
                total=len(chunk_paths),
                desc="Normalizing cells"
            )
            
        for i, cool_path in iterator:
            cell_idx = start_idx + i
            result = self._normalize_single_cell(cool_path, output_dir, cell_idx)
            results.append(result)
            
        return results
    
    def _process_chunk_parallel(
        self,
        chunk_paths: List[Union[str, Path]],
        output_dir: Path,
        start_idx: int,
        n_workers: int,
        show_progress: bool
    ) -> List[Dict]:
        """Process chunk in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, cool_path in enumerate(chunk_paths):
                cell_idx = start_idx + i
                future = executor.submit(
                    self._normalize_single_cell,
                    cool_path,
                    output_dir,
                    cell_idx
                )
                future_to_idx[future] = cell_idx
            
            # Collect results
            iterator = as_completed(future_to_idx)
            if show_progress and TQDM_AVAILABLE:
                iterator = tqdm(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="Normalizing cells"
                )
                
            for future in iterator:
                result = future.result()
                results.append(result)
                
        # Sort by cell index
        results.sort(key=lambda x: x['cell_idx'])
        return results
    
    def _normalize_single_cell(
        self,
        cool_path: Union[str, Path],
        output_dir: Path,
        cell_idx: int
    ) -> Dict:
        """Normalize a single cell."""
        cool_path = Path(cool_path)
        cell_id = cool_path.stem
        
        try:
            # Load cooler
            clr = cooler.Cooler(str(cool_path))
            
            # Check if matrix has enough data
            n_contacts = clr.info['sum']
            if n_contacts < self.config.min_nnz:
                logger.warning(
                    f"Cell {cell_id}: insufficient contacts "
                    f"({n_contacts} < {self.config.min_nnz})"
                )
                return {
                    'cell_idx': cell_idx,
                    'cell_id': cell_id,
                    'success': False,
                    'reason': f'insufficient_contacts: {n_contacts} < {self.config.min_nnz}'
                }
            
            # Get normalization weights based on method
            if self.config.method == "coverage":
                weights, stats = self._coverage_normalization(clr)
            elif self.config.method == "ICE":
                weights, stats = self._ice_normalization(clr)
            elif self.config.method == "VC":
                weights, stats = self._vc_normalization(clr)
            elif self.config.method == "SCALE":
                weights, stats = self._scale_normalization(clr)
            else:
                raise ValueError(f"Unknown normalization method: {self.config.method}")
            
            # Save normalized matrix
            output_path = output_dir / f"{cell_id}_normalized.cool"
            self._save_normalized_matrix(clr, weights, output_path)
            
            return {
                'cell_idx': cell_idx,
                'cell_id': cell_id,
                'success': True,
                'output_path': str(output_path),
                'weights': weights,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize cell {cell_id}: {str(e)}")
            return {
                'cell_idx': cell_idx,
                'cell_id': cell_id,
                'success': False,
                'reason': str(e)
            }
    
    def _coverage_normalization(
        self,
        clr: cooler.Cooler
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple coverage-based normalization for sparse scHiC.
        
        Most suitable for very sparse single-cell data.
        """
        # Calculate coverage (marginal sums)
        if self.config.cis_only:
            coverage = np.zeros(clr.shape[0])
            for chrom in clr.chromnames:
                extent = clr.extent(chrom)
                # Important: use [:] to get actual sparse matrix
                matrix = clr.matrix(balance=False, sparse=True)[
                    extent[0]:extent[1], extent[0]:extent[1]
                ]
                coverage[extent[0]:extent[1]] = np.asarray(
                    matrix.sum(axis=0)
                ).flatten()
        else:
            # Get full matrix with [:]
            matrix = clr.matrix(balance=False, sparse=True)[:]
            coverage = np.asarray(matrix.sum(axis=0)).flatten()
        
        # Find valid bins
        valid = coverage >= self.config.coverage_threshold
        n_valid = valid.sum()
        
        # Check if we have enough valid bins
        if n_valid < 10:  # Minimum number of valid bins
            logger.warning(
                f"Only {n_valid} bins have coverage >= {self.config.coverage_threshold}"
            )
            # For very sparse data, relax the threshold
            if n_valid == 0:
                # Use all non-zero bins
                valid = coverage > 0
                n_valid = valid.sum()
                logger.warning(f"Using all {n_valid} non-zero bins instead")
        
        # Calculate weights
        weights = np.ones(clr.shape[0])
        if n_valid > 0:
            mean_coverage = coverage[valid].mean()
            # Coverage normalization: weight = sqrt(mean/coverage)
            weights[valid] = np.sqrt(mean_coverage / coverage[valid])
        else:
            logger.error("No valid bins found for normalization")
            weights[:] = np.nan
        
        # Remove outliers if requested
        if self.config.remove_outliers and n_valid > 0:
            weights_valid = weights[valid]
            z_scores = np.abs(
                (weights_valid - weights_valid.mean()) / weights_valid.std()
            )
            outliers = z_scores > self.config.outlier_zscore
            weights[np.where(valid)[0][outliers]] = np.nan
        
        # Set invalid bins to NaN
        weights[~valid] = np.nan
        
        stats = {
            'method': 'coverage',
            'n_valid_bins': n_valid,
            'coverage_mean': coverage[valid].mean() if n_valid > 0 else 0,
            'coverage_std': coverage[valid].std() if n_valid > 0 else 0,
            'fraction_valid': n_valid / clr.shape[0]
        }
        
        weights = np.nan_to_num(weights, nan=1.0)
        
        return weights, stats
    
    def _ice_normalization(
        self,
        clr: cooler.Cooler
    ) -> Tuple[np.ndarray, Dict]:
        """
        ICE normalization using cooltools.
        
        Note: Only suitable for matrices with sufficient coverage.
        Not recommended for ultra-sparse scHiC data.
        """
        try:
            # Check if matrix has enough data for ICE
            matrix = clr.matrix(balance=False, sparse=True)[:]
            total_contacts = matrix.sum()
            
            if total_contacts < 10000:
                logger.warning(
                    f"Matrix has only {total_contacts} contacts, "
                    "ICE is not suitable for ultra-sparse data"
                )
                logger.warning("Falling back to coverage normalization")
                return self._coverage_normalization(clr)
            
            # Use cooltools ICE implementation
            bias, stats = cooltools.balance_cooler(
                clr,
                cis_only=self.config.cis_only,
                trans_only=False,
                max_iters=self.config.ice_max_iters,
                tol=self.config.ice_tol,
                min_nnz=self.config.ice_min_nnz_per_bin,
                min_count=self.config.coverage_threshold,
                ignore_diags=self.config.ice_ignore_diags,
                rescale_marginals=True,
                use_lock=False,
                store=False
            )
            
            # Convert bias to weights (1/bias)
            weights = np.ones_like(bias)
            valid = ~np.isnan(bias) & (bias > 0)
            weights[valid] = 1.0 / bias[valid]
            weights[~valid] = np.nan
            
            ice_stats = {
                'method': 'ICE',
                'converged': stats.get('converged', True),
                'n_iter': stats.get('n_iter', 0),
                'n_valid_bins': valid.sum(),
                'fraction_valid': valid.sum() / len(bias)
            }
            
            return weights, ice_stats
            
        except Exception as e:
            logger.error(f"ICE normalization failed: {str(e)}")
            logger.warning("Falling back to coverage normalization")
            # Fallback to coverage normalization
            return self._coverage_normalization(clr)
    
    def _vc_normalization(
        self,
        clr: cooler.Cooler
    ) -> Tuple[np.ndarray, Dict]:
        """
        Vanilla Coverage (VC) normalization - sqrt of coverage.
        
        Similar to coverage but uses square root.
        """
        # Calculate coverage
        if self.config.cis_only:
            coverage = np.zeros(clr.shape[0])
            for chrom in clr.chromnames:
                extent = clr.extent(chrom)
                matrix = clr.matrix(balance=False, sparse=True)[
                    extent[0]:extent[1], extent[0]:extent[1]
                ]
                coverage[extent[0]:extent[1]] = np.asarray(
                    matrix.sum(axis=0)
                ).flatten()
        else:
            matrix = clr.matrix(balance=False, sparse=True)[:]
            coverage = np.asarray(matrix.sum(axis=0)).flatten()
        
        # Find valid bins
        valid = coverage >= self.config.coverage_threshold
        n_valid = valid.sum()
        
        if n_valid == 0:
            valid = coverage > 0
            n_valid = valid.sum()
        
        # VC uses sqrt of coverage for weights
        weights = np.ones(clr.shape[0])
        if n_valid > 0:
            # VC normalization: weight = 1/sqrt(coverage)
            weights[valid] = 1.0 / np.sqrt(coverage[valid])
            # Normalize to mean of 1
            weights[valid] = weights[valid] / np.nanmean(weights[valid])
        else:
            weights[:] = np.nan
        
        # Set invalid bins to NaN
        weights = np.nan_to_num(weights, nan=1.0)
        
        stats = {
            'method': 'VC',
            'n_valid_bins': n_valid,
            'coverage_mean': coverage[valid].mean() if n_valid > 0 else 0,
            'coverage_std': coverage[valid].std() if n_valid > 0 else 0,
            'fraction_valid': n_valid / clr.shape[0]
        }
        
        return weights, stats
    
    def _scale_normalization(
        self,
        clr: cooler.Cooler
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple scaling normalization - scale all contacts by total sum.
        
        Most robust for ultra-sparse data. Normalizes to 1M total contacts.
        """
        total_contacts = clr.info['sum']
        
        if total_contacts == 0:
            logger.error("Matrix has zero contacts, cannot normalize")
            return np.full(clr.shape[0], np.nan), {
                'method': 'SCALE',
                'total_contacts': 0,
                'scale_factor': np.nan,
                'n_valid_bins': 0,
                'fraction_valid': 0.0
            }
        
        # Scale factor to normalize to 1M contacts
        scale_factor = 1e6 / total_contacts
        
        # For SCALE method, apply sqrt of scale factor as weight
        # This ensures that when weights are applied as w[i]*w[j],
        # the total scaling is scale_factor
        weights = np.full(clr.shape[0], np.sqrt(scale_factor))
        
        stats = {
            'method': 'SCALE',
            'total_contacts': total_contacts,
            'scale_factor': scale_factor,
            'n_valid_bins': clr.shape[0],
            'fraction_valid': 1.0
        }
        
        return weights, stats
    
    def _save_normalized_matrix(
        self,
        clr: cooler.Cooler,
        weights: np.ndarray,
        output_path: Path
    ) -> None:
        """Save normalized matrix to a new cool file."""
        try:
            # Get bins and add weights column
            bins_df = clr.bins()[:].copy()
            bins_df['weight'] = weights
            
            # Create a new cooler file with weights
            cooler.create_cooler(
                cool_uri=str(output_path),
                bins=bins_df,
                pixels=clr.pixels()[:],
                dtypes={'count': np.float32},
                ordered=True,
                symmetric_upper=True,
                mode='w'
            )
            
            # Add metadata about normalization
            with h5py.File(str(output_path), 'r+') as f:
                grp = f['/']
                grp.attrs['normalization-method'] = self.config.method
                grp.attrs['normalization-params'] = str({
                    'min_coverage': self.config.min_coverage,
                    'coverage_threshold': self.config.coverage_threshold,
                    'cis_only': self.config.cis_only,
                    'min_nnz': self.config.min_nnz
                })
                
            logger.debug(f"Successfully saved normalized matrix to {output_path}")
            
        except Exception as e:
            logger.error(
                f"Failed to save normalized matrix to {output_path}: {str(e)}"
            )
            raise
    
    def _aggregate_stats(self, stats_list: List[Dict]) -> Dict:
        """Aggregate statistics across all cells."""
        if not stats_list:
            return {
                'n_cells': 0,
                'method': self.config.method,
                'error': 'No cells were successfully normalized'
            }
            
        aggregated = {
            'n_cells': len(stats_list),
            'method': self.config.method,
            'fraction_valid_bins': {
                'mean': np.mean([s['fraction_valid'] for s in stats_list]),
                'std': np.std([s['fraction_valid'] for s in stats_list]),
                'min': np.min([s['fraction_valid'] for s in stats_list]),
                'max': np.max([s['fraction_valid'] for s in stats_list])
            }
        }
        
        if self.config.method == 'ICE':
            converged = [s.get('converged', True) for s in stats_list]
            aggregated['fraction_converged'] = (
                sum(converged) / len(converged) if converged else 0
            )
            aggregated['mean_iterations'] = np.mean([
                s.get('n_iter', 0) for s in stats_list
            ]) if stats_list else 0
            
        return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────


def diagnose_sparsity(
    cool_paths: List[Union[str, Path]],
    sample_size: int = 10
) -> Dict:
    """
    Diagnose sparsity patterns in scHiC data.
    
    Useful for choosing normalization parameters.
    """
    sample_paths = cool_paths[:min(sample_size, len(cool_paths))]
    
    stats = {
        'n_contacts': [],
        'fraction_nonzero': [],
        'coverage_per_bin': [],
        'cis_trans_ratio': []
    }
    
    for path in sample_paths:
        clr = cooler.Cooler(str(path))
        
        # Total contacts
        stats['n_contacts'].append(clr.info['sum'])
        
        # Fraction of non-zero pixels
        n_pixels = clr.pixels()[:].shape[0]
        n_possible = clr.shape[0] * (clr.shape[0] + 1) // 2  # upper triangle
        stats['fraction_nonzero'].append(n_pixels / n_possible)
        
        # Average coverage per bin
        matrix = clr.matrix(balance=False, sparse=True)[:]
        coverage = np.asarray(matrix.sum(axis=0)).flatten()
        stats['coverage_per_bin'].append(coverage.mean())
        
        # Cis/trans ratio
        cis_sum = 0
        trans_sum = 0
        for chrom in clr.chromnames:
            extent = clr.extent(chrom)
            cis_matrix = clr.matrix(balance=False, sparse=True)[
                extent[0]:extent[1], extent[0]:extent[1]
            ]
            cis_sum += cis_matrix.sum()
        trans_sum = clr.info['sum'] - cis_sum
        ratio = cis_sum / trans_sum if trans_sum > 0 else np.inf
        stats['cis_trans_ratio'].append(ratio)
    
    # Compute summary statistics
    summary = {}
    for key, values in stats.items():
        if len(values) > 0:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        else:
            summary[key] = {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan
            }
    
    # Recommendations based on sparsity
    mean_contacts = (
        summary['n_contacts']['mean']
        if not np.isnan(summary['n_contacts']['mean'])
        else 0
    )
    mean_coverage = (
        summary['coverage_per_bin']['mean']
        if not np.isnan(summary['coverage_per_bin']['mean'])
        else 0
    )
    
    # Adjust recommendations for very sparse data
    if mean_contacts < 1e4:  # Ultra-sparse
        recommendations = {
            'suggested_method': 'SCALE',  # Simple scaling best for ultra-sparse
            'suggested_min_coverage': 1,  # Accept any non-zero bin
            'suggested_coverage_threshold': 1,  # Very low threshold
            'use_cis_only': True,  # Focus on cis contacts
            'warning': 'Data is extremely sparse. Use SCALE method or merge cells.'
        }
    elif mean_contacts < 1e5:  # Sparse
        recommendations = {
            'suggested_method': 'coverage',
            'suggested_min_coverage': max(1, int(mean_coverage * 0.05)),
            'suggested_coverage_threshold': max(2, int(mean_coverage * 0.1)),
            'use_cis_only': True
        }
    elif mean_contacts < 1e6:  # Medium sparse
        recommendations = {
            'suggested_method': 'VC',  # VC for medium coverage
            'suggested_min_coverage': max(1, int(mean_coverage * 0.1)),
            'suggested_coverage_threshold': max(5, int(mean_coverage * 0.2)),
            'use_cis_only': True
        }
    else:  # Dense (>1M contacts) - only here recommend ICE
        recommendations = {
            'suggested_method': 'ICE',
            'suggested_min_coverage': max(1, int(mean_coverage * 0.1)),
            'suggested_coverage_threshold': max(10, int(mean_coverage * 0.2)),
            'use_cis_only': (
                summary['cis_trans_ratio']['mean'] > 3
                if not np.isnan(summary['cis_trans_ratio']['mean'])
                else True
            )
        }
    
    return {
        'stats': summary,
        'recommendations': recommendations,
        'n_cells_analyzed': len(sample_paths)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main API Functions
# ──────────────────────────────────────────────────────────────────────────────


def normalize_schic(
    cool_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    method: str = "coverage",
    config: Optional[scHiCNormConfig] = None,
    n_workers: Optional[int] = None,
    diagnose_first: bool = True,
    show_progress: bool = True
) -> NormalizationResult:
    """
    Main function to normalize single-cell Hi-C data.
    
    Parameters
    ----------
    cool_paths : list of paths
        Paths to single-cell .cool files
    output_dir : path
        Directory to save normalized files
    method : str
        Normalization method: "coverage", "ICE", "VC", "SCALE"
        Note: ICE is only recommended for dense matrices
    config : scHiCNormConfig, optional
        Configuration object. If None, will auto-configure based on data
    n_workers : int, optional
        Number of parallel workers. None for auto-detect
    diagnose_first : bool
        Whether to run sparsity diagnosis to auto-configure parameters
    show_progress : bool
        Show progress bars
        
    Returns
    -------
    NormalizationResult
        Contains normalized file paths, weights, and statistics
    """
    # Auto-configure if needed
    if config is None:
        config = scHiCNormConfig(method=method)
        
        if diagnose_first:
            logger.info("Diagnosing data sparsity patterns...")
            diagnosis = diagnose_sparsity(cool_paths)
            
            # Apply recommendations
            rec = diagnosis['recommendations']
            if method == "auto":
                config.method = rec['suggested_method']
                logger.info(f"Auto-selected method: {config.method}")
            
            config.min_coverage = rec['suggested_min_coverage'] / 100  # as fraction
            config.coverage_threshold = rec['suggested_coverage_threshold']
            config.cis_only = rec['use_cis_only']
            
            # Check for ultra-sparse warning
            if 'warning' in rec:
                logger.warning(rec['warning'])
                # Use ultra-sparse configuration
                config = scHiCNormConfig.for_ultra_sparse_data()
                if method != "auto":
                    config.method = method  # Keep user's choice if specified
            
            logger.info(
                f"Auto-configured: method={config.method}, "
                f"coverage_threshold={config.coverage_threshold}, "
                f"cis_only={config.cis_only}"
            )
    
    # Warn if ICE is selected for sparse data
    if config.method == "ICE" and diagnose_first:
        diagnosis = diagnose_sparsity(cool_paths[:10])
        mean_contacts = diagnosis['stats']['n_contacts']['mean']
        if mean_contacts < 1e6:
            logger.warning(
                f"ICE normalization is not recommended for sparse data "
                f"(mean contacts: {mean_contacts:.0f}). "
                "Consider using 'coverage', 'VC', or 'SCALE' instead."
            )
    
    # Set default workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 for memory efficiency
    
    # Initialize normalizer
    normalizer = scHiCNormalizer(config)
    
    # Run normalization
    logger.info(f"Starting {config.method} normalization on {len(cool_paths)} cells")
    logger.info(
        f"Parameters: min_nnz={config.min_nnz}, "
        f"coverage_threshold={config.coverage_threshold}"
    )
    
    result = normalizer.normalize_batch(
        cool_paths,
        output_dir,
        n_workers=n_workers,
        show_progress=show_progress
    )
    
    # Log summary
    success_rate = len(result.cool_paths) / len(cool_paths) if cool_paths else 0
    logger.info(
        f"Normalization complete: {len(result.cool_paths)}/{len(cool_paths)} "
        f"cells normalized successfully ({success_rate:.1%})"
    )
    
    if result.failed_cells:
        logger.warning(f"Failed cells: {len(result.failed_cells)}")
        if len(result.failed_cells) == len(cool_paths):
            logger.error("All cells failed normalization!")
            logger.error("Suggestions:")
            logger.error("1. Use scHiCNormConfig.for_ultra_sparse_data() configuration")
            logger.error("2. Try method='SCALE' for simple scaling normalization")
            logger.error("3. Check if your data files are valid with cooler.info()")
    
    return result


def test_single_cell(
    cool_path: Union[str, Path],
    method: str = "auto",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test normalization on a single cell for debugging.
    
    Parameters
    ----------
    cool_path : str or Path
        Path to single .cool file
    method : str
        Normalization method to test
    verbose : bool
        Print detailed information
        
    Returns
    -------
    dict
        Test results including success status and statistics
    """    
    cool_path = Path(cool_path)
    if not cool_path.exists():
        raise FileNotFoundError(f"File not found: {cool_path}")
    
    # Load cooler and get basic stats
    clr = cooler.Cooler(str(cool_path))
    
    test_info = {
        'file': cool_path.name,
        'shape': clr.shape,
        'total_contacts': clr.info.get('sum', 0),
        'bins': len(clr.bins()),
        'chromosomes': len(clr.chromnames)
    }
    
    if verbose:
        print(f"Testing normalization on: {cool_path.name}")
        print(f"Matrix shape: {test_info['shape']}")
        print(f"Total contacts: {test_info['total_contacts']:,}")
        print(f"Number of bins: {test_info['bins']:,}")
        print(f"Chromosomes: {test_info['chromosomes']}")
    
    # Calculate sparsity
    pixels = clr.pixels()[:]
    n_pixels = len(pixels)
    max_pixels = test_info['shape'][0] * test_info['shape'][1]
    sparsity = 100 * (1 - n_pixels / max_pixels)
    test_info['sparsity_percent'] = sparsity
    test_info['non_zero_pixels'] = n_pixels
    
    if verbose:
        print(f"Non-zero pixels: {n_pixels:,}")
        print(f"Sparsity: {sparsity:.6f}%")
    
    # Try normalization
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Configure based on sparsity
            if test_info['total_contacts'] < 10000:
                config = scHiCNormConfig.for_ultra_sparse_data()
                if verbose:
                    print(f"\nUsing ultra-sparse configuration")
            else:
                config = scHiCNormConfig(method=method)
            
            normalizer = scHiCNormalizer(config)
            result = normalizer._normalize_single_cell(
                cool_path,
                Path(tmpdir),
                0
            )
            
            test_info['normalization_success'] = result['success']
            
            if result['success']:
                test_info['normalization_stats'] = result['stats']
                test_info['valid_bins'] = result['stats'].get('n_valid_bins', 0)
                test_info['fraction_valid'] = result['stats'].get('fraction_valid', 0)
                
                if verbose:
                    print(f"\nNormalization SUCCESSFUL!")
                    print(f"Method: {result['stats']['method']}")
                    print(
                        f"Valid bins: {test_info['valid_bins']:,} / "
                        f"{test_info['bins']:,}"
                    )
                    print(f"Fraction valid: {test_info['fraction_valid']:.2%}")
            else:
                test_info['failure_reason'] = result.get('reason', 'Unknown')
                if verbose:
                    print(f"\nNormalization FAILED!")
                    print(f"Reason: {test_info['failure_reason']}")
                    
        except Exception as e:
            test_info['normalization_success'] = False
            test_info['failure_reason'] = str(e)
            if verbose:
                print(f"\nNormalization ERROR: {e}")
    
    # Recommendations
    if not test_info['normalization_success']:
        recommendations = []
        
        if test_info['total_contacts'] < 1000:
            recommendations.append("Use SCALE method for ultra-sparse data")
            recommendations.append("Consider merging with other cells")
        elif test_info['total_contacts'] < 10000:
            recommendations.append("Use coverage method with threshold=1")
            recommendations.append("Disable outlier removal")
        elif test_info['total_contacts'] < 100000:
            recommendations.append("Use VC normalization for medium-sparse data")
        
        test_info['recommendations'] = recommendations
        
        if verbose and recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
    
    return test_info