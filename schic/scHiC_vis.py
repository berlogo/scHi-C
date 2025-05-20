import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.sparse import issparse, coo_matrix
from typing import Dict, List, Optional, Any
from schic import scHiC  
from matplotlib.colors import LogNorm

logger = logging.getLogger(__name__)

def plot_hic_maps(schic_obj, region, group_col, cmap='Reds', resolution=100000):
    """
    Plot Hi-C contact maps with automatic color scaling based on data values.
    
    Parameters:
    - schic_obj: Single-cell Hi-C object containing data
    - region: Genomic region in format 'chr:start-end'
    - group_col: Metadata column to group cells by
    - cmap: Colormap to use (default 'Reds')
    - resolution: Resolution for tick labels in bp (default 100kb)
    """
    
    try:
        chrom, pos = region.split(":")
        start, end = map(int, pos.split("-"))
    except:
        raise ValueError("Region format is invalid. Use 'chr:start-end'.")

    # Get the bins
    bins = schic_obj.bins.reset_index()
    region_bins = bins[
        (bins["chrom"] == chrom) &
        (bins["start"] >= start) &
        (bins["end"] <= end)
    ]

    if region_bins.empty:
        logger.warning(f"No bins found for region {region}")
        return

    logger.info(f"Region {region}: Found {len(region_bins)} bins")
    region_bin_ids = region_bins['index'].values

    # Get genomic positions for ticks
    genomic_positions = region_bins['start'].values
    tick_positions = np.arange(len(genomic_positions))
    tick_labels = [f"{pos//1000000}M" if pos % 1000000 == 0 else "" 
                  for pos in genomic_positions]

    # Get unique groups
    group_values = schic_obj.metadata[group_col].dropna().unique()
    
    # Create figure with subplots
    n_groups = len(group_values)
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    if n_groups == 1:  # if only one group, axes is not an array
        axes = [axes]

    # First pass: calculate global min/max for consistent color scaling
    global_min = float('inf')
    global_max = -float('inf')
    
    for group in group_values:
        group_cells = schic_obj.metadata[schic_obj.metadata[group_col] == group]['Cool_name']
        avg_matrix = np.zeros((len(region_bin_ids), len(region_bin_ids)))
        count = 0
        
        for cell_name in group_cells:
            cell_idx = schic_obj.cell_names.index(cell_name)
            mat = schic_obj.hic_data[cell_idx].tocoo()

            mask = (np.isin(mat.row, region_bin_ids) & 
                   np.isin(mat.col, region_bin_ids))
            if np.sum(mask) > 0:
                local_row = np.searchsorted(region_bin_ids, mat.row[mask])
                local_col = np.searchsorted(region_bin_ids, mat.col[mask])
                avg_matrix[local_row, local_col] += mat.data[mask]
                avg_matrix[local_col, local_row] += mat.data[mask]
                count += 1

        if count > 0:
            avg_matrix /= count
            # Only consider non-zero values for min/max
            nonzero_vals = avg_matrix[avg_matrix > 0]
            if len(nonzero_vals) > 0:
                current_min = np.percentile(nonzero_vals, 5)  # 5th percentile as min
                current_max = np.percentile(nonzero_vals, 95)  # 95th percentile as max
                global_min = min(global_min, current_min)
                global_max = max(global_max, current_max)

    # Adjust global min/max if needed
    if global_min == float('inf'):
        global_min = 1
    if global_max == -float('inf'):
        global_max = 10
    elif global_max <= global_min:
        global_max = global_min * 10

    # Second pass: plot with consistent color scaling
    for idx, group in enumerate(group_values):
        logger.info(f"Processing group: {group}")
        group_cells = schic_obj.metadata[schic_obj.metadata[group_col] == group]['Cool_name']
        avg_matrix = np.zeros((len(region_bin_ids), len(region_bin_ids)))
        count = 0
        
        for cell_name in group_cells:
            cell_idx = schic_obj.cell_names.index(cell_name)
            mat = schic_obj.hic_data[cell_idx].tocoo()

            mask = (np.isin(mat.row, region_bin_ids) & 
                   np.isin(mat.col, region_bin_ids))
            if np.sum(mask) > 0:
                local_row = np.searchsorted(region_bin_ids, mat.row[mask])
                local_col = np.searchsorted(region_bin_ids, mat.col[mask])
                avg_matrix[local_row, local_col] += mat.data[mask]
                avg_matrix[local_col, local_row] += mat.data[mask]
                count += 1

        if count > 0:
            avg_matrix /= count
            
            # Plot with automatically determined color range
            ax = axes[idx]
            im = ax.imshow(avg_matrix, 
                          cmap=cmap, 
                          norm=LogNorm(vmin=global_min, vmax=global_max),
                          aspect='auto',
                          origin='lower',
                          extent=[0, len(region_bin_ids), 0, len(region_bin_ids)])
            
            # Set ticks and labels
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            
            ax.set_title(f"{group}", pad=20)
            ax.set_xlabel('Genomic Position')
            ax.set_ylabel('Genomic Position')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Contact Frequency')
            
            # Add diagonal line
            ax.plot([0, len(region_bin_ids)], [0, len(region_bin_ids)], 
                   'k--', linewidth=0.5, alpha=0.5)
        else:
            logger.warning(f"No valid data found for group {group}")
            axes[idx].axis('off')

    fig.suptitle(f"Hi-C Contact Maps\n{region} (color range: {global_min:.1f}-{global_max:.1f})", y=1.05)
    plt.tight_layout()
    plt.show()


def plot_contact_frequency_distribution(schic: scHiC, bins: int = 50, color: str = "blue") -> None:
    """
    Plot the distribution of contact frequencies across all cells.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    bins : int, optional
        Number of bins for the histogram (default: 50).
    color : str, optional
        Color of the histogram (default: "blue").
    """
    if not schic.hic_data:
        raise ValueError("No Hi-C data loaded.")

    non_zero_values = np.concatenate([matrix.data for matrix in schic.hic_data])

    plt.figure(figsize=(10, 6))
    sns.histplot(non_zero_values, bins=bins, kde=True, color=color)
    plt.title("Distribution of Contact Frequencies")
    plt.xlabel("Contact Frequency")
    plt.ylabel("Density")
    plt.show()

def plot_sparsity_distribution(schic: scHiC, bins: int = 20, color: str = "green") -> None:
    """
    Plot the distribution of sparsity (percentage of zero values) across all cells.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    bins : int, optional
        Number of bins for the histogram (default: 20).
    color : str, optional
        Color of the histogram (default: "green").
    """
    if not schic.hic_data:
        raise ValueError("No Hi-C data loaded.")

    # Calculate sparsity for each cell
    sparsity_values = [100 * (1 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))) for matrix in schic.hic_data]

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(sparsity_values, bins=bins, kde=True, color=color)
    plt.title("Distribution of Sparsity Across Cells")
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Density")
    plt.show()

def plot_mean_contact_boxplot(schic: scHiC, group_column: str, palette: str = "Set2") -> None:
    """
    Plot a boxplot of mean contact values grouped by a metadata column.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    group_column : str
        Column name in metadata used for grouping.
    palette : str, optional
        Color palette for the boxplot (default: "Set2").
    """
    if not schic.hic_data or schic.metadata.empty:
        raise ValueError("No Hi-C data or metadata loaded.")
    if group_column not in schic.metadata.columns:
        raise ValueError(f"Column {group_column} not found in metadata.")

    # Calculate mean contact values for each cell
    mean_values = [matrix.data.mean() for matrix in schic.hic_data]
    schic.metadata["Mean Contact Value"] = mean_values

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_column, y="Mean Contact Value", data=schic.metadata, palette=palette)
    plt.title(f"Distribution of Mean Contact Values by {group_column}")
    plt.xlabel(group_column)
    plt.ylabel("Mean Contact Value")
    plt.show()

def plot_non_zero_violin(schic: scHiC, group_column: str, palette: str = "Set3") -> None:
    """
    Plot a violin plot of non-zero contacts grouped by a metadata column.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    group_column : str
        Column name in metadata used for grouping.
    palette : str, optional
        Color palette for the violin plot (default: "Set3").
    """
    if not schic.hic_data or schic.metadata.empty:
        raise ValueError("No Hi-C data or metadata loaded.")
    if group_column not in schic.metadata.columns:
        raise ValueError(f"Column {group_column} not found in metadata.")

    # Calculate the number of non-zero contacts for each cell
    non_zero_counts = [matrix.nnz for matrix in schic.hic_data]
    schic.metadata["Non-zero Contacts"] = non_zero_counts

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=group_column, y="Non-zero Contacts", data=schic.metadata, palette=palette)
    plt.title(f"Distribution of Non-zero Contacts by {group_column}")
    plt.xlabel(group_column)
    plt.ylabel("Non-zero Contacts")
    plt.show()


