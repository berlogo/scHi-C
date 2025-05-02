import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import logging
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


# Example usage:
plot_hic_maps(
    schic_obj=example,
    region="chr1:1000000-10000000",
    group_col="MajorType",
    cmap="Reds",
    resolution=1000000  # Show ticks every 1Mb
)
