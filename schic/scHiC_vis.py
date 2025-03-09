# visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from typing import Dict, List, Optional, Any
from schic_lib import scHiC  # Импортируем класс scHiC

def extract_group_maps(schic: scHiC, group_column: str) -> Dict[str, np.ndarray]:
    """
    Compute average Hi-C maps for groups of cells based on metadata.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    group_column : str
        Column name in metadata used for grouping.

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping group names to their average contact matrices.
    """
    if not schic.hic_data or schic.metadata.empty:
        raise ValueError("No Hi-C data or metadata loaded.")
    if group_column not in schic.metadata.columns:
        raise ValueError(f"Column {group_column} not found in metadata.")

    group_maps = {}
    for group, group_indices in schic.metadata.groupby(group_column).groups.items():
        group_matrices = [schic.hic_data[i] for i in group_indices]
        group_maps[group] = np.mean([matrix.toarray() for matrix in group_matrices], axis=0)
    return group_maps

def visualize_maps(
    schic: scHiC,
    group_column: Optional[str] = None,
    average: bool = False,
    cell_names: Optional[List[str]] = None,
    n: int = 5
) -> None:
    """
    Visualize Hi-C maps, either grouped by metadata or specific cells, using sparse matrices to save memory.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    group_column : str, optional
        Column name in metadata used for grouping. If None, no grouping is applied.
    average : bool, optional
        Whether to visualize average maps for groups (default: False).
    cell_names : List[str], optional
        List of specific cell names to visualize. If provided, `group_column` and `average` are ignored.
    n : int, optional
        Number of random maps to visualize if no specific cells are provided (default: 5).
    """
    if not schic.hic_data:
        raise ValueError("No Hi-C data loaded.")

    if cell_names:
        # Visualize specific cells
        indices = [schic.cell_names.index(name) for name in cell_names if name in schic.cell_names]
        if not indices:
            raise ValueError("No valid cell names provided.")
        matrices = [schic.hic_data[i] for i in indices]
        titles = [schic.cell_names[i] for i in indices]
    elif group_column:
        # Group by metadata and visualize
        if group_column not in schic.metadata.columns:
            raise ValueError(f"Column {group_column} not found in metadata.")
        
        if average:
            # Visualize average maps for each group
            group_maps = extract_group_maps(schic, group_column)
            matrices = list(group_maps.values())
            titles = list(group_maps.keys())
        else:
            # Visualize random maps from each group
            groups = schic.metadata.groupby(group_column).groups
            matrices = []
            titles = []
            for group, indices in groups.items():
                n_samples = min(n, len(indices))
                random_indices = np.random.choice(indices, n_samples, replace=False)
                matrices.extend([schic.hic_data[i] for i in random_indices])
                titles.extend([f"{group} - {schic.cell_names[i]}" for i in random_indices])
    else:
        # Visualize random maps
        n = min(n, len(schic.hic_data))
        random_indices = np.random.choice(len(schic.hic_data), n, replace=False)
        matrices = [schic.hic_data[i] for i in random_indices]
        titles = [schic.cell_names[i] for i in random_indices]

    # Plot the maps
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]  # Ensure axes is a list for single plots

    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        if issparse(matrix):
            # Convert sparse matrix to dense only for visualization
            matrix = matrix.toarray()
        # Ensure no negative values before applying log1p
        if np.any(matrix < 0):
            raise ValueError("Matrix contains negative values. Log scaling cannot be applied.")
        axes[i].imshow(np.log1p(matrix), cmap="Reds")
        axes[i].set_title(title)
        axes[i].axis("off")

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

    # Collect all non-zero contact frequencies
    non_zero_values = np.concatenate([matrix.data for matrix in schic.hic_data])

    # Plot the distribution
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

    # Add mean values to metadata
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

    # Add non-zero counts to metadata
    schic.metadata["Non-zero Contacts"] = non_zero_counts

    # Plot the violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=group_column, y="Non-zero Contacts", data=schic.metadata, palette=palette)
    plt.title(f"Distribution of Non-zero Contacts by {group_column}")
    plt.xlabel(group_column)
    plt.ylabel("Non-zero Contacts")
    plt.show()

def plot_cumulative_contact_distribution(schic: scHiC, color: str = "purple") -> None:
    """
    Plot the cumulative distribution of contact frequencies.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    color : str, optional
        Color of the cumulative distribution plot (default: "purple").
    """
    if not schic.hic_data:
        raise ValueError("No Hi-C data loaded.")

    # Collect all non-zero contact frequencies
    non_zero_values = np.concatenate([matrix.data for matrix in schic.hic_data])

    # Plot the cumulative distribution
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(non_zero_values, color=color)
    plt.title("Cumulative Distribution of Contact Frequencies")
    plt.xlabel("Contact Frequency")
    plt.ylabel("Cumulative Probability")
    plt.show()

def plot_correlation_heatmap(schic: scHiC, cmap: str = "coolwarm") -> None:
    """
    Plot a heatmap of correlation between cells based on their contact maps.

    Parameters:
    -----------
    schic : scHiC
        An instance of the scHiC class containing Hi-C data.
    cmap : str, optional
        Color map for the heatmap (default: "coolwarm").
    """
    if not schic.hic_data:
        raise ValueError("No Hi-C data loaded.")

    # Compute pairwise correlations between cells
    correlation_matrix = np.corrcoef([matrix.toarray().flatten() for matrix in schic.hic_data])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, xticklabels=schic.cell_names, yticklabels=schic.cell_names)
    plt.title("Correlation Between Cells")
    plt.xlabel("Cells")
    plt.ylabel("Cells")
    plt.show()
    