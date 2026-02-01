"""
Analysis and visualization utilities for LTW experiments.

Provides tools for:
- ROC curve generation
- Confusion matrix plotting
- Embedding trajectory visualization
- Comparative analysis
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd


def plot_roc_curve(
    labels: List[bool],
    scores: List[float],
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> Tuple[float, plt.Figure]:
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        labels: True labels (True=positive)
        scores: Detection scores (higher=more positive)
        title: Plot title
        save_path: Path to save figure (optional)
        
    Returns:
        Tuple of (AUC score, matplotlib figure)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return roc_auc, fig


def plot_roc_comparison(
    methods: Dict[str, Tuple[List[bool], List[float]]],
    title: str = "ROC Curve Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple ROC curves for comparison.
    
    Args:
        methods: Dict mapping method name to (labels, scores)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for (name, (labels, scores)), color in zip(methods.items(), colors):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    labels: List[bool],
    predictions: List[bool],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Human', 'AI/Watermarked'],
        yticklabels=['Human', 'AI/Watermarked'],
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trajectory_2d(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    watermarked_rotations: Optional[np.ndarray] = None,
    title: str = "Embedding Trajectory",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 2D PCA projection of embedding trajectory.
    
    Args:
        embeddings: Embeddings array of shape (n, dim)
        labels: Optional text labels for each point
        watermarked_rotations: Optional rotated embeddings to show expected path
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if embeddings.shape[0] < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Not enough data points", ha='center', va='center')
        return fig
    
    # PCA to 2D
    pca = PCA(n_components=2)
    
    # Combine actual and rotated if available
    if watermarked_rotations is not None:
        all_embeddings = np.vstack([embeddings, watermarked_rotations])
        coords = pca.fit_transform(all_embeddings)
        actual_coords = coords[:len(embeddings)]
        rotated_coords = coords[len(embeddings):]
    else:
        actual_coords = pca.fit_transform(embeddings)
        rotated_coords = None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot actual trajectory
    ax.plot(actual_coords[:, 0], actual_coords[:, 1], 'b-o', 
            linewidth=2, markersize=8, label='Actual trajectory')
    
    # Plot expected watermarked trajectory if available
    if rotated_coords is not None:
        ax.plot(rotated_coords[:, 0], rotated_coords[:, 1], 'r--^',
                linewidth=1.5, markersize=6, alpha=0.7, label='Expected (watermarked)')
    
    # Add labels if provided
    if labels:
        for i, (coord, label) in enumerate(zip(actual_coords, labels)):
            ax.annotate(
                f'{i}: {label[:15]}...' if len(label) > 15 else f'{i}: {label}',
                coord, fontsize=8, alpha=0.8
            )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_score_distribution(
    watermarked_scores: List[float],
    human_scores: List[float],
    score_name: str = "Score",
    threshold: Optional[float] = None,
    title: str = "Score Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of scores for watermarked vs human text.
    
    Args:
        watermarked_scores: Scores for watermarked texts
        human_scores: Scores for human texts
        score_name: Name of the score metric
        threshold: Optional detection threshold to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(human_scores, bins=30, alpha=0.6, label='Human', color='blue', density=True)
    ax.hist(watermarked_scores, bins=30, alpha=0.6, label='Watermarked/AI', color='red', density=True)
    
    # Add threshold line if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', 
                   linewidth=2, label=f'Threshold = {threshold:.3f}')
    
    ax.set_xlabel(score_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_anisotropy(
    embeddings: np.ndarray,
    title: str = "Embedding Space Anisotropy",
    save_path: Optional[str] = None
) -> Tuple[Dict[str, float], plt.Figure]:
    """
    Analyze and visualize anisotropy in embedding space.
    
    Args:
        embeddings: Sample embeddings
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Tuple of (anisotropy metrics, figure)
    """
    # Compute covariance and eigenvalues
    centered = embeddings - np.mean(embeddings, axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Anisotropy metrics
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # IsoScore: measure of isotropy (1 = isotropic, 0 = highly anisotropic)
    n_components = min(50, len(eigenvalues))
    top_eigenvalues = eigenvalues[:n_components]
    iso_score = 1 - (np.max(top_eigenvalues) / np.sum(top_eigenvalues))
    
    metrics = {
        "iso_score": iso_score,
        "top_1_variance": explained_variance_ratio[0],
        "top_10_cumulative": cumulative_variance[min(9, len(cumulative_variance)-1)],
        "effective_dimensions": 1 / np.sum(explained_variance_ratio**2),
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Explained variance
    axes[0].plot(range(1, min(51, len(eigenvalues)+1)), 
                 explained_variance_ratio[:50], 'b-o', markersize=3)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Explained Variance per Component', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, min(51, len(eigenvalues)+1)), 
                 cumulative_variance[:50], 'r-o', markersize=3)
    axes[1].axhline(y=0.9, color='green', linestyle='--', label='90% variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Variance', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{title}\n(IsoScore = {iso_score:.3f}, Effective Dims = {metrics["effective_dimensions"]:.1f})', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return metrics, fig


def create_results_dataframe(
    experiment_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from experiment results.
    
    Args:
        experiment_results: List of result dictionaries
        
    Returns:
        Pandas DataFrame
    """
    return pd.DataFrame(experiment_results)


def generate_summary_table(
    methods_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Generate summary comparison table.
    
    Args:
        methods_results: Dict mapping method name to metrics dict
        
    Returns:
        Summary DataFrame
    """
    df = pd.DataFrame(methods_results).T
    df.index.name = 'Method'
    
    # Round numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(4)
    
    return df
