import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Figure S1A: PCA of molecular descriptor space
def create_figure_s1():
    """
    PCA analysis showing uniform distribution of 1,200 molecules
    across chemical descriptor space
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    
    # ========== Panel A: PCA Biplot ==========
    ax1 = axes[0, 0]
    
    # Simulate data (replace with actual data)
    np.random.seed(42)
    n_molecules = 1200
    
    # Generate realistic molecular descriptors
    MW = np.random.gamma(shape=5, scale=80, size=n_molecules) + 180
    logP = np.random.normal(2, 1.5, n_molecules)
    TPSA = np.random.gamma(shape=3, scale=20, size=n_molecules) + 20
    HBD = np.random.poisson(2, n_molecules)
    HBA = np.random.poisson(4, n_molecules)
    nRotB = np.random.poisson(5, n_molecules)
    nArom = np.random.poisson(2, n_molecules)
    
    # Target classes
    target_class = np.array(['Kinases']*450 + ['GPCRs']*400 + ['Proteases']*350)
    
    # Create descriptor matrix
    X = np.column_stack([MW, logP, TPSA, HBD, HBA, nRotB, nArom])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    colors = {'Kinases': '#E74C3C', 'GPCRs': '#3498DB', 'Proteases': '#2ECC71'}
    for target in ['Kinases', 'GPCRs', 'Proteases']:
        mask = target_class == target
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[target], label=target, 
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    # Add loading vectors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    feature_names = ['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'nRotB', 'nArom']
    
    for i, feature in enumerate(feature_names):
        ax1.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
                 head_width=0.15, head_length=0.15, 
                 fc='black', ec='black', alpha=0.7, linewidth=2)
        ax1.text(loadings[i, 0]*3.5, loadings[i, 1]*3.5, feature,
                fontsize=11, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                   fontsize=12, fontweight='bold')
    ax1.set_title('A. PCA Biplot: Chemical Space Distribution', 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # ========== Panel B: Variance Explained ==========
    ax2 = axes[0, 1]
    
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    x_vals = np.arange(1, len(variance_ratio) + 1)
    
    ax2.bar(x_vals, variance_ratio * 100, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=1.5, label='Individual')
    ax2.plot(x_vals, cumulative_variance * 100, 'ro-', linewidth=2.5, 
             markersize=8, label='Cumulative', markerfacecolor='red', 
             markeredgecolor='darkred', markeredgewidth=1.5)
    
    ax2.axhline(y=90, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='90% Threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='95% Threshold')
    
    ax2.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Scree Plot: Variance Explained', 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='center right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(x_vals)
    
    # Add text annotation
    ax2.text(5, 75, f'PC1+PC2: {cumulative_variance[1]*100:.1f}%', 
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # ========== Panel C: Descriptor Distributions ==========
    ax3 = axes[1, 0]
    
    df_descriptors = pd.DataFrame({
        'MW': MW, 'logP': logP, 'TPSA': TPSA, 
        'Target': target_class
    })
    
    df_melted = df_descriptors.melt(id_vars=['Target'], 
                                     var_name='Descriptor', 
                                     value_name='Value')
    
    sns.violinplot(data=df_melted, x='Descriptor', y='Value', 
                   hue='Target', palette=colors, ax=ax3, 
                   split=False, inner='quartile', alpha=0.7)
    
    ax3.set_xlabel('Molecular Descriptor', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
    ax3.set_title('C. Descriptor Distributions by Target Class', 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.legend(title='Target Class', loc='upper right', 
               frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # ========== Panel D: Correlation Heatmap ==========
    ax4 = axes[1, 1]
    
    df_corr = pd.DataFrame({
        'MW': MW, 'logP': logP, 'TPSA': TPSA,
        'HBD': HBD, 'HBA': HBA, 'nRotB': nRotB, 'nArom': nArom
    })
    
    correlation_matrix = df_corr.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, ax=ax4,
                linewidths=1.5, cbar_kws={'label': 'Pearson Correlation'},
                vmin=-1, vmax=1)
    
    ax4.set_title('D. Descriptor Correlation Matrix', 
                  fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save
    plt.savefig('Figure_S1_Molecular_Descriptor_Space.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('Figure_S1_Molecular_Descriptor_Space.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("✓ Figure S1 saved successfully!")
    print(f"  - PC1+PC2 variance: {cumulative_variance[1]*100:.1f}%")
    print(f"  - Total molecules: {n_molecules}")
    print(f"  - Descriptor correlation range: [{correlation_matrix.min().min():.2f}, {correlation_matrix.max().max():.2f}]")
    
    return fig, X_pca, pca

# Execute
fig_s1, X_pca, pca_model = create_figure_s1()
plt.show()