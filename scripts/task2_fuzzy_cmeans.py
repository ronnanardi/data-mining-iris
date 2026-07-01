"""
Tugas 2: Fuzzy C-Means Clustering pada Dataset Iris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import skfuzzy as fuzz
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("=" * 60)
print("TUGAS 2: FUZZY C-MEANS CLUSTERING - DATASET IRIS")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 2. Fuzzy C-Means
# ─────────────────────────────────────────
n_clusters = 3  # sesuai jumlah kelas Iris
m = 2.0         # fuzziness parameter

# skfuzzy cmeans input: (features, n_samples)
X_T = X_scaled.T

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_T, c=n_clusters, m=m,
    error=1e-6, maxiter=1000, init=None
)

# Label hard cluster (highest membership)
cluster_labels = np.argmax(u, axis=0)

# ─────────────────────────────────────────
# 3. Evaluasi
# ─────────────────────────────────────────
ari   = adjusted_rand_score(y, cluster_labels)
silh  = silhouette_score(X_scaled, cluster_labels)
fpc_val = fpc  # Fuzzy Partition Coefficient (1 = crisp, 1/c = fully fuzzy)

print(f"\nParameter:")
print(f"  Jumlah cluster (c) : {n_clusters}")
print(f"  Fuzziness (m)      : {m}")
print(f"  Iterasi konvergen  : {p}")
print(f"\nMetrik Evaluasi:")
print(f"  Adjusted Rand Index : {ari:.4f}  (1.0 = sempurna)")
print(f"  Silhouette Score    : {silh:.4f}  (mendekati 1 = baik)")
print(f"  Fuzzy Part. Coeff.  : {fpc_val:.4f}  (mendekati 1 = crisp)")

# Tabel distribusi cluster vs label asli
print("\nDistribusi Cluster vs Label Asli:")
df = pd.DataFrame({'Cluster': cluster_labels, 'Label Asli': y,
                   'Nama': [target_names[i] for i in y]})
print(pd.crosstab(df['Cluster'], df['Nama']))

# ─────────────────────────────────────────
# 4. Membership degree sample
# ─────────────────────────────────────────
print("\nContoh Derajat Keanggotaan (10 sampel pertama):")
membership_df = pd.DataFrame(u.T[:10], columns=[f'Cluster {i}' for i in range(n_clusters)])
membership_df.index.name = 'Sampel'
print(membership_df.round(4).to_string())

# ─────────────────────────────────────────
# 5. Visualisasi
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Tugas 2: Fuzzy C-Means Clustering — Dataset Iris',
             fontsize=14, fontweight='bold')

colors_cluster = ['#E74C3C', '#3498DB', '#2ECC71']
colors_true    = ['#E67E22', '#9B59B6', '#1ABC9C']
markers        = ['o', 's', '^']

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
pair_labels = [(feature_names[a], feature_names[b]) for a, b in pairs]

# Plot 4 panel: hasil FCM cluster
for idx, (fi, fj) in enumerate(pairs[:4]):
    ax = axes[idx // 2, idx % 2] if idx < 4 else None
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    for c in range(n_clusters):
        mask = cluster_labels == c
        ax.scatter(X_scaled[mask, fi], X_scaled[mask, fj],
                   c=colors_cluster[c], marker=markers[c], s=60,
                   alpha=0.7, label=f'Cluster {c}')
    # Plot centroid
    ax.scatter(cntr[:, fi], cntr[:, fj], c='black', marker='*', s=200,
               zorder=5, label='Centroid')
    ax.set_xlabel(feature_names[fi], fontsize=9)
    ax.set_ylabel(feature_names[fj], fontsize=9)
    ax.set_title(f'{feature_names[fi][:10]} vs {feature_names[fj][:10]}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Panel 5: Convergence (Objective Function)
axes[1, 0].plot(jm, 'o-', color='steelblue', linewidth=2)
axes[1, 0].set_xlabel('Iterasi')
axes[1, 0].set_ylabel('Objective Function (Jm)')
axes[1, 0].set_title('Konvergensi Fuzzy C-Means')
axes[1, 0].grid(True, alpha=0.3)

# Panel 6: Membership heatmap (50 sampel)
sample_idx = np.concatenate([np.where(y==0)[0][:17],
                              np.where(y==1)[0][:17],
                              np.where(y==2)[0][:16]])
u_sample = u.T[sample_idx]
sns.heatmap(u_sample,
            xticklabels=[f'C{i}' for i in range(n_clusters)],
            cmap='YlOrRd', ax=axes[1, 2], cbar_kws={'label': 'Derajat Keanggotaan'})
axes[1, 2].set_title('Derajat Keanggotaan (50 sampel)')
axes[1, 2].set_xlabel('Cluster')
axes[1, 2].set_ylabel('Sampel')
# Garis pemisah kelas
for line in [17, 34]:
    axes[1, 2].axhline(y=line, color='blue', linestyle='--', linewidth=1.5)

plt.tight_layout()
plt.savefig('results/task2_fuzzy_cmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafik disimpan: results/task2_fuzzy_cmeans_clusters.png")
