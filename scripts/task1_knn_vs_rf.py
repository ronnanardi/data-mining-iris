"""
Tugas 1: Perbandingan K-Nearest Neighbor (KNN) vs Random Forest (RF)
Dataset: Iris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────
# 1. Load & Persiapan Dataset
# ─────────────────────────────────────────
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("=" * 60)
print("TUGAS 1: KNN vs RANDOM FOREST - DATASET IRIS")
print("=" * 60)
print(f"Jumlah sampel : {X.shape[0]}")
print(f"Jumlah fitur  : {X.shape[1]}")
print(f"Kelas         : {list(target_names)}")
print()

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data latih : {len(X_train)} sampel")
print(f"Data uji   : {len(X_test)} sampel\n")

# ─────────────────────────────────────────
# 2. Tuning K untuk KNN
# ─────────────────────────────────────────
print("Mencari nilai K terbaik untuk KNN...")
k_values = range(1, 21)
k_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy').mean()
    k_scores.append(score)

best_k = k_values[np.argmax(k_scores)]
print(f"K terbaik: {best_k} (akurasi CV: {max(k_scores):.4f})\n")

# ─────────────────────────────────────────
# 3. Training Model
# ─────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=best_k)
rf  = RandomForestClassifier(n_estimators=100, random_state=42)

knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
y_pred_rf  = rf.predict(X_test)

# ─────────────────────────────────────────
# 4. Evaluasi Metrik
# ─────────────────────────────────────────
def get_metrics(y_true, y_pred, name):
    return {
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, average='weighted'), 4),
        'Recall'   : round(recall_score(y_true, y_pred, average='weighted'), 4),
        'F1-Score' : round(f1_score(y_true, y_pred, average='weighted'), 4),
    }

metrics = pd.DataFrame([
    get_metrics(y_test, y_pred_knn, f'KNN (k={best_k})'),
    get_metrics(y_test, y_pred_rf,  'Random Forest'),
])

print("─" * 60)
print("PERBANDINGAN METRIK EVALUASI")
print("─" * 60)
print(metrics.to_string(index=False))
print()

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_knn = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
cv_rf  = cross_val_score(rf,  X_scaled, y, cv=cv, scoring='accuracy')

print(f"CV KNN            : {cv_knn.mean():.4f} ± {cv_knn.std():.4f}")
print(f"CV Random Forest  : {cv_rf.mean():.4f} ± {cv_rf.std():.4f}\n")

print("Classification Report - KNN:")
print(classification_report(y_test, y_pred_knn, target_names=target_names))
print("Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# ─────────────────────────────────────────
# 5. Visualisasi
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Tugas 1: Perbandingan KNN vs Random Forest — Dataset Iris',
             fontsize=14, fontweight='bold', y=1.02)

# (a) K vs Accuracy
axes[0, 0].plot(k_values, k_scores, 'o-', color='steelblue', linewidth=2)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', label=f'K optimal={best_k}')
axes[0, 0].set_xlabel('Nilai K')
axes[0, 0].set_ylabel('Akurasi (CV 10-Fold)')
axes[0, 0].set_title('Tuning Parameter K pada KNN')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) Confusion Matrix KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=target_names, yticklabels=target_names)
axes[0, 1].set_title(f'Confusion Matrix KNN (k={best_k})')
axes[0, 1].set_xlabel('Prediksi')
axes[0, 1].set_ylabel('Aktual')

# (c) Confusion Matrix RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[0, 2],
            xticklabels=target_names, yticklabels=target_names)
axes[0, 2].set_title('Confusion Matrix Random Forest')
axes[0, 2].set_xlabel('Prediksi')
axes[0, 2].set_ylabel('Aktual')

# (d) Bar Chart Metrik
metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metric_cols))
width = 0.35
bars1 = axes[1, 0].bar(x - width/2, metrics.iloc[0][metric_cols], width,
                        label=f'KNN (k={best_k})', color='steelblue', alpha=0.8)
bars2 = axes[1, 0].bar(x + width/2, metrics.iloc[1][metric_cols], width,
                        label='Random Forest', color='seagreen', alpha=0.8)
axes[1, 0].set_xlabel('Metrik')
axes[1, 0].set_ylabel('Nilai')
axes[1, 0].set_title('Perbandingan Metrik Evaluasi')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metric_cols)
axes[1, 0].set_ylim(0.8, 1.05)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar in bars1: axes[1, 0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2: axes[1, 0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# (e) CV Boxplot
axes[1, 1].boxplot([cv_knn, cv_rf], tick_labels=[f'KNN (k={best_k})', 'Random Forest'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
axes[1, 1].set_ylabel('Akurasi')
axes[1, 1].set_title('Distribusi Akurasi (10-Fold CV)')
axes[1, 1].grid(True, alpha=0.3)

# (f) Feature Importance RF
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
feat_imp.plot(kind='barh', ax=axes[1, 2], color='seagreen', alpha=0.8)
axes[1, 2].set_title('Feature Importance (Random Forest)')
axes[1, 2].set_xlabel('Importance')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results/task1_knn_vs_rf_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Grafik disimpan: results/task1_knn_vs_rf_comparison.png")
