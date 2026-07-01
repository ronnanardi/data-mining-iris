"""
Tugas 3: Perbandingan RapidMiner vs Orange — Decision Tree pada Dataset Iris

Karena RapidMiner & Orange adalah GUI tools, script ini:
- Mereplikasi hasil Decision Tree kedua tools menggunakan sklearn
- Mensimulasikan perbedaan konfigurasi default masing-masing tools
- Menghasilkan visualisasi perbandingan yang setara dengan output tools tersebut
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("=" * 65)
print("TUGAS 3: PERBANDINGAN RAPIDMINER vs ORANGE — DECISION TREE")
print("=" * 65)
print("Dataset: Iris | Algoritma: Decision Tree\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 2. Konfigurasi Default Masing-Masing Tool
# ─────────────────────────────────────────
# RapidMiner default: criterion=gini, max_depth=20 (unlimited practical), min_samples_split=4
# Orange default    : criterion=gini, max_depth=100 (unlimited), min_samples_split=2 (lebih liberal)

configs = {
    'RapidMiner': {
        'criterion'       : 'gini',
        'max_depth'       : 10,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'random_state'    : 42,
        'color'           : '#2980B9',
        'description'     : 'Gini, min_split=4, min_leaf=2, depth≤10'
    },
    'Orange': {
        'criterion'       : 'gini',
        'max_depth'       : None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state'    : 42,
        'color'           : '#E67E22',
        'description'     : 'Gini, min_split=2, min_leaf=1, depth=∞'
    }
}

results = {}
models  = {}

for tool, cfg in configs.items():
    params = {k: v for k, v in cfg.items() if k not in ('color', 'description')}
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')

    results[tool] = {
        'accuracy'   : accuracy_score(y_test, y_pred),
        'precision'  : precision_score(y_test, y_pred, average='weighted'),
        'recall'     : recall_score(y_test, y_pred, average='weighted'),
        'f1'         : f1_score(y_test, y_pred, average='weighted'),
        'cv_mean'    : cv_scores.mean(),
        'cv_std'     : cv_scores.std(),
        'depth'      : dt.get_depth(),
        'n_leaves'   : dt.get_n_leaves(),
        'y_pred'     : y_pred,
        'cm'         : confusion_matrix(y_test, y_pred),
    }
    models[tool] = dt

    print(f"[{tool}] ({cfg['description']})")
    print(f"  Accuracy     : {results[tool]['accuracy']:.4f}")
    print(f"  Precision    : {results[tool]['precision']:.4f}")
    print(f"  Recall       : {results[tool]['recall']:.4f}")
    print(f"  F1-Score     : {results[tool]['f1']:.4f}")
    print(f"  CV (10-fold) : {results[tool]['cv_mean']:.4f} ± {results[tool]['cv_std']:.4f}")
    print(f"  Tree Depth   : {results[tool]['depth']}")
    print(f"  Jumlah Daun  : {results[tool]['n_leaves']}")
    print()

# ─────────────────────────────────────────
# 3. Tabel Perbandingan
# ─────────────────────────────────────────
compare_df = pd.DataFrame({
    'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score',
               'CV Mean', 'CV Std', 'Tree Depth', 'Jumlah Daun'],
    'RapidMiner': [
        f"{results['RapidMiner']['accuracy']:.4f}",
        f"{results['RapidMiner']['precision']:.4f}",
        f"{results['RapidMiner']['recall']:.4f}",
        f"{results['RapidMiner']['f1']:.4f}",
        f"{results['RapidMiner']['cv_mean']:.4f}",
        f"{results['RapidMiner']['cv_std']:.4f}",
        str(results['RapidMiner']['depth']),
        str(results['RapidMiner']['n_leaves']),
    ],
    'Orange': [
        f"{results['Orange']['accuracy']:.4f}",
        f"{results['Orange']['precision']:.4f}",
        f"{results['Orange']['recall']:.4f}",
        f"{results['Orange']['f1']:.4f}",
        f"{results['Orange']['cv_mean']:.4f}",
        f"{results['Orange']['cv_std']:.4f}",
        str(results['Orange']['depth']),
        str(results['Orange']['n_leaves']),
    ],
})

print("─" * 50)
print("TABEL PERBANDINGAN RAPIDMINER vs ORANGE")
print("─" * 50)
print(compare_df.to_string(index=False))

# Tree structure
for tool in configs:
    print(f"\nStruktur Decision Tree [{tool}]:")
    print(export_text(models[tool], feature_names=feature_names, max_depth=4))

# ─────────────────────────────────────────
# 4. Visualisasi
# ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Tugas 3: Perbandingan RapidMiner vs Orange — Decision Tree (Iris)',
             fontsize=14, fontweight='bold')

# (a) Decision Tree RapidMiner
ax1 = fig.add_subplot(3, 3, 1)
ax1.axis('off')
ax1.set_title('Decision Tree — RapidMiner', fontweight='bold', color='#2980B9')

ax2 = fig.add_subplot(3, 3, (1, 2))
plot_tree(models['RapidMiner'], feature_names=feature_names,
          class_names=target_names, filled=True, rounded=True,
          fontsize=7, ax=ax2, impurity=False, precision=3)
ax2.set_title('Decision Tree — RapidMiner (depth ≤ 10)', fontweight='bold', color='#2980B9')

# (b) Decision Tree Orange
ax3 = fig.add_subplot(3, 3, (3, 3))
plot_tree(models['Orange'], feature_names=feature_names,
          class_names=target_names, filled=True, rounded=True,
          fontsize=7, ax=ax3, impurity=False, precision=3)
ax3.set_title('Decision Tree — Orange (depth tidak dibatasi)', fontweight='bold', color='#E67E22')

# (c) Confusion Matrix RapidMiner
ax4 = fig.add_subplot(3, 3, 4)
import seaborn as sns
sns.heatmap(results['RapidMiner']['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax4)
ax4.set_title('Confusion Matrix — RapidMiner', color='#2980B9')
ax4.set_xlabel('Prediksi'); ax4.set_ylabel('Aktual')

# (d) Confusion Matrix Orange
ax5 = fig.add_subplot(3, 3, 5)
sns.heatmap(results['Orange']['cm'], annot=True, fmt='d', cmap='Oranges',
            xticklabels=target_names, yticklabels=target_names, ax=ax5)
ax5.set_title('Confusion Matrix — Orange', color='#E67E22')
ax5.set_xlabel('Prediksi'); ax5.set_ylabel('Aktual')

# (e) Bar chart metrik
ax6 = fig.add_subplot(3, 3, 6)
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rm_vals = [results['RapidMiner']['accuracy'], results['RapidMiner']['precision'],
           results['RapidMiner']['recall'],   results['RapidMiner']['f1']]
or_vals = [results['Orange']['accuracy'],     results['Orange']['precision'],
           results['Orange']['recall'],        results['Orange']['f1']]
x = np.arange(len(metric_names))
w = 0.35
b1 = ax6.bar(x - w/2, rm_vals, w, label='RapidMiner', color='#2980B9', alpha=0.85)
b2 = ax6.bar(x + w/2, or_vals, w, label='Orange',     color='#E67E22', alpha=0.85)
ax6.set_xticks(x); ax6.set_xticklabels(metric_names)
ax6.set_ylim(0.8, 1.05); ax6.set_ylabel('Nilai')
ax6.set_title('Perbandingan Metrik Evaluasi')
ax6.legend(); ax6.grid(True, alpha=0.3, axis='y')
for b in b1: ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                       f'{b.get_height():.3f}', ha='center', fontsize=8)
for b in b2: ax6.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                       f'{b.get_height():.3f}', ha='center', fontsize=8)

# (f) CV Distribution
ax7 = fig.add_subplot(3, 3, 7)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_rm = cross_val_score(models['RapidMiner'], X, y, cv=cv)
cv_or = cross_val_score(models['Orange'],     X, y, cv=cv)
ax7.boxplot([cv_rm, cv_or], labels=['RapidMiner', 'Orange'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red', linewidth=2))
ax7.set_ylabel('Akurasi'); ax7.set_title('Distribusi CV (10-Fold)')
ax7.grid(True, alpha=0.3)

# (g) Tree complexity comparison
ax8 = fig.add_subplot(3, 3, 8)
tools = ['RapidMiner', 'Orange']
depths  = [results[t]['depth']    for t in tools]
n_leaves= [results[t]['n_leaves'] for t in tools]
x = np.arange(2)
ax8.bar(x - 0.2, depths,   0.35, label='Tree Depth',  color='#8E44AD', alpha=0.8)
ax8.bar(x + 0.2, n_leaves, 0.35, label='Jumlah Daun', color='#27AE60', alpha=0.8)
ax8.set_xticks(x); ax8.set_xticklabels(tools)
ax8.set_title('Kompleksitas Model')
ax8.legend(); ax8.grid(True, alpha=0.3, axis='y')
for i, (d, l) in enumerate(zip(depths, n_leaves)):
    ax8.text(i-0.2, d+0.1,   str(d), ha='center', fontweight='bold')
    ax8.text(i+0.2, l+0.1,   str(l), ha='center', fontweight='bold')

# (h) Ringkasan teks
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
summary_text = (
    "RINGKASAN PERBANDINGAN\n"
    "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"RapidMiner\n"
    f"  Accuracy : {results['RapidMiner']['accuracy']:.4f}\n"
    f"  CV Mean  : {results['RapidMiner']['cv_mean']:.4f}\n"
    f"  Depth    : {results['RapidMiner']['depth']}\n"
    f"  Daun     : {results['RapidMiner']['n_leaves']}\n\n"
    f"Orange\n"
    f"  Accuracy : {results['Orange']['accuracy']:.4f}\n"
    f"  CV Mean  : {results['Orange']['cv_mean']:.4f}\n"
    f"  Depth    : {results['Orange']['depth']}\n"
    f"  Daun     : {results['Orange']['n_leaves']}\n\n"
    "Kesimpulan:\n"
    "Kedua tools menghasilkan akurasi\n"
    "tinggi. RapidMiner menghasilkan\n"
    "tree lebih ringkas (pruning aktif).\n"
    "Orange lebih fleksibel namun\n"
    "berpotensi overfitting.\n"
)
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('results/task3_decision_tree_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrafik disimpan: results/task3_decision_tree_comparison.png")
