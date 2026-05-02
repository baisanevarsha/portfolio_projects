# Pokémon Stat Segmentation — Unsupervised Clustering Analysis

> **Context:** Academic project applying data science fundamentals to a real-world multi-attribute dataset.  
> **Skills demonstrated:** Data cleaning, EDA, dimensionality reduction (PCA), clustering (K-Means, GMM, Hierarchical), model evaluation, business interpretation.

---

## Problem Statement

Given a dataset of 800+ Pokémon with attributes across HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Type, and Generation — can we identify **natural performance archetypes** through unsupervised segmentation? And which clustering method best captures those groupings?

---

## Dataset

- **Source:** [Pokémon Dataset](https://www.kaggle.com/datasets/abcsds/pokemon) — 800 Pokémon, 13 features
- **Key features used:** `Total`, `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`, `Generation`
- **Challenge:** ~400 Pokémon had missing `Type 2` values requiring principled imputation

---

## Step 1 — Data Cleaning: Similarity-Based Imputation

Rather than filling missing `Type 2` values with the column mode (which would distort the distribution), a **stat-similarity approach** was used:

**Logic:**
1. For each Pokémon missing `Type 2`, find candidates with the **same `Type 1` and `Generation`** that have a known `Type 2`
2. Compute **Manhattan distance** across all 6 battle stats
3. Assign the `Type 2` of the closest statistical neighbor
4. Fallback: if no candidates exist in the same generation, use the most common `Type 2` for that `Type 1` globally

```python
def find_closest_type2(row, df):
    if pd.notna(row["Type 2"]) or row["Legendary"]:
        return row["Type 2"]

    candidates = df[
        (df["Type 1"] == row["Type 1"]) &
        (df["Generation"] == row["Generation"]) &
        (df["Type 2"].notna())
    ]

    if candidates.empty:
        most_common = df[df["Type 1"] == row["Type 1"]]["Type 2"].mode()
        return most_common[0] if not most_common.empty else None

    candidates["similarity"] = candidates[stat_columns].apply(
        lambda x: sum(abs(x - row[stat_columns])), axis=1
    )
    return df.loc[candidates["similarity"].idxmin(), "Type 2"]
```

> **Why this matters:** Naive imputation (mean/mode) would introduce systematic bias. Stat-similarity preserves the data's natural structure, which is critical for downstream clustering.

---

## Step 2 — Exploratory Data Analysis

Computed a full statistical profile across all numerical features:

| Metric | Purpose |
|---|---|
| Mean, Median, Mode | Central tendency |
| Variance, Std Dev | Spread of stats across Pokémon |
| Coefficient of Variation | Relative variability (normalized by mean) |
| Skewness & Kurtosis | Distribution shape — most stats are right-skewed |
| IQR & Outlier Detection | Identifying legendaries as statistical outliers |

**Key findings:**
- `Total` stat has right-skew — a small group of legendaries pulls the distribution
- `Attack` and `Sp. Atk` show moderate positive correlation (~0.5)
- `Speed` is the most independent stat — weakest correlations with others
- Legendaries clearly emerge as outliers in the `Total` distribution via IQR analysis

**Visualizations produced:** Histograms, boxplots, scatter plots (Attack vs Speed), correlation heatmaps, and average stat comparisons by Type.

---

## Step 3 — Dimensionality Reduction (PCA)

Before clustering, PCA was applied to reduce the 6 stat dimensions into **principal components**, capturing maximum variance with fewer features. The first 3 components were used for clustering and 3D visualization.

> This reduces noise and multicollinearity (e.g., Total is a linear combination of the other stats) before passing data to clustering algorithms.

---

## Step 4 — Clustering: Three Methods Compared

All three methods were run with **k=6 clusters** on the PCA-transformed data.

### K-Means

Partitions Pokémon into 6 clusters by minimizing within-cluster variance. Assumes spherical, equally-sized clusters.

### Gaussian Mixture Model (GMM)

Probabilistic soft-assignment model — each Pokémon gets a probability of belonging to each cluster. Allows elliptical cluster shapes.

### Hierarchical Clustering (Agglomerative, Ward linkage)

Builds clusters bottom-up by merging the pair that minimizes within-cluster variance at each step. Does not require pre-specifying k (though k=6 was used here for comparison).

---

## Step 5 — Model Evaluation

Three metrics were used to evaluate clustering quality objectively:

| Metric | K-Means | GMM | Hierarchical |
|---|---|---|---|
| **Silhouette Score** ↑ | 0.2468 | 0.2255 | **0.2662** |
| **Davies-Bouldin Index** ↓ | **1.1834** | 1.7323 | 1.1893 |
| **Calinski-Harabasz Index** ↑ | **429.20** | 316.10 | 376.99 |

- **Silhouette Score:** Measures how similar each point is to its own cluster vs. other clusters (range: -1 to 1)
- **Davies-Bouldin Index:** Ratio of within-cluster scatter to between-cluster separation — lower is better
- **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster dispersion — higher is better

### Recommendation: K-Means

Despite Hierarchical Clustering edging out on silhouette score, **K-Means is the recommended method** for this use case:
- Strongest Davies-Bouldin and Calinski-Harabasz scores
- Significantly faster and more scalable
- Clusters are compact and interpretable
- GMM's Gaussian assumption did not improve results, suggesting Pokémon stat distributions are not strongly Gaussian

---

## Step 6 — Cluster Interpretation & Business Application

The 6 clusters map to interpretable Pokémon archetypes:

| Cluster | Archetype | Stat Profile |
|---|---|---|
| 1 | **Speedsters** | High Speed, low bulk |
| 2 | **Bulky Tanks** | High HP & Defense, low Speed |
| 3 | **Glass Cannons** | High Attack, low Defense |
| 4 | **All-Rounders** | Balanced across all stats |
| 5 | **Legendaries** | Outlier-tier Total stats |
| 6 | **Basic Pokémon** | Low overall stats, early-game |

**Applications of this segmentation:**

- **Game Balance:** Developers can identify underperforming archetypes (e.g., Basics) for stat buffs, or cap Legendaries in competitive formats
- **Team Strategy:** Trainers can build synergistic teams by combining archetypes (e.g., Tank + Speedster + All-Rounder)
- **Marketing:** Target merchandise and promotions by archetype appeal — Legendaries for premium collectibles, Speedsters for action-oriented audiences

---

## Tech Stack

```
Python 3.x
pandas · numpy · matplotlib · seaborn
scikit-learn (KMeans, GaussianMixture, AgglomerativeClustering, PCA)
```

---

## Files

| File | Description |
|---|---|
| `Varsha_Baisane_DSM_A1.ipynb` | Full analysis notebook |
| `pokemon_data_filled_by_stats.csv` | Cleaned dataset post-imputation |
| `pca_transformed_data.csv` | PCA-reduced feature matrix used for clustering |

---

*Academic project — Data Science & Management, Assignment 1*
