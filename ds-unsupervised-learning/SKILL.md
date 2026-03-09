---
name: ds-unsupervised-learning
description: Guides clustering, dimensionality reduction, and recommendation system tasks. Use this skill whenever the user wants to find groups in data, segment customers, cluster documents, reduce dimensions for visualization or denoising, apply PCA/t-SNE/UMAP, build a recommendation engine, detect anomalies using unsupervised methods, or evaluate clustering quality. Trigger when the user says "cluster this data", "segment my customers", "find groups", "reduce dimensions", "visualize high-dimensional data", "build a recommender", "recommend items", "what's the right number of clusters", "anomaly detection", or drops unlabeled data and asks what structure exists. Also use when the user is unsure whether to use K-Means, DBSCAN, or hierarchical clustering, or needs help interpreting silhouette scores or elbow plots.
---

# Unsupervised Learning Skill

This skill covers three major unsupervised learning areas: clustering (finding groups), dimensionality reduction (compressing features), and recommendation systems (predicting preferences). The common thread is that there's no target variable — you're discovering structure that the data itself reveals.

## Part 1: Clustering

### Choosing the Right Algorithm

The choice depends on what kind of clusters you expect and how your data behaves.

**K-Means** — The default starting point.
- Finds spherical, equally-sized clusters
- Fast, scales to large datasets
- You must specify k (number of clusters) in advance
- Sensitive to outliers and feature scales — always standardize first
- Use when: you have a rough idea of how many groups exist, clusters are roughly round

**Hierarchical (Agglomerative)** — When you want to see cluster structure at multiple levels.
- Produces a dendrogram showing nested cluster relationships
- No need to pre-specify k — cut the dendrogram at the desired level
- Slow on large datasets (O(n²) memory)
- Use when: you want to explore how clusters nest, dataset is <10k rows, or domain experts want to see the tree

**DBSCAN** — When clusters have irregular shapes or you have noise.
- Finds clusters of arbitrary shape
- Automatically identifies noise points (outliers)
- No need to specify number of clusters
- Two parameters: eps (neighborhood radius) and min_samples (density threshold)
- Use when: clusters aren't spherical, you expect outliers, or you don't know how many groups exist

### The Clustering Workflow

#### Step 1: Prepare the Data

Clustering is distance-based, so feature preparation is critical.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Select features for clustering (exclude IDs, targets, etc.)
features = df[['feature_1', 'feature_2', 'feature_3']].copy()

# Handle missing values
features = features.dropna()  # or impute — but be thoughtful about it

# Standardize — this is NOT optional for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

If you skip scaling, features with larger magnitudes will dominate the distance calculations and the clusters will reflect that feature's range rather than meaningful structure.

#### Step 2: Determine the Number of Clusters

For K-Means, use multiple methods and look for agreement:

**Elbow method:**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Silhouette analysis** (more informative than the elbow):
```python
from sklearn.metrics import silhouette_score, silhouette_samples

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  k={k}: silhouette = {sil:.3f}")
```

Silhouette scores range from -1 to 1. Above 0.5 is good structure, 0.25–0.5 is reasonable, below 0.25 suggests weak or overlapping clusters. But always visualize — numbers don't tell the whole story.

#### Step 3: Fit and Interpret

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Profile each cluster
cluster_profiles = df.groupby('cluster').agg(['mean', 'median', 'count'])
print(cluster_profiles)
```

The cluster labels (0, 1, 2, ...) are arbitrary. Give them meaningful names based on the profiles. "Cluster 0" means nothing; "High-Value Frequent Buyers" means everything.

#### Step 4: Visualize

For 2D visualization of high-dimensional clusters, use PCA or t-SNE:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6, s=15)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Clusters in PCA Space')
plt.colorbar(label='Cluster')
plt.show()
```

### DBSCAN Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Use k-distance graph to find a good eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel('5th nearest neighbor distance')
plt.title('K-Distance Graph (look for the "elbow")')
plt.show()

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X_scaled)

n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
n_noise = (df['cluster'] == -1).sum()
print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
```

## Part 2: Dimensionality Reduction

### Algorithm Selection

| Method | Type | Best For | Preserves |
|---|---|---|---|
| PCA | Linear | General compression, denoising | Global variance structure |
| t-SNE | Nonlinear | 2D/3D visualization | Local neighborhood structure |
| UMAP | Nonlinear | Visualization + some global structure | Local + some global structure |
| Autoencoder | Deep nonlinear | Complex compression, anomaly detection | Learned representations |

### PCA (Principal Component Analysis)

The go-to method for linear dimensionality reduction.

```python
from sklearn.decomposition import PCA

# Determine how many components to keep
pca_full = PCA().fit(X_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumulative_var, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.show()

n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")

# Apply PCA
pca = PCA(n_components=n_components_95)
X_pca = pca.fit_transform(X_scaled)
```

### t-SNE and UMAP for Visualization

```python
from sklearn.manifold import TSNE

# t-SNE — perplexity controls neighborhood size
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
plt.title('t-SNE Visualization')
plt.show()
```

**Critical t-SNE caveats:**
- Distances between clusters in t-SNE plots are NOT meaningful — only within-cluster structure is reliable
- Different perplexity values can produce very different-looking plots. Try multiple (5, 30, 50).
- t-SNE is stochastic — set random_state for reproducibility
- Cannot transform new data — it's a one-shot embedding

UMAP is often preferred over t-SNE because it's faster, preserves more global structure, and can transform new data:

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)
```

## Part 3: Recommendation Systems

### Three Approaches

**Rank-Based** — Recommend the most popular items. No personalization, but a solid cold-start fallback.

**Content-Based** — Recommend items similar to what the user has liked, based on item features.

```python
from sklearn.metrics.pairwise import cosine_similarity

# Items represented as TF-IDF vectors of their descriptions
item_similarity = cosine_similarity(item_features_tfidf)

def get_recommendations(item_id, n=5):
    idx = item_to_index[item_id]
    sim_scores = list(enumerate(item_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in sim_scores[1:n+1]]
    return items.iloc[top_indices]
```

**Collaborative Filtering** — Recommend based on what similar users liked. The most powerful approach when you have enough interaction data.

**Matrix Factorization (SVD):**
```python
from scipy.sparse.linalg import svds
import numpy as np

# Create user-item matrix (rows=users, cols=items, values=ratings)
R = user_item_matrix.values
user_mean = np.mean(R, axis=1).reshape(-1, 1)
R_centered = R - user_mean

# Decompose
U, sigma, Vt = svds(R_centered, k=50)
sigma = np.diag(sigma)

# Reconstruct predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_mean
predictions_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
```

### Choosing an Approach

- New system, no user data → Rank-based
- Have item metadata but sparse interactions → Content-based
- Rich interaction history → Collaborative filtering
- In practice, hybrid systems (combining content-based and collaborative) work best

## Evaluating Unsupervised Models

Evaluating unsupervised models is harder than supervised because there's no ground truth. Use multiple lenses:

**Internal metrics (no labels needed):**
- Silhouette Score: -1 to 1, higher = better separation
- Davies-Bouldin Index: lower = better (measures cluster compactness vs separation)
- Calinski-Harabasz Index: higher = better (ratio of between-cluster to within-cluster variance)

**External metrics (when labels exist for validation):**
- Adjusted Rand Index (ARI): -1 to 1, measures agreement with true labels adjusted for chance
- Normalized Mutual Information (NMI): 0 to 1, information-theoretic measure of cluster-label agreement

**Visual evaluation:**
- Cluster profiles (summary statistics per cluster)
- t-SNE/UMAP scatter plots colored by cluster
- Silhouette plots per cluster (shows internal consistency)

**The most important question:** Do the clusters make sense to a domain expert? A technically optimal clustering that doesn't map to meaningful business categories is useless. Always present cluster profiles in plain language.

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

print(f"Silhouette Score:       {silhouette_score(X_scaled, labels):.3f}")
print(f"Davies-Bouldin Index:   {davies_bouldin_score(X_scaled, labels):.3f}")
print(f"Calinski-Harabasz:      {calinski_harabasz_score(X_scaled, labels):.1f}")
```

## Output Format

For any unsupervised task, always deliver:
1. Data preparation steps with justification
2. Algorithm choice rationale
3. Parameter selection process (elbow plots, silhouette analysis, etc.)
4. Results with visualizations
5. Interpretation in domain-relevant terms
6. Evaluation metrics with context on what the numbers mean
7. Actionable next steps
