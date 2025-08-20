import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, homogeneity_score, completeness_score, v_measure_score
import pandas as pd
import os

# Create output folder if it doesn't exist
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Cargar un subconjunto de categorías para hacerlo más manejable
categories = ['alt.atheism', 'comp.graphics', 'sci.space', 'talk.religion.misc']
dataset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
documents = dataset.data
true_labels = dataset.target

print(f"Documents loaded: {len(documents)} - Categories: {len(categories)}")
print(f"Categories: {categories}")
print(f"Document distribution by category:")
for i, cat in enumerate(categories):
    count = np.sum(true_labels == i)
    print(f"  {cat}: {count} documents")

# 2. Define function to vectorize with different weighting functions
def vectorize(docs, mode='tf', min_df=2, max_df=0.95):
    if mode == 'tf':
        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df)
    elif mode == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df, max_df=max_df)
    elif mode == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    else:
        raise ValueError("Unknown mode. Use 'tf', 'binary' or 'tfidf'.")
    X = vectorizer.fit_transform(docs)
    return X, vectorizer

# 3. Execute clustering and evaluate with multiple metrics
def cluster_and_evaluate(X, true_labels, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    
    # Multiple evaluation metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    silhouette = silhouette_score(X, pred_labels)
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    
    return {
        'pred_labels': pred_labels,
        'ari': ari,
        'silhouette': silhouette,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure
    }

# 4. Test the three representations with detailed analysis
modes = ['tf', 'binary', 'tfidf']
results = {}

print("\n" + "="*60)
print("COMPARISON OF WEIGHTING FUNCTIONS")
print("="*60)

for mode in modes:
    print(f"\n--- Analysis for {mode.upper()} ---")
    X, vectorizer = vectorize(documents, mode=mode)
    
    print(f"Matrix dimensions: {X.shape}")
    print(f"Number of unique terms: {X.shape[1]}")
    
    # Basic matrix statistics
    if mode == 'binary':
        sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        print(f"Sparsity (binary matrix): {sparsity:.4f}")
    else:
        mean_weight = X.mean()
        std_weight = np.sqrt(X.power(2).mean() - mean_weight**2)
        print(f"Mean weight: {mean_weight:.4f}")
        print(f"Standard deviation: {std_weight:.4f}")
    
    # Clustering evaluation
    eval_results = cluster_and_evaluate(X, true_labels, n_clusters=len(categories))
    results[mode] = eval_results
    
    print(f"ARI: {eval_results['ari']:.4f}")
    print(f"Silhouette: {eval_results['silhouette']:.4f}")
    print(f"Homogeneity: {eval_results['homogeneity']:.4f}")
    print(f"Completeness: {eval_results['completeness']:.4f}")
    print(f"V-measure: {eval_results['v_measure']:.4f}")

# 5. Detailed analysis of the IDF factor
print("\n" + "="*60)
print("ANALYSIS OF THE IDF FACTOR")
print("="*60)

# Get TF matrix for IDF analysis
X_tf, vectorizer_tf = vectorize(documents, mode='tf')
term_doc_freq = np.array((X_tf > 0).sum(axis=0)).flatten()
terms = np.array(vectorizer_tf.get_feature_names_out())
total_docs = X_tf.shape[0]

# Calculate IDF manually
idf_values = np.log(total_docs / (term_doc_freq + 1))

# Analysis of terms by document frequency
print(f"\nTotal documents: {total_docs}")
print(f"Total unique terms: {len(terms)}")

# Most and least frequent terms
top_terms_idx = np.argsort(-term_doc_freq)[:20]
bottom_terms_idx = np.argsort(term_doc_freq)[:20]

print(f"\nMost frequent terms (appear in more documents):")
for i, idx in enumerate(top_terms_idx):
    freq = term_doc_freq[idx]
    idf = idf_values[idx]
    print(f"  {i+1:2d}. '{terms[idx]}': {freq} docs, IDF={idf:.4f}")

print(f"\nLeast frequent terms (appear in fewer documents):")
for i, idx in enumerate(bottom_terms_idx):
    freq = term_doc_freq[idx]
    idf = idf_values[idx]
    print(f"  {i+1:2d}. '{terms[idx]}': {freq} docs, IDF={idf:.4f}")

# Analysis of terms by category
print(f"\n" + "="*60)
print("ANALYSIS OF TERMS BY CATEGORY")
print("="*60)

for cat_idx, category in enumerate(categories):
    print(f"\n--- Category: {category} ---")
    
    # Documents from this category
    cat_docs = X_tf[true_labels == cat_idx]
    cat_terms_freq = np.array((cat_docs > 0).sum(axis=0)).flatten()
    
    # Most characteristic terms for this category
    # Calculate the ratio of frequency in category vs total frequency
    ratio = np.where(term_doc_freq > 0, cat_terms_freq / term_doc_freq, 0)
    top_cat_terms_idx = np.argsort(-ratio)[:10]
    
    print(f"Most characteristic terms of '{category}':")
    for i, idx in enumerate(top_cat_terms_idx):
        cat_freq = cat_terms_freq[idx]
        total_freq = term_doc_freq[idx]
        ratio_val = ratio[idx]
        if total_freq > 0:
            print(f"  {i+1:2d}. '{terms[idx]}': {cat_freq}/{total_freq} docs ({ratio_val:.3f})")

# 6. Visualization of Results
print(f"\n" + "="*60)
print("VISUALIZATION OF RESULTS")
print("="*60)

# Create individual plots for the document

# 1. Comparison of metrics by mode
plt.figure(figsize=(12, 8))
metrics = ['ari', 'silhouette', 'homogeneity', 'completeness', 'v_measure']
metric_names = ['ARI', 'Silhouette', 'Homogeneity', 'Completeness', 'V-measure']

metric_data = []
for mode in modes:
    metric_data.append([results[mode][metric] for metric in metrics])

x = np.arange(len(metrics))
width = 0.25

for i, mode in enumerate(modes):
    plt.bar(x + i*width, metric_data[i], width, label=mode.upper())

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparison of Metrics by Weighting Function')
plt.xticks(x + width, metric_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_comparison_metrics.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. IDF Distribution
plt.figure(figsize=(10, 6))
plt.hist(idf_values, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
plt.xlabel('IDF Value')
plt.ylabel('Frequency')
plt.title('Distribution of IDF Values')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_idf_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Document frequency vs IDF
plt.figure(figsize=(10, 6))
plt.scatter(term_doc_freq, idf_values, alpha=0.6, s=10, color='coral')
plt.xlabel('Document Frequency')
plt.ylabel('IDF')
plt.title('Relationship: Document Frequency vs IDF')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_freq_vs_idf.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. ARI comparison by mode
plt.figure(figsize=(8, 6))
modes_plot = list(results.keys())
ari_values = [results[mode]['ari'] for mode in modes_plot]

bars = plt.bar(modes_plot, ari_values, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.ylabel('Adjusted Rand Index')
plt.title('ARI Comparison by Weighting Function')
plt.grid(True, alpha=0.3)

# Add values on bars
for bar, value in zip(bars, ari_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_ari_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Category-specific terms analysis (TF vs TF-IDF comparison)
plt.figure(figsize=(20, 14))
fig, axes = plt.subplots(2, 4, figsize=(20, 14))

# Get TF-IDF matrix for comparison
X_tfidf, vectorizer_tfidf = vectorize(documents, mode='tfidf')
tfidf_terms = np.array(vectorizer_tfidf.get_feature_names_out())

for cat_idx, category in enumerate(categories):
    # TF analysis
    cat_docs_tf = X_tf[true_labels == cat_idx]
    cat_terms_freq_tf = np.array((cat_docs_tf > 0).sum(axis=0)).flatten()
    ratio_tf = np.where(term_doc_freq > 0, cat_terms_freq_tf / term_doc_freq, 0)
    
    # TF-IDF analysis
    cat_docs_tfidf = X_tfidf[true_labels == cat_idx]
    # Calculate mean TF-IDF weights for terms in this category
    cat_terms_tfidf_mean = np.array(cat_docs_tfidf.mean(axis=0)).flatten()
    
    # Get top terms for both representations
    top_tf_terms_idx = np.argsort(-ratio_tf)[:8]
    top_tfidf_terms_idx = np.argsort(-cat_terms_tfidf_mean)[:8]
    
    # Filter terms with ratio > 0.1 for TF
    filtered_tf_indices = [idx for idx in top_tf_terms_idx if ratio_tf[idx] > 0.1]
    
    # Plot TF terms (top row)
    if filtered_tf_indices:
        terms_tf = [terms[idx] for idx in filtered_tf_indices[:5]]
        ratios_tf = [ratio_tf[idx] for idx in filtered_tf_indices[:5]]
        
        axes[0, cat_idx].barh(terms_tf, ratios_tf, color='lightblue')
        axes[0, cat_idx].set_title(f'TF - {category}', fontsize=12, pad=20)
        axes[0, cat_idx].set_xlabel('Ratio (cat_freq/total_freq)')
        axes[0, cat_idx].grid(True, alpha=0.3)
    
    # Plot TF-IDF terms (bottom row)
    terms_tfidf = [tfidf_terms[idx] for idx in top_tfidf_terms_idx[:5]]
    weights_tfidf = [cat_terms_tfidf_mean[idx] for idx in top_tfidf_terms_idx[:5]]
    
    axes[1, cat_idx].barh(terms_tfidf, weights_tfidf, color='lightcoral')
    axes[1, cat_idx].set_title(f'TF-IDF - {category}', fontsize=12, pad=20)
    axes[1, cat_idx].set_xlabel('Mean TF-IDF Weight')
    axes[1, cat_idx].grid(True, alpha=0.3)

# Add more space between rows
plt.subplots_adjust(hspace=0.4)
plt.savefig(os.path.join(output_folder, '1_category_terms_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# Print detailed comparison
print(f"\n" + "="*80)
print("DETAILED COMPARISON: TF vs TF-IDF CHARACTERISTIC TERMS")
print("="*80)

for cat_idx, category in enumerate(categories):
    print(f"\n--- Category: {category} ---")
    
    # TF analysis
    cat_docs_tf = X_tf[true_labels == cat_idx]
    cat_terms_freq_tf = np.array((cat_docs_tf > 0).sum(axis=0)).flatten()
    ratio_tf = np.where(term_doc_freq > 0, cat_terms_freq_tf / term_doc_freq, 0)
    top_tf_terms_idx = np.argsort(-ratio_tf)[:10]
    
    # TF-IDF analysis
    cat_docs_tfidf = X_tfidf[true_labels == cat_idx]
    cat_terms_tfidf_mean = np.array(cat_docs_tfidf.mean(axis=0)).flatten()
    top_tfidf_terms_idx = np.argsort(-cat_terms_tfidf_mean)[:10]
    
    print(f"\nTop 10 TF characteristic terms:")
    for i, idx in enumerate(top_tf_terms_idx):
        cat_freq = cat_terms_freq_tf[idx]
        total_freq = term_doc_freq[idx]
        ratio_val = ratio_tf[idx]
        if total_freq > 0:
            print(f"  {i+1:2d}. '{terms[idx]}': {cat_freq}/{total_freq} docs ({ratio_val:.3f})")
    
    print(f"\nTop 10 TF-IDF characteristic terms (by mean weight):")
    for i, idx in enumerate(top_tfidf_terms_idx):
        mean_weight = cat_terms_tfidf_mean[idx]
        print(f"  {i+1:2d}. '{tfidf_terms[idx]}': mean weight = {mean_weight:.4f}")

# 6. Correlation matrix between metrics
plt.figure(figsize=(8, 6))
metrics_df = pd.DataFrame({
    'TF': [results['tf'][m] for m in metrics],
    'BINARY': [results['binary'][m] for m in metrics],
    'TFIDF': [results['tfidf'][m] for m in metrics]
}, index=metric_names)

sns.heatmap(metrics_df.T, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Correlation Matrix between Metrics')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_metrics_correlation.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. Additional statistical analysis
print(f"\n" + "="*60)
print("ADDITIONAL STATISTICAL ANALYSIS")
print("="*60)

# IDF statistics
print(f"IDF Statistics:")
print(f"  Mean: {np.mean(idf_values):.4f}")
print(f"  Median: {np.median(idf_values):.4f}")
print(f"  Standard deviation: {np.std(idf_values):.4f}")
print(f"  Minimum: {np.min(idf_values):.4f}")
print(f"  Maximum: {np.max(idf_values):.4f}")

# Sparsity analysis by mode
print(f"\nSparsity analysis by mode:")
for mode in modes:
    X, _ = vectorize(documents, mode=mode)
    sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
    print(f"  {mode.upper()}: {sparsity:.4f}")

# Create simplified statistical analysis plots
plt.figure(figsize=(12, 8))
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. IDF Statistics (most important)
stats_names = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
stats_values = [np.mean(idf_values), np.median(idf_values), np.std(idf_values), 
                np.min(idf_values), np.max(idf_values)]

axes[0, 0].bar(stats_names, stats_values, color='skyblue')
axes[0, 0].set_title('IDF Statistics', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Value')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Sparsity by Mode (important for understanding matrix properties)
sparsity_values = []
for mode in modes:
    X, _ = vectorize(documents, mode=mode)
    sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
    sparsity_values.append(sparsity)

axes[0, 1].bar(modes, sparsity_values, color=['lightcoral', 'lightgreen', 'lightblue'])
axes[0, 1].set_title('Sparsity by Weighting Function', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Sparsity')
axes[0, 1].grid(True, alpha=0.3)

# 3. Mean Weights by Mode (key for understanding weight distributions)
matrix_stats = {}
for mode in modes:
    X, _ = vectorize(documents, mode=mode)
    if mode == 'binary':
        matrix_stats[mode] = {'mean': 0, 'std': 0}
    else:
        matrix_stats[mode] = {
            'mean': float(X.mean()),
            'std': float(np.sqrt(X.power(2).mean() - X.mean()**2))
        }

means = [matrix_stats[mode]['mean'] for mode in modes]
axes[1, 0].bar(modes, means, color=['lightcoral', 'lightgreen', 'lightblue'])
axes[1, 0].set_title('Mean Weights by Mode', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Mean Weight')
axes[1, 0].grid(True, alpha=0.3)

# 4. Document Frequency Distribution (most important for understanding corpus)
doc_freq_bins = [0, 1, 5, 10, 50, 100, 500, 1000, 2000, 3000]
doc_freq_counts = []
for i in range(len(doc_freq_bins)-1):
    count = np.sum((term_doc_freq >= doc_freq_bins[i]) & (term_doc_freq < doc_freq_bins[i+1]))
    doc_freq_counts.append(count)

bin_labels = [f'{doc_freq_bins[i]}-{doc_freq_bins[i+1]-1}' for i in range(len(doc_freq_bins)-1)]
axes[1, 1].bar(bin_labels, doc_freq_counts, color='gold')
axes[1, 1].set_title('Term Distribution by Document Frequency', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Number of Terms')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '1_statistical_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# 8. Conclusions and recommendations
print(f"\n" + "="*60)
print("CONCLUSIONS AND RECOMMENDATIONS")
print("="*60)

best_mode = max(results.keys(), key=lambda x: results[x]['ari'])
print(f"Best weighting function according to ARI: {best_mode.upper()}")

print(f"\nAnalysis of the IDF factor:")
print(f"- Terms with higher IDF are more specific and discriminative")
print(f"- Terms with lower IDF are more common and less discriminative")
print(f"- The IDF factor helps penalize very frequent terms that are not useful for clustering")

print(f"\nRecommendations:")
print(f"- For document clustering, TF-IDF is usually the best option")
print(f"- Analysis of terms by category helps understand which terms are characteristic")
print(f"- Visualization of IDF distribution helps understand corpus behavior")
