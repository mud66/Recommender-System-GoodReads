import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
from umap import UMAP
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
sns.set_theme(style="white", palette="muted")

def load_books(pickle_path):
    return pd.read_pickle(pickle_path)

def preprocess_embeddings(books):
    embeddings = np.vstack(books['embeddings'].values)
    train_indices, test_indices = train_test_split(np.arange(embeddings.shape[0]), test_size=0.2, random_state=42)
    return embeddings, train_indices, test_indices


class DimensionalityReducer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50, random_state=42)
        self.umap = UMAP(n_components=3, random_state=42)
        
    def fit_transform(self, embeddings):
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        pca_embeddings = self.pca.fit_transform(scaled_embeddings)
        umap_embeddings = self.umap.fit_transform(pca_embeddings)
        return umap_embeddings
    
    def transform(self, embeddings):
        scaled_embeddings = self.scaler.transform(embeddings)
        pca_embeddings = self.pca.transform(scaled_embeddings)
        umap_embeddings = self.umap.transform(pca_embeddings)
        return umap_embeddings
    

def detect_outliers(embeddings):
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    return iso_forest.fit_predict(embeddings)


def apply_kernel_pca_in_batches(embeddings, n_components=45, kernel='rbf', batch_size=10000):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
    n_samples = embeddings.shape[0]
    transformed_embeddings = np.zeros((n_samples, n_components))
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Processing Batches"):
        end_idx = min(i + batch_size, n_samples)
        batch = embeddings[i:end_idx]
        transformed_batch = kpca.fit_transform(batch)
        transformed_embeddings[i:end_idx] = transformed_batch
    
    return transformed_embeddings


class Recommender:
    def __init__(self, n_clusters=11):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def fit_clusters(self, embeddings):
        return self.kmeans.fit_predict(embeddings)
    
    def predict_clusters(self, embeddings):
        return self.kmeans.predict(embeddings)
    
    def get_recommendations_by_cluster(self, book_id, books_df, embeddings, top_n=5):
        if book_id not in books_df['book_id'].values:
            print(f"Book ID {book_id} not found in the books DataFrame.")
            return pd.DataFrame(columns=['title', 'authors', 'book_id'])
        
        book_cluster = books_df.loc[books_df['book_id'] == book_id, 'cluster'].values[0]
        cluster_books = books_df[books_df['cluster'] == book_cluster]
        
        if len(cluster_books) <= top_n:
            return cluster_books[['title', 'authors', 'book_id']]
        
        book_id_to_index = {id_: idx for idx, id_ in enumerate(books_df['book_id'].values)}
        book_idx = book_id_to_index[book_id]
        cluster_book_indices = [book_id_to_index[id_] for id_ in cluster_books['book_id'].values]
        cluster_embedding_matrix = embeddings[cluster_book_indices]
        
        sim_scores = cosine_similarity(embeddings[book_idx].reshape(1, -1), cluster_embedding_matrix).flatten()
        cluster_book_ids = cluster_books['book_id'].values
        sim_scores_dict = {cluster_book_ids[i]: sim_scores[i] for i in range(len(cluster_book_ids)) if cluster_book_ids[i] != book_id}
        
        sorted_book_ids = sorted(sim_scores_dict, key=sim_scores_dict.get, reverse=True)
        top_book_ids = sorted_book_ids[:top_n]
        top_books = cluster_books[cluster_books['book_id'].isin(top_book_ids)]
        
        return top_books[['title', 'authors', 'book_id']]


if __name__ == "__main__":
    books = load_books('Pickle/books.pkl')
    embeddings, train_indices, test_indices = preprocess_embeddings(books)
    
    reducer = DimensionalityReducer()
    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]
    
    umap_train_embeddings = reducer.fit_transform(train_embeddings)
    umap_test_embeddings = reducer.transform(test_embeddings)
    
    outliers = detect_outliers(umap_train_embeddings)
    clean_train_embeddings = umap_train_embeddings[outliers == 1]
    
    kpca_train_embeddings = apply_kernel_pca_in_batches(clean_train_embeddings)
    kpca_train_embeddings_unclean = apply_kernel_pca_in_batches(umap_train_embeddings)
    kpca_test_embeddings = apply_kernel_pca_in_batches(umap_test_embeddings)
    
    recommender = Recommender()
    clean_train_clusters = recommender.fit_clusters(kpca_train_embeddings)
    train_clusters = np.full(len(kpca_train_embeddings_unclean), -1)
    train_clusters[outliers == 1] = clean_train_clusters
    test_clusters = recommender.predict_clusters(kpca_test_embeddings)
    
    train_books = books.iloc[train_indices].copy()
    train_books['cluster'] = train_clusters
    test_books = books.iloc[test_indices].copy()
    test_books['cluster'] = test_clusters
    
    all_books = pd.concat([train_books, test_books])
