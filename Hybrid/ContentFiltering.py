import pandas as pd
import numpy as np
import faiss
import hdbscan
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.ensemble import IsolationForest
from scipy.sparse import csr_matrix

class BookRecommender:
    def __init__(self, file_path):
        # Load and process data during initialization
        self.books, self.embedding_matrix = self.load_data(file_path)
        self.book_id_to_index = None
        self.cleaned_embeddings = None
        self.valid_indices = None
        self.scaled_embeddings = None
        self.pca_embeddings = None
        self.umap_embeddings = None
        self.full_clusters = None

    def load_data(self, file_path):
        books = pd.read_pickle(file_path)
        books = books.drop_duplicates(subset='title', keep='first')
        embedding_matrix = np.vstack(books['embeddings'].values)
        return books, embedding_matrix

    def standardize_embeddings(self, train_embeddings, test_embeddings):
        scaler = StandardScaler(with_mean=False)  # Avoid modifying sparsity
        return scaler.fit_transform(train_embeddings), scaler.transform(test_embeddings)

    def apply_pca(self, embeddings, n_components=50, batch_size=1000):
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        return ipca.fit_transform(embeddings)

    def apply_umap(self, embeddings, n_components=20, n_neighbors=200, min_dist=0.005):
        umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', n_jobs=-1)
        return umap_model.fit_transform(embeddings)

    def remove_outliers(self, embeddings, contamination=0.05, max_samples=0.2, random_state=42):
        iso_forest = IsolationForest(contamination=contamination, max_samples=max_samples, random_state=random_state, n_jobs=-1)
        outliers = iso_forest.fit_predict(embeddings)
        valid_indices = np.where(outliers == 1)[0]  # Indices of non-outliers
        return embeddings[valid_indices], valid_indices  # Return the cleaned embeddings and valid indices

    def convert_to_sparse(self, embeddings):
        return csr_matrix(embeddings)

    def perform_hdbscan_clustering(self, embeddings, min_cluster_size=500, min_samples=300):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
        clusters = clusterer.fit_predict(embeddings)
        return clusters

    def assign_clusters_to_books(self, books, indices, clusters, cluster_column="cluster"):
        books_copy = books.copy()
        books_copy[cluster_column] = -1
        books_copy.iloc[indices, books_copy.columns.get_loc(cluster_column)] = clusters
        return books_copy

    def get_recommendations_by_cluster(self, book_id, top_n=5):
        if self.book_id_to_index is None:
            raise ValueError("Book ID to index mapping is not initialized. Please run preprocessing steps first.")

        if book_id not in self.book_id_to_index:
            return []  # Return empty if book_id is not in book_id_to_index

        book_idx = self.book_id_to_index[book_id]
        input_book_title = self.books.loc[self.books['book_id'] == book_id, 'title'].values[0]

        # Create a FAISS index (using L2 or IP depending on embeddings)
        dimension = self.cleaned_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # Using L2 for non-normalized or IP for normalized
        index.add(self.cleaned_embeddings)  # Add all book embeddings to FAISS index

        # Search for nearest neighbors
        _, indices = index.search(np.array([self.cleaned_embeddings[book_idx]]), top_n + 1)  # +1 to exclude itself

        recommendations = []
        for idx in indices[0][1:]:  # Exclude the book itself
            recommended_book = self.books.iloc[idx]
            recommendations.append({"title": recommended_book["title"], "authors": recommended_book["authors"]})

        return recommendations

    def preprocess_data(self):
        # Remove outliers and get valid indices
        self.cleaned_embeddings, self.valid_indices = self.remove_outliers(self.embedding_matrix)

        # Filter the books DataFrame using the valid indices
        self.books = self.books.iloc[self.valid_indices]

        # Standardize embeddings
        self.scaled_embeddings = StandardScaler().fit_transform(self.cleaned_embeddings)

        # Dimensionality reduction using PCA
        self.pca_embeddings = self.apply_pca(self.scaled_embeddings)

        # UMAP reduction
        self.umap_embeddings = self.apply_umap(self.pca_embeddings)

        # Apply HDBSCAN clustering
        self.full_clusters = self.perform_hdbscan_clustering(self.umap_embeddings, min_cluster_size=10, min_samples=5)

        # Ensure that indices match the embeddings used to generate clusters
        indices = np.arange(self.umap_embeddings.shape[0])

        # Assign clusters to the books
        self.books = self.assign_clusters_to_books(self.books, indices, self.full_clusters, cluster_column="cluster")

        # Map book_id to index (make sure this map is consistent with the embeddings)
        self.book_id_to_index = {book_id: idx for idx, book_id in enumerate(self.books['book_id'])}
