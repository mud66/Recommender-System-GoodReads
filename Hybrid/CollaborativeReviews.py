import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GATRecommender:
    def __init__(self, seed=42, num_epochs=10, batch_size=32, lr=0.001, n_components=5, test_size=0.2):
        # Set random seed for reproducibility
        self.set_random_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_components = n_components
        self.test_size = test_size
    
    def set_random_seed(self, seed_value):
        """Set the random seed for reproducibility."""
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_data(self):
        reviews = pd.read_pickle('../Pickle/reviews.pkl')
        books = pd.read_pickle('../Pickle/books.pkl')
        read = pd.read_pickle('../Pickle/read.pkl')
        user_genres = pd.read_pickle('../Pickle/user_most_common_genres.pkl')
        review_embeddings = pd.read_pickle('../Pickle/review_embeddings.pkl')
        return reviews, books, read, user_genres, review_embeddings
    
    def initialize_id_mappings(self, combined_data):
        unique_user_ids = set(combined_data['user_id'])
        unique_book_ids = set(combined_data['book_id'])

        user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        book_id_to_index = {book_id: idx for idx, book_id in enumerate(unique_book_ids)}

        return user_id_to_index, book_id_to_index
    
    def filter_and_split_data(self, ratings_data, user_genres):
        book_user_counts = ratings_data['book_id'].value_counts()
        eligible_books = book_user_counts[book_user_counts > 5].index  
        ratings_data = ratings_data[ratings_data['book_id'].isin(eligible_books)]

        user_book_counts = ratings_data['user_id'].value_counts()
        eligible_users = user_book_counts[user_book_counts > 5].index  
        ratings_data = ratings_data[ratings_data['user_id'].isin(eligible_users)]    

        eligible_users_in_genres = user_genres['user_id'].isin(eligible_users)
        user_genres = user_genres[eligible_users_in_genres]

        filtered_data = ratings_data.merge(user_genres[['user_id', 'most_common_genres']], on='user_id', how='inner')

        train_dfs = []
        test_dfs = []

        user_data_valid = filtered_data.groupby('user_id').filter(lambda x: len(x) > 5)

        for user_id, user_data in user_data_valid.groupby('user_id'):
            books = user_data['book_id'].unique()
            train_books, test_books = train_test_split(books, test_size=self.test_size, random_state=42)
            
            user_train_data = user_data[user_data['book_id'].isin(train_books)]
            user_test_data = user_data[user_data['book_id'].isin(test_books)]
            
            train_dfs.append(user_train_data)
            test_dfs.append(user_test_data)
        
        train_data = pd.concat(train_dfs)
        test_data = pd.concat(test_dfs)

        return train_data, test_data, user_genres, filtered_data
    
    def normalize_ratings(self, train_data, test_data):
        min_rating = train_data['rating'].min()
        
        if min_rating < 0:
            train_data['rating'] = train_data['rating'] - min_rating
            test_data['rating'] = test_data['rating'] - min_rating

        train_data['rating'] = np.log1p(train_data['rating'])
        test_data['rating'] = np.log1p(test_data['rating'])

        return train_data, test_data, min_rating
    
    def balance_ratings(self, train_data):
        rating_counts = train_data['rating'].value_counts()
        target_count = 10000
        balanced_data = []

        for rating, count in rating_counts.items():
            rating_data = train_data[train_data['rating'] == rating]
            
            if count > target_count:
                rating_data = resample(rating_data, replace=False, n_samples=target_count, random_state=42)
            elif count < target_count:
                rating_data = resample(rating_data, replace=True, n_samples=target_count, random_state=42)

            balanced_data.append(rating_data)

        balanced_train_data = pd.concat(balanced_data, axis=0)
        balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

        return balanced_train_data
    
    def prepare_edge_index_ratings_attributes(self, df, user_id_to_index, book_id_to_index):
        user_indices = df['user_id'].map(user_id_to_index).dropna().astype(int).values
        book_indices = df['book_id'].map(book_id_to_index).dropna().astype(int).values

        valid_mask = (user_indices >= 0) & (book_indices >= 0)
        user_indices = user_indices[valid_mask]
        book_indices = book_indices[valid_mask]

        edge_index = torch.tensor([user_indices, book_indices], dtype=torch.long)

        ratings_tensor = torch.tensor(df.loc[valid_mask, 'rating'].values, dtype=torch.float32).view(-1, 1)
        confidence_tensor = torch.tensor(df.loc[valid_mask, 'Confidence Score'].values, dtype=torch.float32).view(-1, 1)

        embeddings_np = np.stack(df.loc[valid_mask, 'embeddings'].values)
        embeddings_tensor = torch.from_numpy(embeddings_np).float()

        edge_attr = torch.cat([ratings_tensor, confidence_tensor, embeddings_tensor], dim=1)

        return edge_index, edge_attr
    
    def align_user_and_book_features(self, filtered_data, user_id_to_index, book_id_to_index):
        unique_book_genres = sorted(set(filtered_data['filtered_genres'].str.split(',').explode()))
        book_genre_dict = {genre: idx for idx, genre in enumerate(unique_book_genres)}

        user_genre_features = {}
        for user_id, group in filtered_data.groupby('user_id'):
            genres = group['most_common_genres'].iloc[0]
            genre_vector = np.zeros(len(book_genre_dict))
            for genre in genres:
                if genre in book_genre_dict:
                    genre_vector[book_genre_dict[genre]] = 1
            user_genre_features[user_id_to_index[user_id]] = torch.tensor(genre_vector, dtype=torch.float32)

        book_genre_features = {}
        for book_id, group in filtered_data.groupby('book_id'):
            genres = group['filtered_genres'].iloc[0].split(',')
            genre_vector = np.zeros(len(book_genre_dict))
            for genre in genres:
                if genre in book_genre_dict:
                    genre_vector[book_genre_dict[genre]] = 1
            book_genre_features[book_id_to_index[book_id]] = torch.tensor(genre_vector, dtype=torch.float32)

        return user_genre_features, book_genre_features
    
    def apply_pca_on_features(self, user_genre_features, book_genre_features):
        all_user_features = torch.stack(list(user_genre_features.values()))
        all_book_features = torch.stack(list(book_genre_features.values()))

        all_features = torch.cat([all_user_features, all_book_features], dim=0)

        pca = PCA(n_components=self.n_components)
        reduced_features = pca.fit_transform(all_features)

        reduced_user_features = reduced_features[:len(user_genre_features)]
        reduced_book_features = reduced_features[len(user_genre_features):]

        updated_user_genre_features = {key: torch.tensor(val) for key, val in zip(user_genre_features.keys(), reduced_user_features)}
        updated_book_genre_features = {key: torch.tensor(val) for key, val in zip(book_genre_features.keys(), reduced_book_features)}

        return updated_user_genre_features, updated_book_genre_features
    
    def prepare_data_objects(self, train_data, test_data, user_genre_features, book_genre_features, user_id_to_index, book_id_to_index):
        train_edge_index, train_edge_attr = self.prepare_edge_index_ratings_attributes(
            train_data, user_id_to_index, book_id_to_index
        )
        test_edge_index, test_edge_attr = self.prepare_edge_index_ratings_attributes(
            test_data, user_id_to_index, book_id_to_index
        )

        user_embeddings = torch.from_numpy(np.stack(list(user_genre_features.values()))).float()
        book_embeddings = torch.from_numpy(np.stack(list(book_genre_features.values()))).float()

        node_embeddings = torch.cat([user_embeddings, book_embeddings], dim=0)

        train_data_obj = Data(
            x=node_embeddings,
            edge_index=train_edge_index,
            edge_attr=train_edge_attr
        )
        
        test_data_obj = Data(
            x=node_embeddings,
            edge_index=test_edge_index,
            edge_attr=test_edge_attr
        )

        return train_data_obj, test_data_obj
    
    class GATModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
            super(GATRecommender.GATModel, self).__init__()
            self.convs = nn.ModuleList([
                GATv2Conv(in_channels, hidden_channels, heads=8, dropout=dropout)
            ])
            self.convs.extend([
                GATv2Conv(hidden_channels * 8, hidden_channels, heads=8, dropout=dropout)
            ] * (num_layers - 1))
            self.lin = nn.Linear(hidden_channels * 8, out_channels)
            self.dropout = dropout

        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            for conv in self.convs:
                x = F.relu(conv(x, edge_index, edge_attr))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin(x)
            return x
    
    def train_and_evaluate(self):
        reviews, books, read, user_genres, review_embeddings = self.load_data()
        train_data, test_data, user_genres, filtered_data = self.filter_and_split_data(reviews, user_genres)
        train_data, test_data, min_rating = self.normalize_ratings(train_data, test_data)
        balanced_train_data = self.balance_ratings(train_data)

        user_id_to_index, book_id_to_index = self.initialize_id_mappings(filtered_data)
        user_genre_features, book_genre_features = self.align_user_and_book_features(filtered_data, user_id_to_index, book_id_to_index)

        updated_user_genre_features, updated_book_genre_features = self.apply_pca_on_features(user_genre_features, book_genre_features)

        train_data_obj, test_data_obj = self.prepare_data_objects(
            balanced_train_data, test_data, updated_user_genre_features, updated_book_genre_features, user_id_to_index, book_id_to_index
        )

        model = self.GATModel(in_channels=train_data_obj.x.shape[1], hidden_channels=64, out_channels=1)
        model = model.to(self.device)
        train_data_obj = train_data_obj.to(self.device)
        test_data_obj = test_data_obj.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_data_obj)
            loss = criterion(out.view(-1), train_data_obj.edge_attr[:, 0])  # Assuming rating is the first edge attribute
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                test_out = model(test_data_obj)
                test_loss = mean_squared_error(test_out.cpu().numpy().flatten(), test_data_obj.edge_attr[:, 0].cpu().numpy())
            
            print(f'Epoch: {epoch+1}, Train Loss: {loss.item()}, Test Loss: {test_loss}')

        self.evaluate_model(model, test_data_obj)

    def evaluate_model(self, model, test_data_obj):
        model.eval()
        with torch.no_grad():
            out = model(test_data_obj)
            predicted_ratings = out.cpu().numpy().flatten()
            true_ratings = test_data_obj.edge_attr[:, 0].cpu().numpy()
            
            mse = mean_squared_error(true_ratings, predicted_ratings)
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            print(f'Mean Squared Error: {mse}')
            print(f'Mean Absolute Error: {mae}')
