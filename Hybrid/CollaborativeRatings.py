import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, GridSearchCV
from collections import defaultdict
from math import log2
from sklearn.utils import resample
from custom_svd import RandomizedSVD

class BookRecommender:
    def __init__(self, interactions_file_path):
        # Load interactions data
        self.interactions = pd.read_pickle(interactions_file_path)
        self.model = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.best_randomized_svd = None

        self.train_df, self.test_df = None, None
        self.reader = None
        self.train_data = None
        self.test_data = None
        self.trainset = None
        self.testset = None

    def preprocess_data(self):
        # Filter columns and prepare for processing
        self.interactions = self.interactions[['user_id', 'book_id', 'rating', 'is_read']]
        
        # Separate lower ratings by their respective values
        ratings_0 = self.interactions[self.interactions['rating'] == 0]
        ratings_1 = self.interactions[self.interactions['rating'] == 1]
        ratings_2 = self.interactions[self.interactions['rating'] == 2]
        higher_ratings = self.interactions[self.interactions['rating'] > 3]

        # Calculate the number of samples needed for each lower rating
        higher_count = len(higher_ratings)

        # Define sampling factors
        factor_0 = 0.2
        factor_1 = 0.2
        factor_2 = 0.2

        # Ensure n_samples are within valid limits
        n_samples_0 = min(int(higher_count * factor_0), len(ratings_0))
        n_samples_1 = min(int(higher_count * factor_1), len(ratings_1))
        n_samples_2 = min(int(higher_count * factor_2), len(ratings_2))

        # Apply oversampling
        ratings_0_oversampled = resample(ratings_0, replace=True, n_samples=n_samples_0, random_state=42)
        ratings_1_oversampled = resample(ratings_1, replace=True, n_samples=n_samples_1, random_state=42)
        ratings_2_oversampled = resample(ratings_2, replace=True, n_samples=n_samples_2, random_state=42)

        # Combine the datasets
        self.interactions = pd.concat([higher_ratings, ratings_0_oversampled, ratings_1_oversampled, ratings_2_oversampled])

        # Split data into train and test
        self.train_df, self.test_df = train_test_split(self.interactions, test_size=0.2, random_state=42)

    def train_model(self):
        # Calculate the global mean rating
        self.global_mean = self.train_df['rating'].mean()

        # Calculate user bias with regularization
        lambda_reg = 10
        user_sum_ratings = self.train_df.groupby('user_id')['rating'].sum()
        user_count_ratings = self.train_df.groupby('user_id')['rating'].count()
        self.user_bias = (user_sum_ratings - user_count_ratings * self.global_mean) / (user_count_ratings + lambda_reg)

        # Map user bias back to the original dataframe
        self.train_df['user_bias'] = self.train_df['user_id'].map(self.user_bias)

        # Calculate item bias with regularization
        item_sum_ratings = self.train_df.groupby('book_id')['rating'].sum()
        item_count_ratings = self.train_df.groupby('book_id')['rating'].count()
        self.item_bias = (item_sum_ratings - item_count_ratings * self.global_mean) / (item_count_ratings + lambda_reg)

        # Map item bias back to the original dataframe
        self.train_df['item_bias'] = self.train_df['book_id'].map(self.item_bias)

        # Normalize ratings
        self.train_df['normalised_rating'] = self.train_df['rating'] - self.train_df['user_bias'] - self.train_df['item_bias']

        # Convert to Surprise dataset
        self.reader = Reader(rating_scale=(self.train_df['rating'].min(), self.train_df['rating'].max()))
        self.train_data = Dataset.load_from_df(self.train_df[['user_id', 'book_id', 'normalised_rating']], self.reader)

        # Convert test_df to Surprise dataset without normalization
        self.test_data = Dataset.load_from_df(self.test_df[['user_id', 'book_id', 'rating']], self.reader)

        # Build full trainset and testset
        self.trainset = self.train_data.build_full_trainset()
        self.testset = self.test_data.construct_testset([(uid, iid, r, {}) for uid, iid, r in self.test_df[['user_id', 'book_id', 'rating']].values])

    def grid_search_and_fit(self):
        # Define a parameter grid for RandomizedSVD
        param_grid = {
            'n_factors': [60],
            'n_iter': [18],
            'random_state': [42],
        }

        gs = GridSearchCV(RandomizedSVD, param_grid, measures=['rmse'], cv=2)
        gs.fit(self.train_data)
        best_params = gs.best_params['rmse']
        self.best_randomized_svd = RandomizedSVD(**best_params)
        self.best_randomized_svd.fit(self.trainset)

    def make_predictions(self):
        # Make predictions using the trained model
        predictions = self.best_randomized_svd.test(self.testset)

        # Reverse bias terms
        def reverse_bias_terms(uid, iid, est, user_bias, item_bias, global_mean):
            user_b = user_bias.get(uid, 0)  # Default to 0 if the user/item is not in the training data
            item_b = item_bias.get(iid, 0)
            unbiased_prediction = est - user_b - item_b + global_mean
            return unbiased_prediction

        # Rescale predictions by reversing bias terms
        def unbiased_predictions(predictions, user_bias, item_bias, global_mean):
            adjusted_predictions = []
            for uid, iid, true_r, est, _ in predictions:
                unbiased_prediction = reverse_bias_terms(uid, iid, est, user_bias, item_bias, global_mean)
                unbiased_prediction = min(5, max(1, unbiased_prediction))  # Clip the rating
                adjusted_predictions.append((uid, iid, true_r, unbiased_prediction, _))
            return adjusted_predictions

        # Rescale predictions
        adjusted_predictions = unbiased_predictions(predictions, self.user_bias.to_dict(), self.item_bias.to_dict(), self.global_mean)
        adjusted_predictions_df = pd.DataFrame(adjusted_predictions, columns=['user_id', 'book_id', 'rating', 'adjusted_rating', 'details'])

        return adjusted_predictions_df
