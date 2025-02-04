
import sys
import os
from custom_svd import RandomizedSVD
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, GridSearchCV
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pickle
from custom_svd import RandomizedSVD
from collections import defaultdict
from math import log2
from tqdm import tqdm

tqdm.pandas()

class BiasSVDModel:
    def __init__(self, interactions_path):
        self.interactions = self.load_interactions(interactions_path)
        self.train_df, self.test_df = self.prepare_data()
        self.global_mean = self.train_df['rating'].mean()
        self.user_bias, self.item_bias = self.calculate_biases()
        self.reader = Reader(rating_scale=(self.train_df['rating'].min(), self.train_df['rating'].max()))
        self.train_data = self.convert_to_surprise_dataset(self.train_df, normalized=True)
        self.test_data = self.convert_to_surprise_dataset(self.test_df, normalized=False)
        self.trainset = self.train_data.build_full_trainset()
        self.testset = self.construct_testset(self.test_data)

    @staticmethod
    def load_interactions(path):
        interactions = pd.read_pickle(path)
        return interactions[['user_id', 'book_id', 'rating', 'is_read']]

    def prepare_data(self):
        interactions = self.interactions
        # Separate lower ratings by their respective values
        ratings_0 = interactions[interactions['rating'] == 0]
        ratings_1 = interactions[interactions['rating'] == 1]
        ratings_2 = interactions[interactions['rating'] == 2]
        higher_ratings = interactions[interactions['rating'] > 3]
        
        # Calculate the number of samples needed for each lower rating
        higher_count = len(higher_ratings)
        factor_0, factor_1, factor_2 = 0.2, 0.2, 0.2
        n_samples_0 = min(int(higher_count * factor_0), len(ratings_0))
        n_samples_1 = min(int(higher_count * factor_1), len(ratings_1))
        n_samples_2 = min(int(higher_count * factor_2), len(ratings_2))
        
        # Apply oversampling
        ratings_0_oversampled = resample(ratings_0, replace=True, n_samples=n_samples_0, random_state=42)
        ratings_1_oversampled = resample(ratings_1, replace=True, n_samples=n_samples_1, random_state=42)
        ratings_2_oversampled = resample(ratings_2, replace=True, n_samples=n_samples_2, random_state=42)
        
        # Combine the datasets
        interactions = pd.concat([higher_ratings, ratings_0_oversampled, ratings_1_oversampled, ratings_2_oversampled])
        return train_test_split(interactions, test_size=0.2, random_state=42)

    def calculate_biases(self):
        # Calculate user bias with regularization
        lambda_reg = 10
        user_sum_ratings = self.train_df.groupby('user_id')['rating'].sum()
        user_count_ratings = self.train_df.groupby('user_id')['rating'].count()
        user_bias = (user_sum_ratings - user_count_ratings * self.global_mean) / (user_count_ratings + lambda_reg)
        
        # Map user bias back to the original dataframe
        self.train_df['user_bias'] = self.train_df['user_id'].map(user_bias)
        
        # Calculate item bias with regularization
        item_sum_ratings = self.train_df.groupby('book_id')['rating'].sum()
        item_count_ratings = self.train_df.groupby('book_id')['rating'].count()
        item_bias = (item_sum_ratings - item_count_ratings * self.global_mean) / (item_count_ratings + lambda_reg)
        
        # Map item bias back to the original dataframe
        self.train_df['item_bias'] = self.train_df['book_id'].map(item_bias)
        
        return user_bias, item_bias

    def convert_to_surprise_dataset(self, df, normalized):
        if normalized:
            df['normalised_rating'] = df['rating'] - df['user_bias'] - df['item_bias']
            return Dataset.load_from_df(df[['user_id', 'book_id', 'normalised_rating']], self.reader)
        return Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], self.reader)

    def construct_testset(self, test_data):
        return test_data.construct_testset([(uid, iid, r) for uid, iid, r in self.test_df[['user_id', 'book_id', 'rating']].values])

    def train_best_model(self):
        param_grid = {
            'n_factors': [60],
            'n_iter': [18],
            'random_state': [42]
        }
        gs = GridSearchCV(RandomizedSVD, param_grid, measures=['rmse'], cv=2)
        gs.fit(self.train_data)
        best_params = gs.best_params['rmse']
        best_randomized_svd = RandomizedSVD(**best_params)
        best_randomized_svd.fit(self.trainset)
        return best_randomized_svd

    def predict(self, model):
        predictions = model.test(self.testset)
        return self.unbiased_predictions(predictions)

    def unbiased_predictions(self, predictions):
        adjusted_predictions = []
        for uid, iid, true_r, est, _ in predictions:
            unbiased_prediction = self.reverse_bias_terms(uid, iid, est)
            unbiased_prediction = min(5, max(1, unbiased_prediction))
            adjusted_predictions.append((uid, iid, true_r, unbiased_prediction, _))
        return adjusted_predictions

    def reverse_bias_terms(self, uid, iid, est):
        user_b = self.user_bias.get(uid, 0)
        item_b = self.item_bias.get(iid, 0)
        unbiased_prediction = est - user_b - item_b + self.global_mean
        return unbiased_prediction

    def precision_recall_ndcg_at_k(self, predictions, k, threshold):
        def dcg_at_k(scores, k):
            return sum([rel / log2(idx + 2) for idx, rel in enumerate(scores[:k])])

        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions, recalls, ndcgs = dict(), dict(), dict()

        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

            actual = [true_r for (_, true_r) in user_ratings]
            ideal = sorted(actual, reverse=True)
            idcg = dcg_at_k(ideal, k)
            dcg = dcg_at_k([rel for (est, rel) in user_ratings], k)
            ndcgs[uid] = dcg / idcg if idcg > 0 else 0

        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        ndcg = sum(ndcg for ndcg in ndcgs.values()) / len(ndcgs)

        return precision, recall, ndcg

    def save_model_and_biases(self, model_filename, biases_filename, model):
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)

        biases = {
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'global_mean': self.global_mean
        }

        with open(biases_filename, 'wb') as biases_file:
            pickle.dump(biases, biases_file)

if __name__ == "__main__":
    model = BiasSVDModel('../Pickle/interactions.pkl')
    best_model = model.train_best_model()
    predictions = model.predict(best_model)
    precision, recall, ndcg = model.precision_recall_ndcg_at_k(predictions, k=10, threshold=2)
    print(f'Adjusted Precision: {precision}, Adjusted Recall: {recall}, Adjusted nDCG: {ndcg}')
    model.save_model_and_biases('../Pickle/svd_model.pkl', '../Pickle/biases.pkl', best_model)
