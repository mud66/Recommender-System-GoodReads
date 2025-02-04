from sklearn.utils.extmath import randomized_svd 
from surprise import AlgoBase, Trainset
import numpy as np

class RandomizedSVD(AlgoBase):
    def __init__(self, n_factors=60, n_iter=18, random_state=None, power_iteration_normalizer='QR'):
        super().__init__()
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state
        self.power_iteration_normalizer = power_iteration_normalizer

    def fit(self, trainset: Trainset):
        super().fit(trainset)

        num_users = trainset.n_users
        num_items = trainset.n_items
        user_item_matrix = np.zeros((num_users, num_items))

        for user_id, iid, rating in trainset.all_ratings():
            user_item_matrix[int(user_id), int(iid)] = rating

        # Apply SVD directly to the user-item matrix
        U, Sigma, VT = randomized_svd(user_item_matrix, 
                                      n_components=self.n_factors, 
                                      n_iter=self.n_iter, 
                                      random_state=self.random_state,
                                      power_iteration_normalizer=self.power_iteration_normalizer)

        self.user_factors = U.dot(np.diag(Sigma))
        self.item_factors = VT.T

        return self

    def estimate(self, u, i):
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            return np.dot(self.user_factors[u, :], self.item_factors[i, :])
        else:
            return self.trainset.global_mean
