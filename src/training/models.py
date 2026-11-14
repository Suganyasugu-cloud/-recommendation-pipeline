"""Recommender models sharing one interface.

Each model exposes:
    fit(train, store)            - learn from the train split using FeatureStore features
    score_users(user_ids)        - dense (n_users x n_items) relevance scores over `item_ids`
    predict_rating(users, items) - rating estimate (fallbacks documented per model)
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BaseRecommender:
    """Common bookkeeping: id <-> index maps and the seen-items mask."""

    name = "base"
    supports_rating = True

    def _index(self, train: pd.DataFrame) -> None:
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = np.sort(train["item_id"].unique())
        self.user_pos = {u: i for i, u in enumerate(self.user_ids)}
        self.item_pos = {i: j for j, i in enumerate(self.item_ids)}
        rows = train["user_id"].map(self.user_pos).to_numpy()
        cols = train["item_id"].map(self.item_pos).to_numpy()
        self.ratings = sparse.csr_matrix((train["rating"].astype(float), (rows, cols)),
                                         shape=(len(self.user_ids), len(self.item_ids)))
        self.global_mean = float(train["rating"].mean())

    def seen_mask(self, user_ids) -> np.ndarray:
        """Boolean (n_users x n_items) matrix of items each user already rated in train."""
        rows = [self.user_pos[u] for u in user_ids]
        return self.ratings[rows].toarray() > 0

    def get_params(self) -> dict:
        """Hyper-parameters logged to the experiment tracker."""
        return {}


class PopularityRecommender(BaseRecommender):
    """Baseline: recommends the same globally most-positively-rated items to everyone."""

    name = "popularity"

    def fit(self, train, store):
        self._index(train)
        feats = store.get_item_features(self.item_ids)
        self.popularity = feats["item_positive_count"].to_numpy(dtype=float)
        self.item_mean = feats["item_avg_rating"].fillna(self.global_mean).to_numpy()
        return self

    def score_users(self, user_ids) -> np.ndarray:
        return np.tile(self.popularity, (len(user_ids), 1))

    def predict_rating(self, users, items) -> np.ndarray:
        """Item mean rating; global mean for unseen items."""
        return np.array([self.item_mean[self.item_pos[i]] if i in self.item_pos else self.global_mean
                         for i in items])


class ItemKNNRecommender(BaseRecommender):
    """Personalised item-based collaborative filtering on implicit positive feedback."""

    name = "item_knn"
    supports_rating = False  # ranking-only model

    def __init__(self, neighbors: int = 50, positive_threshold: int = 4):
        self.neighbors = neighbors
        self.threshold = positive_threshold

    def get_params(self) -> dict:
        return {"neighbors": self.neighbors, "similarity": "cosine", "positive_threshold": self.threshold}

    def fit(self, train, store):
        self._index(train)
        implicit = (self.ratings >= self.threshold).astype(float).tocsr()
        normed = normalize(implicit, axis=0)
        sim = (normed.T @ normed).toarray()
        np.fill_diagonal(sim, 0.0)
        keep = np.argsort(-sim, axis=0)[self.neighbors:]  # indices to zero per column
        np.put_along_axis(sim, keep, 0.0, axis=0)
        self.similarity = sim
        self.implicit = implicit
        return self

    def score_users(self, user_ids) -> np.ndarray:
        rows = [self.user_pos[u] for u in user_ids]
        return self.implicit[rows] @ self.similarity

    def predict_rating(self, users, items) -> np.ndarray:
        """Not a rating model: RMSE/MAE are reported as NaN for it."""
        raise NotImplementedError("item_knn is a ranking-only model")


class SVDRecommender(BaseRecommender):
    """Matrix factorisation: damped user/item biases plus truncated SVD on the residuals."""

    name = "svd_mf"

    def __init__(self, factors: int = 30, damping: float = 10.0, min_item_support: int = 20,
                 seed: int = 42):
        self.factors, self.damping = factors, damping
        self.min_item_support, self.seed = min_item_support, seed

    def get_params(self) -> dict:
        return {"factors": self.factors, "bias_damping": self.damping,
                "min_item_support": self.min_item_support, "seed": self.seed}

    def fit(self, train, store):
        self._index(train)
        mu = self.global_mean
        counts = np.asarray((self.ratings > 0).sum(axis=0)).ravel()
        item_sum = np.asarray(self.ratings.sum(axis=0)).ravel()
        self.item_bias = (item_sum - mu * counts) / (counts + self.damping)
        residual = self.ratings.copy().tocoo()
        residual.data = residual.data - mu - self.item_bias[residual.col]
        user_counts = np.asarray((self.ratings > 0).sum(axis=1)).ravel()
        user_sum = np.asarray(residual.tocsr().sum(axis=1)).ravel()
        self.user_bias = user_sum / (user_counts + self.damping)
        residual.data = residual.data - self.user_bias[residual.row]
        self.svd = TruncatedSVD(n_components=self.factors, random_state=self.seed)
        self.user_factors = self.svd.fit_transform(residual.tocsr())
        self.item_factors = self.svd.components_
        self.supported = counts >= self.min_item_support
        return self

    def _predict_dense(self, rows) -> np.ndarray:
        return (self.global_mean + self.user_bias[rows][:, None] + self.item_bias[None, :]
                + self.user_factors[rows] @ self.item_factors)

    def score_users(self, user_ids) -> np.ndarray:
        rows = [self.user_pos[u] for u in user_ids]
        scores = self._predict_dense(rows)
        scores[:, ~self.supported] = -np.inf  # avoid recommending items with too little evidence
        return scores

    def predict_rating(self, users, items) -> np.ndarray:
        """Clipped bias + latent-factor estimate; global mean when user or item is unseen."""
        out = np.full(len(users), self.global_mean)
        for n, (u, i) in enumerate(zip(users, items)):
            if u in self.user_pos and i in self.item_pos:
                r, c = self.user_pos[u], self.item_pos[i]
                out[n] = (self.global_mean + self.user_bias[r] + self.item_bias[c]
                          + self.user_factors[r] @ self.item_factors[:, c])
        return np.clip(out, 1.0, 5.0)
