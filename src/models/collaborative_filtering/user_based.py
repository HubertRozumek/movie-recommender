"""User-based collaborative filtering implementation."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine, correlation
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering recommender."""
    
    def __init__(self,
                 k_neighbors: int = 50,
                 similarity_metric: str = 'cosine',
                 min_k: int = 5,
                 shrinkage: float = 100,
                 normalize: bool = True,
                 weighted: bool = True,
                 **kwargs):
        """
        Initialize user-based CF model.
        
        Args:
            k_neighbors: Number of similar users to consider
            similarity_metric: Similarity metric ('cosine', 'pearson', 'jaccard', 'euclidean')
            min_k: Minimum number of neighbors required for prediction
            shrinkage: Shrinkage parameter for similarity calculation
            normalize: Whether to normalize ratings
            weighted: Whether to use weighted average
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name="UserBasedCF", **kwargs)
        
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_k = min_k
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.weighted = weighted
        
        # Model components
        self.user_matrix = None
        self.user_similarities = None
        self.user_means = None
        self.user_stds = None
        self.global_mean = None
        
        # Store parameters
        self.params = {
            'k_neighbors': k_neighbors,
            'similarity_metric': similarity_metric,
            'min_k': min_k,
            'shrinkage': shrinkage,
            'normalize': normalize,
            'weighted': weighted
        }
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'UserBasedCF':
        """
        Train user-based CF model.
        
        Args:
            train_data: Training data
            val_data: Validation data (not used in this implementation)
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        if self.verbose:
            logger.info("Training User-Based Collaborative Filtering model...")
        
        start_time = time.time()
        
        # Prepare data
        self.user_matrix = self._prepare_data(train_data)
        
        # Calculate global statistics
        self._calculate_statistics()
        
        # Normalize if needed
        if self.normalize:
            self._normalize_ratings()
        
        # Compute user similarities
        self._compute_similarities()
        
        # Mark as fitted
        self.is_fitted = True
        
        # Record training time
        self.training_history['train_time'] = time.time() - start_time
        
        if self.verbose:
            logger.info(f"Training completed in {self.training_history['train_time']:.2f} seconds")
            logger.info(f"Model trained on {self.n_users} users and {self.n_items} items")
        
        return self
    
    def predict(self, user_ids: Union[int, List[int], np.ndarray],
                item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: User ID(s)
            item_ids: Item ID(s)
            
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        user_ids = self._validate_user_ids(user_ids)
        item_ids = self._validate_item_ids(item_ids)
        
        # Ensure same length
        if len(user_ids) == 1:
            user_ids = np.repeat(user_ids, len(item_ids))
        elif len(item_ids) == 1:
            item_ids = np.repeat(item_ids, len(user_ids))
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                # Return global mean for unknown user/item
                predictions.append(self.global_mean)
                continue
            
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            # Get k most similar users who rated this item
            similar_users = self._get_similar_users_for_item(user_idx, item_idx)
            
            if len(similar_users) < self.min_k:
                # Not enough neighbors, use user mean or global mean
                if self.user_means is not None:
                    pred = self.user_means[user_idx]
                else:
                    pred = self.global_mean
            else:
                # Weighted average of neighbors' ratings
                if self.weighted:
                    weights = self.user_similarities[user_idx, similar_users]
                    ratings = self.user_matrix[similar_users, item_idx].toarray().ravel()

                    # Apply shrinkage
                    weights = weights * len(similar_users) / (len(similar_users) + self.shrinkage)

                    if np.all(weights == 0):
                        # fallback: zwykła średnia zamiast ważonej
                        pred = ratings.mean() if ratings.size > 0 else self.global_mean
                    else:
                        pred = np.average(ratings, weights=weights)

                # Add back user bias if normalized
                if self.normalize and self.user_means is not None:
                    pred = pred * self.user_stds[user_idx] + self.user_means[user_idx]
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def recommend(self, user_ids: Union[int, List[int], np.ndarray],
                  n_recommendations: int = 10,
                  filter_seen: bool = True,
                  return_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate top-N recommendations for users.
        
        Args:
            user_ids: User ID(s)
            n_recommendations: Number of recommendations per user
            filter_seen: Whether to filter out already rated items
            return_scores: Whether to return recommendation scores
            
        Returns:
            Array of recommended item IDs (and optionally scores)
        """
        user_ids = self._validate_user_ids(user_ids)
        
        all_recommendations = []
        all_scores = []
        
        for user_id in user_ids:
            if user_id not in self.user_to_idx:
                # For new users, recommend popular items
                recommendations = self._get_popular_items(n_recommendations)
                scores = np.zeros(n_recommendations)
            else:
                user_idx = self.user_to_idx[user_id]
                
                # Get all items
                all_items = np.arange(self.n_items)
                
                # Filter seen items if needed
                if filter_seen:
                    seen_items = self.user_matrix[user_idx].nonzero()[1]
                    unseen_mask = np.ones(self.n_items, dtype=bool)
                    unseen_mask[seen_items] = False
                    candidate_items = all_items[unseen_mask]
                else:
                    candidate_items = all_items
                
                if len(candidate_items) == 0:
                    recommendations = np.array([])
                    scores = np.array([])
                else:
                    # Predict scores for all candidate items
                    candidate_scores = []
                    for item_idx in candidate_items:
                        item_id = self.idx_to_item[item_idx]
                        score = self.predict([user_id], [item_id])[0]
                        candidate_scores.append(score)
                    
                    candidate_scores = np.array(candidate_scores)
                    
                    # Get top-N items
                    top_indices = np.argsort(candidate_scores)[-n_recommendations:][::-1]
                    recommendations = candidate_items[top_indices]
                    scores = candidate_scores[top_indices]
                    
                    # Convert to original IDs
                    recommendations = [self.idx_to_item[idx] for idx in recommendations]
            
            all_recommendations.append(recommendations)
            all_scores.append(scores)
        
        if return_scores:
            return np.array(all_recommendations), np.array(all_scores)
        else:
            return np.array(all_recommendations)
    
    def _calculate_statistics(self):
        """Calculate global and user statistics."""
        self.global_mean = float(self.user_matrix.data.mean()) if self.user_matrix.nnz > 0 else 0.0

        eps = 1e-8
        self.user_means = np.zeros(self.n_users, dtype=float)
        self.user_stds = np.ones(self.n_users, dtype=float)

        for u in range(self.n_users):
            user_ratings = self.user_matrix[u].data
            if user_ratings.size > 0:
                mu = float(user_ratings.mean())
                self.user_means[u] = mu
                if user_ratings.size > 1:
                    sd = float(user_ratings.std())
                    if sd < eps:
                        sd = 1.0
                    self.user_stds[u] = sd
                else:
                    self.user_stds[u] = 1.0
            else:
                self.user_stds[u] = 1.0

    
    def _normalize_ratings(self):
        """Normalize ratings by removing user bias."""
        if self.verbose:
            logger.info("Normalizing ratings...")

        normalized_matrix = self.user_matrix.copy().tocsr()

        for u in range(self.n_users):
            item_idx = normalized_matrix[u].indices 
            if item_idx.size > 0:
                values = normalized_matrix[u, item_idx].toarray().ravel() 
                normed = (values - self.user_means[u]) / self.user_stds[u]
                np.nan_to_num(normed, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                normalized_matrix[u, item_idx] = normed

        self.user_matrix = normalized_matrix
    
    def _compute_similarities(self):
        """Compute user-user similarities."""
        if self.verbose:
            logger.info(f"Computing user similarities using {self.similarity_metric} metric...")
        
        if sparse.isspmatrix(self.user_matrix):
            d = self.user_matrix.data
            if np.isnan(d).any() or np.isinf(d).any():
                logger.warning("Detected NaN/Inf in rating matrix; replacing with 0.0 before similarity.")
                np.nan_to_num(d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            self.user_similarities = cosine_similarity(self.user_matrix)
        
        elif self.similarity_metric == 'pearson':
            # Pearson correlation
            # Convert to dense for correlation calculation
            dense_matrix = self.user_matrix.toarray()
            self.user_similarities = np.corrcoef(dense_matrix)
            # Handle NaN values
            self.user_similarities = np.nan_to_num(self.user_similarities)
        
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity (for binary ratings)
            binary_matrix = (self.user_matrix > 0).astype(float)
            self.user_similarities = self._jaccard_similarity(binary_matrix)
        
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance converted to similarity
            distances = pairwise_distances(self.user_matrix, metric='euclidean')
            self.user_similarities = 1 / (1 + distances)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (don't consider self-similarity)
        np.fill_diagonal(self.user_similarities, 0)
        
        if self.verbose:
            logger.info("User similarities computed")
    
    def _jaccard_similarity(self, binary_matrix):
        """Compute Jaccard similarity between users."""
        intersection = binary_matrix.dot(binary_matrix.T)
        row_sums = binary_matrix.sum(axis=1).A1
        union = row_sums[:, None] + row_sums[None, :] - intersection
        
        # Avoid division by zero
        union[union == 0] = 1
        
        return intersection / union
    
    def _get_similar_users_for_item(self, user_idx: int, item_idx: int) -> np.ndarray:
        """Get k most similar users who rated the item."""
        # Find users who rated this item
        users_who_rated = self.user_matrix[:, item_idx].nonzero()[0]
        
        # Remove the target user
        users_who_rated = users_who_rated[users_who_rated != user_idx]
        
        if len(users_who_rated) == 0:
            return np.array([])
        
        # Get similarities for these users
        similarities = self.user_similarities[user_idx, users_who_rated]
        
        # Get top-k
        k = min(self.k_neighbors, len(users_who_rated))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return users_who_rated[top_k_indices]
    
    def _get_popular_items(self, n_items: int) -> List[int]:
        """Get most popular items."""
        item_counts = np.array(self.user_matrix.sum(axis=0)).flatten()
        popular_indices = np.argsort(item_counts)[-n_items:][::-1]
        return [self.idx_to_item[idx] for idx in popular_indices]
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar users.
        
        Args:
            user_id: User ID
            n_similar: Number of similar users to return
            
        Returns:
            Tuple of (user IDs, similarity scores)
        """
        if user_id not in self.user_to_idx:
            return np.array([]), np.array([])
        
        user_idx = self.user_to_idx[user_id]
        similarities = self.user_similarities[user_idx]
        
        # Get top-N similar users
        top_indices = np.argsort(similarities)[-n_similar:][::-1]
        
        similar_users = [self.idx_to_user[idx] for idx in top_indices]
        similarity_scores = similarities[top_indices]
        
        return np.array(similar_users), similarity_scores
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """
        Explain why an item was recommended.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Explanation dictionary
        """
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return super().explain_recommendation(user_id, item_id)
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        # Get similar users who rated this item
        similar_users = self._get_similar_users_for_item(user_idx, item_idx)
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': self.predict([user_id], [item_id])[0],
            'n_similar_users': len(similar_users),
            'similar_users': [self.idx_to_user[idx] for idx in similar_users[:5]],
            'explanation': f"Based on ratings from {len(similar_users)} similar users"
        }
        
        return explanation