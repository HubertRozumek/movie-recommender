"""Item-based collaborative filtering implementation."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering recommender."""
    
    def __init__(self,
                 k_neighbors: int = 50,
                 similarity_metric: str = 'pearson',
                 min_k: int = 5,
                 alpha: float = 0.5,
                 normalize: bool = True,
                 weighted: bool = True,
                 **kwargs):
        """
        Initialize item-based CF model.
        
        Args:
            k_neighbors: Number of similar items to consider
            similarity_metric: Similarity metric ('cosine', 'pearson', 'adjusted_cosine', 'jaccard')
            min_k: Minimum number of neighbors required for prediction
            alpha: Significance weighting parameter
            normalize: Whether to normalize ratings
            weighted: Whether to use weighted average
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name="ItemBasedCF", **kwargs)
        
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_k = min_k
        self.alpha = alpha
        self.normalize = normalize
        self.weighted = weighted
        
        # Model components
        self.item_matrix = None
        self.item_similarities = None
        self.item_means = None
        self.item_stds = None
        self.global_mean = None
        
        # Store parameters
        self.params = {
            'k_neighbors': k_neighbors,
            'similarity_metric': similarity_metric,
            'min_k': min_k,
            'alpha': alpha,
            'normalize': normalize,
            'weighted': weighted
        }
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'ItemBasedCF':
        """
        Train item-based CF model.
        
        Args:
            train_data: Training data
            val_data: Validation data (not used in this implementation)
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        if self.verbose:
            logger.info("Training Item-Based Collaborative Filtering model...")
        
        start_time = time.time()
        
        # Prepare data
        user_item_matrix = self._prepare_data(train_data)
        # Transpose to get item-user matrix
        self.item_matrix = user_item_matrix.T.tocsr()
        
        # Calculate global statistics
        self._calculate_statistics()
        
        # Normalize if needed
        if self.normalize:
            self._normalize_ratings()
        
        # Compute item similarities
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
            
            # Get k most similar items rated by this user
            similar_items = self._get_similar_items_for_user(user_idx, item_idx)
            
            if len(similar_items) < self.min_k:
                # Not enough neighbors, use item mean or global mean
                if self.item_means is not None:
                    pred = self.item_means[item_idx]
                else:
                    pred = self.global_mean
            else:
                # Weighted average of user's ratings on similar items
                if self.weighted:
                    weights = self.item_similarities[item_idx, similar_items]
                    # Get user's ratings on similar items (transpose back to user-item)
                    ratings = self.item_matrix.T[user_idx, similar_items].toarray().flatten()
                    
                    # Apply significance weighting
                    n_common = np.sum(ratings > 0)
                    weight_factor = n_common / (n_common + self.alpha)
                    weights = weights * weight_factor
                    
                    if np.all(weights == 0) or weights.sum() == 0:
                        if self.item_means is not None and not np.isnan(self.item_means[item_idx]):
                            pred = self.item_means[item_idx]
                        else:
                            pred = self.global_mean
                    else:
                        pred = np.average(ratings, weights=weights)

                    if np.isnan(pred):
                        pred = self.global_mean
                else:
                    # Simple average
                    ratings = self.item_matrix.T[user_idx, similar_items].toarray().flatten()
                    pred = np.mean(ratings)
                
                # Add back item bias if normalized
                if self.normalize and self.item_means is not None:
                    pred = pred * self.item_stds[item_idx] + self.item_means[item_idx]
            
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
                
                # Get user's rated items
                user_rated_items = self.item_matrix.T[user_idx].nonzero()[1]
                
                if len(user_rated_items) == 0:
                    # User has no ratings, recommend popular items
                    recommendations = self._get_popular_items(n_recommendations)
                    scores = np.zeros(n_recommendations)
                else:
                    # Calculate scores for all items based on similarity to user's rated items
                    all_scores_dict = {}
                    
                    for rated_item_idx in user_rated_items:
                        user_rating = self.item_matrix.T[user_idx, rated_item_idx]
                        
                        # Get similar items to this rated item
                        similarities = self.item_similarities[rated_item_idx]
                        similar_items = np.argsort(similarities)[-self.k_neighbors:][::-1]
                        
                        for similar_item_idx in similar_items:
                            if filter_seen and similar_item_idx in user_rated_items:
                                continue
                            
                            if similar_item_idx not in all_scores_dict:
                                all_scores_dict[similar_item_idx] = 0
                            
                            # Add weighted contribution
                            similarity = similarities[similar_item_idx]
                            all_scores_dict[similar_item_idx] += user_rating * similarity
                    
                    if not all_scores_dict:
                        # No recommendations possible
                        recommendations = []
                        scores = []
                    else:
                        # Sort by score and get top-N
                        sorted_items = sorted(all_scores_dict.items(), 
                                            key=lambda x: x[1], reverse=True)
                        
                        recommendations = []
                        scores = []
                        for item_idx, score in sorted_items[:n_recommendations]:
                            recommendations.append(self.idx_to_item[item_idx])
                            scores.append(score)
            
            all_recommendations.append(recommendations)
            all_scores.append(scores)
        
        if return_scores:
            return np.array(all_recommendations), np.array(all_scores)
        else:
            return np.array(all_recommendations)
    
    def _calculate_statistics(self):
        """Calculate global and item statistics."""
        # Global mean
        self.global_mean = (
            np.nanmean(self.item_matrix.data) 
            if self.item_matrix.data.size > 0 
            else 0.0
            )
        
        # Item means and stds
        self.item_means = np.zeros(self.n_items)
        self.item_stds = np.ones(self.n_items)
        
        for i in range(self.n_items):
            item_ratings = self.item_matrix[i].data
            if len(item_ratings) > 0:
                self.item_means[i] = item_ratings.mean()
                std = item_ratings.std()
                self.item_stds[i] = std if std > 0 else 1.0
            elif self.similarity_metric == 'pearson':
                dense_matrix = self.item_matrix.toarray()
                self.item_similarities = np.corrcoef(dense_matrix)
                self.item_similarities = np.nan_to_num(self.item_similarities, nan=0.0)
    
    def _normalize_ratings(self):
        """Normalize ratings by removing item bias."""
        if self.verbose:
            logger.info("Normalizing ratings...")
        
        # Create normalized matrix
        normalized_matrix = self.item_matrix.copy()
        
        for i in range(self.n_items):
            item_ratings_indices = self.item_matrix[i].nonzero()[1]
            if len(item_ratings_indices) > 0:
                normalized_matrix[i, item_ratings_indices] = (
                    (self.item_matrix[i, item_ratings_indices].toarray() - self.item_means[i]) 
                    / (self.item_stds[i] if self.item_stds[i] > 0 else 1.0)
                )

        
        self.item_matrix = normalized_matrix
    
    def _compute_similarities(self):
        """Compute item-item similarities."""
        if self.verbose:
            logger.info(f"Computing item similarities using {self.similarity_metric} metric...")
        
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            self.item_similarities = cosine_similarity(self.item_matrix)
        
        elif self.similarity_metric == 'pearson':
            # Pearson correlation
            # Convert to dense for correlation calculation
            dense_matrix = self.item_matrix.toarray()
            self.item_similarities = np.corrcoef(dense_matrix)
            # Handle NaN values
            self.item_similarities = np.nan_to_num(self.item_similarities)
        
        elif self.similarity_metric == 'adjusted_cosine':
            # Adjusted cosine (accounting for user bias)
            # Need to work with user-item matrix
            user_item_matrix = self.item_matrix.T
            
            # Subtract user means
            user_means = user_item_matrix.mean(axis=1).A1
            adjusted_matrix = user_item_matrix.copy()
            
            for u in range(self.n_users):
                user_items = user_item_matrix[u].nonzero()[1]
                if len(user_items) > 0:
                    adjusted_matrix[u, user_items] -= user_means[u]
            
            # Now compute cosine similarity on adjusted matrix
            self.item_similarities = cosine_similarity(adjusted_matrix.T)
        
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity (for binary ratings)
            binary_matrix = (self.item_matrix > 0).astype(float)
            self.item_similarities = self._jaccard_similarity(binary_matrix)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Set diagonal to 0 (don't consider self-similarity)
        np.fill_diagonal(self.item_similarities, 0)
        
        if self.verbose:
            logger.info("Item similarities computed")
    
    def _jaccard_similarity(self, binary_matrix):
        """Compute Jaccard similarity between items."""
        intersection = binary_matrix.dot(binary_matrix.T)
        row_sums = binary_matrix.sum(axis=1).A1
        union = row_sums[:, None] + row_sums[None, :] - intersection
        
        # Avoid division by zero
        union[union == 0] = 1
        
        return intersection / union
    
    def _get_similar_items_for_user(self, user_idx: int, item_idx: int) -> np.ndarray:
        """Get k most similar items that the user has rated."""
        # Get items rated by this user
        user_rated_items = self.item_matrix.T[user_idx].nonzero()[1]
        
        # Remove the target item
        user_rated_items = user_rated_items[user_rated_items != item_idx]
        
        if len(user_rated_items) == 0:
            return np.array([])
        
        # Get similarities for these items
        similarities = self.item_similarities[item_idx, user_rated_items]
        
        # Get top-k
        k = min(self.k_neighbors, len(user_rated_items))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return user_rated_items[top_k_indices]
    
    def _get_popular_items(self, n_items: int) -> List[int]:
        """Get most popular items."""
        item_counts = np.array(self.item_matrix.sum(axis=1)).flatten()
        popular_indices = np.argsort(item_counts)[-n_items:][::-1]
        return [self.idx_to_item[idx] for idx in popular_indices]
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            Tuple of (item IDs, similarity scores)
        """
        if item_id not in self.item_to_idx:
            return np.array([]), np.array([])
        
        item_idx = self.item_to_idx[item_id]
        similarities = self.item_similarities[item_idx]
        
        # Get top-N similar items
        top_indices = np.argsort(similarities)[-n_similar:][::-1]
        
        similar_items = [self.idx_to_item[idx] for idx in top_indices]
        similarity_scores = similarities[top_indices]
        
        return np.array(similar_items), similarity_scores
    
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
        
        # Get similar items that user has rated
        similar_items = self._get_similar_items_for_user(user_idx, item_idx)
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': self.predict([user_id], [item_id])[0],
            'n_similar_items': len(similar_items),
            'similar_items': [self.idx_to_item[idx] for idx in similar_items[:5]],
            'explanation': f"Based on similarity to {len(similar_items)} items you've rated"
        }
        
        return explanation