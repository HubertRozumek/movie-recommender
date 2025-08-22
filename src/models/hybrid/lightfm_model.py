"""LightFM hybrid factorization model implementation."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class LightFMModel(BaseRecommender):
    """LightFM hybrid factorization model."""
    
    def __init__(self,
                 no_components: int = 100,
                 k: int = 5,
                 n: int = 10,
                 learning_schedule: str = 'adagrad',
                 loss: str = 'warp',
                 learning_rate: float = 0.05,
                 rho: float = 0.95,
                 epsilon: float = 1e-06,
                 item_alpha: float = 0.0,
                 user_alpha: float = 0.0,
                 max_sampled: int = 10,
                 num_epochs: int = 30,
                 num_threads: int = 1,
                 **kwargs):
        """
        Initialize LightFM model.
        
        Args:
            no_components: Number of latent components
            k: For k-OS training
            n: For adaptive k-OS training
            learning_schedule: Learning rate schedule ('adagrad', 'adadelta')
            loss: Loss function ('warp', 'bpr', 'warp-kos', 'logistic')
            learning_rate: Initial learning rate
            rho: For adadelta learning schedule
            epsilon: Small value for numerical stability
            item_alpha: L2 penalty on item features
            user_alpha: L2 penalty on user features
            max_sampled: Maximum number of negative samples for WARP
            num_epochs: Number of training epochs
            num_threads: Number of parallel threads
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name="LightFM", **kwargs)
        
        self.no_components = no_components
        self.k = k
        self.n = n
        self.learning_schedule = learning_schedule
        self.loss = loss
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.max_sampled = max_sampled
        self.num_epochs = num_epochs
        self.num_threads = num_threads
        
        # Model
        self.model = None
        self.interactions = None
        self.user_features = None
        self.item_features = None
        
        # Store parameters
        self.params = {
            'no_components': no_components,
            'loss': loss,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'item_alpha': item_alpha,
            'user_alpha': user_alpha
        }
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            user_features: Optional[sparse.spmatrix] = None,
            item_features: Optional[sparse.spmatrix] = None,
            **kwargs) -> 'LightFMModel':
        """
        Train LightFM model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            user_features: Optional user feature matrix
            item_features: Optional item feature matrix
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        if self.verbose:
            logger.info("Training LightFM model...")
        
        start_time = time.time()
        
        # Prepare interaction matrix
        self.interactions = self._prepare_data(train_data)
        
        # Store features if provided
        self.user_features = user_features
        self.item_features = item_features
        
        # Initialize LightFM model
        self.model = LightFM(
            no_components=self.no_components,
            k=self.k,
            n=self.n,
            learning_schedule=self.learning_schedule,
            loss=self.loss,
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
            max_sampled=self.max_sampled,
            random_state=self.random_state
        )
        
        # Train model
        for epoch in range(self.num_epochs):
            self.model.fit_partial(
                interactions=self.interactions,
                user_features=self.user_features,
                item_features=self.item_features,
                epochs=1,
                num_threads=self.num_threads,
                verbose=False
            )
            
            if self.verbose and (epoch + 1) % 5 == 0:
                # Calculate training metrics
                train_precision = precision_at_k(
                    self.model,
                    self.interactions,
                    user_features=self.user_features,
                    item_features=self.item_features,
                    k=10,
                    num_threads=self.num_threads
                ).mean()
                
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                           f"Train Precision@10: {train_precision:.4f}")
        
        # Mark as fitted
        self.is_fitted = True
        
        # Record training time
        self.training_history['train_time'] = time.time() - start_time
        self.training_history['n_epochs'] = self.num_epochs
        
        if self.verbose:
            logger.info(f"Training completed in {self.training_history['train_time']:.2f} seconds")
            logger.info(f"LightFM model trained with {self.no_components} components")
        
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
                predictions.append(0.0)
                continue
            
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            # Get prediction from LightFM
            score = self.model.predict(
                user_ids=np.array([user_idx]),
                item_ids=np.array([item_idx]),
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=self.num_threads
            )[0]
            
            predictions.append(score)
        
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
                
                # Get scores for all items
                scores = self.model.predict(
                    user_ids=np.full(self.n_items, user_idx),
                    item_ids=all_items,
                    user_features=self.user_features,
                    item_features=self.item_features,
                    num_threads=self.num_threads
                )
                
                # Filter seen items if needed
                if filter_seen:
                    seen_items = self.interactions[user_idx].nonzero()[1]
                    scores[seen_items] = -np.inf
                
                # Get top-N items
                top_indices = np.argsort(scores)[-n_recommendations:][::-1]
                
                recommendations = [self.idx_to_item[idx] for idx in top_indices]
                top_scores = scores[top_indices]
            
            all_recommendations.append(recommendations)
            all_scores.append(top_scores if 'top_scores' in locals() else scores)
        
        if return_scores:
            return np.array(all_recommendations), np.array(all_scores)
        else:
            return np.array(all_recommendations)
    
    def _get_popular_items(self, n_items: int) -> List[int]:
        """Get most popular items."""
        if self.interactions is None:
            return list(range(n_items))
        
        item_counts = np.array(self.interactions.sum(axis=0)).flatten()
        popular_indices = np.argsort(item_counts)[-n_items:][::-1]
        return [self.idx_to_item[idx] for idx in popular_indices]
    
    def get_user_representations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get user embeddings and biases.
        
        Returns:
            Tuple of (user embeddings, user biases)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        user_embeddings = self.model.user_embeddings
        user_biases = self.model.user_biases
        
        return user_embeddings, user_biases
    
    def get_item_representations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get item embeddings and biases.
        
        Returns:
            Tuple of (item embeddings, item biases)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        item_embeddings = self.model.item_embeddings
        item_biases = self.model.item_biases
        
        return item_embeddings, item_biases
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items based on embeddings.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            Tuple of (item IDs, similarity scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if item_id not in self.item_to_idx:
            return np.array([]), np.array([])
        
        item_idx = self.item_to_idx[item_id]
        
        # Get item embedding
        item_embedding = self.model.item_embeddings[item_idx]
        
        # Compute similarities with all items
        similarities = self.model.item_embeddings.dot(item_embedding)
        
        # Get top-N similar items (excluding self)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        similar_items = [self.idx_to_item[idx] for idx in similar_indices]
        similarity_scores = similarities[similar_indices]
        
        return np.array(similar_items), similarity_scores
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar users based on embeddings.
        
        Args:
            user_id: User ID
            n_similar: Number of similar users to return
            
        Returns:
            Tuple of (user IDs, similarity scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_to_idx:
            return np.array([]), np.array([])
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user embedding
        user_embedding = self.model.user_embeddings[user_idx]
        
        # Compute similarities with all users
        similarities = self.model.user_embeddings.dot(user_embedding)
        
        # Get top-N similar users (excluding self)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        similar_users = [self.idx_to_user[idx] for idx in similar_indices]
        similarity_scores = similarities[similar_indices]
        
        return np.array(similar_users), similarity_scores
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """
        Explain recommendation using feature contributions.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            return super().explain_recommendation(user_id, item_id)
        
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return super().explain_recommendation(user_id, item_id)
        
        # Get embeddings
        user_embedding = self.model.user_embeddings[user_idx]
        item_embedding = self.model.item_embeddings[item_idx]
        user_bias = self.model.user_biases[user_idx]
        item_bias = self.model.item_biases[item_idx]
        
        # Calculate score components
        interaction_score = user_embedding.dot(item_embedding)
        total_score = interaction_score + user_bias + item_bias
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_score': total_score,
            'model': 'LightFM',
            'loss_function': self.loss,
            'components': {
                'interaction': interaction_score,
                'user_bias': user_bias,
                'item_bias': item_bias
            },
            'explanation': (
                f"Score {total_score:.2f} based on:\n"
                f"- User-item interaction: {interaction_score:.2f}\n"
                f"- User preference: {user_bias:.2f}\n"
                f"- Item popularity: {item_bias:.2f}"
            )
        }
        
        return explanation