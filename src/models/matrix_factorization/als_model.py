"""ALS (Alternating Least Squares) model implementation."""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False
    implicit = None

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class ALSModel(BaseRecommender):
    """Alternating Least Squares model using implicit library."""
    
    def __init__(self,
                 factors: int = 100,
                 regularization: float = 0.01,
                 iterations: int = 20,
                 alpha: float = 40,
                 dtype: type = np.float32,
                 use_native: bool = True,
                 use_cg: bool = True,
                 use_gpu: bool = False,
                 calculate_training_loss: bool = False,
                 num_threads: int = 0,
                 **kwargs):
        """
        Initialize ALS model.
        
        Args:
            factors: Number of latent factors (must be > 0)
            regularization: Regularization parameter (0 < reg < 1)
            iterations: Number of ALS iterations (must be > 0)
            alpha: Confidence weight for implicit feedback (must be > 0)
            dtype: Data type for computations
            use_native: Use Cython implementation
            use_cg: Use conjugate gradient solver
            use_gpu: Use GPU acceleration (requires cupy)
            calculate_training_loss: Whether to calculate training loss
            num_threads: Number of threads (0 = all available)
            **kwargs: Additional arguments for base class
            
        Raises:
            ImportError: If implicit library is not installed
            ValueError: If parameters are invalid
        """
        if not HAS_IMPLICIT:
            raise ImportError("implicit library is required. Install with: pip install implicit")
        
        super().__init__(model_name="ALS", **kwargs)
        
        # Validate parameters
        self._validate_parameters(factors, regularization, iterations, alpha)
        
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.dtype = dtype
        self.use_native = use_native
        self.use_cg = use_cg
        self.use_gpu = use_gpu
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads
        
        # Model components
        self.model = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        
        # Store parameters
        self.params = {
            'factors': factors,
            'regularization': regularization,
            'iterations': iterations,
            'alpha': alpha
        }
    
    def _validate_parameters(self, factors: int, regularization: float, 
                           iterations: int, alpha: float) -> None:
        """Validate model parameters."""
        if factors <= 0:
            raise ValueError(f"factors must be positive, got {factors}")
        if not 0 < regularization < 1:
            raise ValueError(f"regularization must be between 0 and 1, got {regularization}")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy
            return hasattr(implicit, 'gpu') and hasattr(implicit.gpu, 'als')
        except ImportError:
            return False
    
    def _estimate_memory_usage(self, matrix_shape: Tuple[int, int]) -> float:
        """Estimate memory usage in GB."""
        n_users, n_items = matrix_shape
        # Rough estimate: user factors + item factors + sparse matrix
        memory_gb = (n_users * self.factors + n_items * self.factors) * 4 / 1e9
        return memory_gb
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'ALSModel':
        """
        Train ALS model.
        
        Args:
            train_data: Training data (DataFrame or sparse matrix)
            val_data: Validation data (not used in ALS)
            **kwargs: Additional training parameters
            
        Returns:
            Self
            
        Raises:
            ValueError: If training data is invalid
            MemoryError: If data is too large for available memory
        """
        if self.verbose:
            logger.info("Training ALS model...")
        
        start_time = time.time()
        
        try:
            # Prepare data
            self.user_item_matrix = self._prepare_data(train_data)
            
            # Check memory usage
            memory_estimate = self._estimate_memory_usage(self.user_item_matrix.shape)
            if memory_estimate > 8:  # 8GB threshold
                logger.warning(f"Large matrix detected (~{memory_estimate:.2f}GB). "
                             f"Consider reducing factors or using sparse operations.")
            
            # Convert to implicit feedback format
            # Correct formula: C = 1 + alpha * R (confidence matrix)
            if sparse.issparse(self.user_item_matrix):
                # For sparse matrices, we need to handle this carefully
                self.user_item_matrix.data = 1 + self.alpha * self.user_item_matrix.data
            else:
                self.user_item_matrix = 1 + self.alpha * self.user_item_matrix
            
            self.user_item_matrix = self.user_item_matrix.astype(self.dtype)
            
            # Transpose for item-user matrix (implicit expects item-user format)
            self.item_user_matrix = self.user_item_matrix.T.tocsr()
            
            # Initialize ALS model
            self.model = self._initialize_model()
            
            # Train model
            self.model.fit(self.item_user_matrix, show_progress=self.verbose)
            
            # Mark as fitted
            self.is_fitted = True
            
            # Record training statistics
            self.training_history['train_time'] = time.time() - start_time
            self.training_history['n_epochs'] = self.iterations
            self.training_history['n_users'] = self.user_item_matrix.shape[0]
            self.training_history['n_items'] = self.user_item_matrix.shape[1]
            
            if self.verbose:
                logger.info(f"Training completed in {self.training_history['train_time']:.2f} seconds")
                logger.info(f"ALS model trained with {self.factors} factors over {self.iterations} iterations")
                logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.is_fitted = False
            raise
        
        return self
    
    def _initialize_model(self):
        """Initialize the ALS model with appropriate backend."""
        # Try GPU first if requested
        if self.use_gpu:
            if self._check_gpu_availability():
                try:
                    model = implicit.gpu.als.AlternatingLeastSquares(
                        factors=self.factors,
                        regularization=self.regularization,
                        iterations=self.iterations,
                        dtype=self.dtype,
                        random_state=self.random_state
                    )
                    if self.verbose:
                        logger.info("Using GPU acceleration")
                    return model
                except Exception as e:
                    logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
            else:
                logger.warning("GPU not available. Falling back to CPU.")
        
        # CPU fallback
        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            dtype=self.dtype,
            use_native=self.use_native,
            use_cg=self.use_cg,
            calculate_training_loss=self.calculate_training_loss,
            num_threads=self.num_threads,
            random_state=self.random_state
        )
        
        if self.verbose:
            logger.info("Using CPU implementation")
        
        return model
    
    def _validate_fitted(self) -> None:
        """Validate that the model is properly fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.model is None:
            raise ValueError("Model is not properly initialized")
        if not hasattr(self, 'user_to_idx') or not hasattr(self, 'item_to_idx'):
            raise ValueError("Model mappings are not available")
    
    def predict(self, user_ids: Union[int, List[int], np.ndarray],
                item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: User ID(s)
            item_ids: Item ID(s)
            
        Returns:
            Predicted ratings
            
        Raises:
            ValueError: If model is not fitted
        """
        self._validate_fitted()
        
        user_ids = self._validate_user_ids(user_ids)
        item_ids = self._validate_item_ids(item_ids)
        
        # Handle broadcasting
        user_ids, item_ids = self._broadcast_ids(user_ids, item_ids)
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            try:
                if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                    predictions.append(0.0)
                    continue
                
                user_idx = self.user_to_idx[user_id]
                item_idx = self.item_to_idx[item_id]
                
                # Get user and item factors
                user_factor = self.model.user_factors[user_idx]
                item_factor = self.model.item_factors[item_idx]
                
                # Ensure factors are in correct format
                if sparse.issparse(user_factor):
                    user_factor = user_factor.toarray().flatten()
                if sparse.issparse(item_factor):
                    item_factor = item_factor.toarray().flatten()
                
                # Compute dot product
                score = float(np.dot(user_factor, item_factor))
                predictions.append(score)
                
            except Exception as e:
                logger.warning(f"Error predicting for user {user_id}, item {item_id}: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def _broadcast_ids(self, user_ids: np.ndarray, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle broadcasting of user and item IDs."""
        if len(user_ids) == 1 and len(item_ids) > 1:
            user_ids = np.repeat(user_ids, len(item_ids))
        elif len(item_ids) == 1 and len(user_ids) > 1:
            item_ids = np.repeat(item_ids, len(user_ids))
        elif len(user_ids) != len(item_ids):
            raise ValueError(f"Incompatible lengths: {len(user_ids)} users, {len(item_ids)} items")
        
        return user_ids, item_ids
    
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
            
        Raises:
            ValueError: If model is not fitted
        """
        self._validate_fitted()
        user_ids = self._validate_user_ids(user_ids)
        
        all_recommendations = []
        all_scores = []
        
        for user_id in user_ids:
            try:
                if user_id not in self.user_to_idx:
                    # For new users, recommend popular items
                    recommendations = self._get_popular_items(n_recommendations)
                    scores = np.zeros(len(recommendations))
                else:
                    user_idx = self.user_to_idx[user_id]
                    
                    # Get recommendations from implicit
                    user_items = self.user_item_matrix[user_idx] if filter_seen else None
                    
                    recs, scores = self.model.recommend(
                        userid=user_idx,
                        user_items=user_items,
                        N=n_recommendations,
                        filter_already_liked_items=filter_seen
                    )
                    
                    # Convert to original IDs
                    recommendations = [self.idx_to_item[idx] for idx in recs]
                
                all_recommendations.append(recommendations)
                all_scores.append(scores)
                
            except Exception as e:
                logger.error(f"Error generating recommendations for user {user_id}: {e}")
                # Fallback to popular items
                recommendations = self._get_popular_items(n_recommendations)
                scores = np.zeros(len(recommendations))
                all_recommendations.append(recommendations)
                all_scores.append(scores)
        
        if return_scores:
            return np.array(all_recommendations, dtype=object), np.array(all_scores, dtype=object)
        else:
            return np.array(all_recommendations, dtype=object)
    
    def _get_popular_items(self, n_items: int) -> List[int]:
        """Get most popular items."""
        if self.user_item_matrix is None or n_items <= 0:
            return []
        
        try:
            # Calculate item popularity (sum of interactions)
            item_counts = np.array(self.user_item_matrix.sum(axis=0)).flatten()
            n_available = len(item_counts)
            n_items = min(n_items, n_available)
            
            if n_items == 0:
                return []
            
            popular_indices = np.argsort(item_counts)[-n_items:][::-1]
            return [self.idx_to_item[idx] for idx in popular_indices if idx in self.idx_to_item]
        except Exception as e:
            logger.error(f"Error getting popular items: {e}")
            return list(self.idx_to_item.values())[:n_items] if hasattr(self, 'idx_to_item') else []
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            Tuple of (item IDs, similarity scores)
            
        Raises:
            ValueError: If model is not fitted
        """
        self._validate_fitted()
        
        if item_id not in self.item_to_idx:
            return np.array([]), np.array([])
        
        try:
            item_idx = self.item_to_idx[item_id]
            
            # Get similar items from implicit
            similar_items, scores = self.model.similar_items(
                itemid=item_idx,
                N=n_similar + 1  # +1 because it includes the item itself
            )
            
            # Remove the item itself (first item is always the item itself)
            if len(similar_items) > 0:
                similar_items = similar_items[1:n_similar+1]
                scores = scores[1:n_similar+1]
            
            # Convert to original IDs
            similar_items = [self.idx_to_item[idx] for idx in similar_items if idx in self.idx_to_item]
            
            return np.array(similar_items), scores[:len(similar_items)]
            
        except Exception as e:
            logger.error(f"Error finding similar items for {item_id}: {e}")
            return np.array([]), np.array([])
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar users.
        
        Args:
            user_id: User ID
            n_similar: Number of similar users to return
            
        Returns:
            Tuple of (user IDs, similarity scores)
            
        Raises:
            ValueError: If model is not fitted
        """
        self._validate_fitted()
        
        if user_id not in self.user_to_idx:
            return np.array([]), np.array([])
        
        try:
            user_idx = self.user_to_idx[user_id]
            
            # Get similar users from implicit
            similar_users, scores = self.model.similar_users(
                userid=user_idx,
                N=n_similar + 1  # +1 because it includes the user itself
            )
            
            # Remove the user itself (first user is always the user itself)
            if len(similar_users) > 0:
                similar_users = similar_users[1:n_similar+1]
                scores = scores[1:n_similar+1]
            
            # Convert to original IDs
            similar_users = [self.idx_to_user[idx] for idx in similar_users if idx in self.idx_to_user]
            
            return np.array(similar_users), scores[:len(similar_users)]
            
        except Exception as e:
            logger.error(f"Error finding similar users for {user_id}: {e}")
            return np.array([]), np.array([])
    
    def get_user_factors(self) -> np.ndarray:
        """Get user latent factors."""
        self._validate_fitted()
        return self.model.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """Get item latent factors."""
        self._validate_fitted()
        return self.model.item_factors
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """
        Explain recommendation using factor contributions.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Explanation dictionary
        """
        if not self.is_fitted:
            return super().explain_recommendation(user_id, item_id)
        
        try:
            user_idx = self.user_to_idx.get(user_id)
            item_idx = self.item_to_idx.get(item_id)
            
            if user_idx is None or item_idx is None:
                return {
                    'user_id': user_id,
                    'item_id': item_id,
                    'error': 'User or item not found in training data',
                    'model': 'ALS'
                }
            
            # Get factors
            user_factor = self.model.user_factors[user_idx]
            item_factor = self.model.item_factors[item_idx]
            
            # Ensure factors are arrays
            if sparse.issparse(user_factor):
                user_factor = user_factor.toarray().flatten()
            if sparse.issparse(item_factor):
                item_factor = item_factor.toarray().flatten()
            
            # Calculate score
            score = float(np.dot(user_factor, item_factor))
            
            # Find top contributing factors
            factor_contributions = user_factor * item_factor
            top_factors_idx = np.argsort(np.abs(factor_contributions))[-5:][::-1]
            
            explanation = {
                'user_id': user_id,
                'item_id': item_id,
                'predicted_score': score,
                'model': 'ALS',
                'n_factors': self.factors,
                'top_contributing_factors': top_factors_idx.tolist(),
                'factor_contributions': factor_contributions[top_factors_idx].tolist(),
                'explanation': f"Score {score:.3f} based on {self.factors} latent factors. "
                             f"Top contributing factor: {top_factors_idx[0]} "
                             f"(contribution: {factor_contributions[top_factors_idx[0]]:.3f})"
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return {
                'user_id': user_id,
                'item_id': item_id,
                'error': str(e),
                'model': 'ALS'
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'model_name': 'ALS',
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'n_users': len(self.user_to_idx) if hasattr(self, 'user_to_idx') else 0,
            'n_items': len(self.item_to_idx) if hasattr(self, 'item_to_idx') else 0,
            'matrix_shape': self.user_item_matrix.shape if self.user_item_matrix is not None else None,
            'training_time': self.training_history.get('train_time', 0),
            'use_gpu': self.use_gpu and self._check_gpu_availability(),
        }
        
        return info