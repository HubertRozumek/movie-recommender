"""SVD (Singular Value Decomposition) model implementation."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class SVDModel(BaseRecommender):
    """SVD matrix factorization model using Surprise library."""
    
    def __init__(self,
                 n_factors: int = 100,
                 n_epochs: int = 30,
                 biased: bool = True,
                 init_mean: float = 0,
                 init_std_dev: float = 0.1,
                 lr_all: float = 0.005,
                 reg_all: float = 0.02,
                 lr_bu: Optional[float] = None,
                 lr_bi: Optional[float] = None,
                 lr_pu: Optional[float] = None,
                 lr_qi: Optional[float] = None,
                 reg_bu: Optional[float] = None,
                 reg_bi: Optional[float] = None,
                 reg_pu: Optional[float] = None,
                 reg_qi: Optional[float] = None,
                 **kwargs):
        """
        Initialize SVD model.
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Number of SGD epochs
            biased: Whether to use baselines (biases)
            init_mean: Mean of normal distribution for initialization
            init_std_dev: Standard deviation for initialization
            lr_all: Learning rate for all parameters
            reg_all: Regularization for all parameters
            lr_bu: Learning rate for user biases
            lr_bi: Learning rate for item biases
            lr_pu: Learning rate for user factors
            lr_qi: Learning rate for item factors
            reg_bu: Regularization for user biases
            reg_bi: Regularization for item biases
            reg_pu: Regularization for user factors
            reg_qi: Regularization for item factors
            **kwargs: Additional arguments for base class
        """
        super().__init__(model_name="SVD", **kwargs)
        
        # Model parameters
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        
        # Learning rates
        self.lr_all = lr_all
        self.lr_bu = lr_bu or lr_all
        self.lr_bi = lr_bi or lr_all
        self.lr_pu = lr_pu or lr_all
        self.lr_qi = lr_qi or lr_all
        
        # Regularization
        self.reg_all = reg_all
        self.reg_bu = reg_bu or reg_all
        self.reg_bi = reg_bi or reg_all
        self.reg_pu = reg_pu or reg_all
        self.reg_qi = reg_qi or reg_all
        
        # Surprise model
        self.model = None
        self.trainset = None
        
        # Initialize mappings
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
        # Store parameters
        self.params = {
            'n_factors': n_factors,
            'n_epochs': n_epochs,
            'biased': biased,
            'lr_all': lr_all,
            'reg_all': reg_all
        }
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'SVDModel':
        """
        Train SVD model.
        
        Args:
            train_data: Training data
            val_data: Validation data for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        if self.verbose:
            logger.info("Training SVD model...")
        
        start_time = time.time()
        
        # Convert data to Surprise format
        self.trainset = self._prepare_surprise_data(train_data)
        
        # Initialize SVD model
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            biased=self.biased,
            init_mean=self.init_mean,
            init_std_dev=self.init_std_dev,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            lr_bu=self.lr_bu,
            lr_bi=self.lr_bi,
            lr_pu=self.lr_pu,
            lr_qi=self.lr_qi,
            reg_bu=self.reg_bu,
            reg_bi=self.reg_bi,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Train model
        self.model.fit(self.trainset)
        
        # Store user and item mappings
        self._store_mappings()
        
        # Mark as fitted
        self.is_fitted = True
        
        # Record training time
        self.training_history['train_time'] = time.time() - start_time
        self.training_history['n_epochs'] = self.n_epochs
        
        if self.verbose:
            logger.info(f"Training completed in {self.training_history['train_time']:.2f} seconds")
            logger.info(f"Model trained with {self.n_factors} factors over {self.n_epochs} epochs")
        
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
        
        user_ids = np.atleast_1d(user_ids)
        item_ids = np.atleast_1d(item_ids)
        
        # Ensure same length
        if len(user_ids) == 1:
            user_ids = np.repeat(user_ids, len(item_ids))
        elif len(item_ids) == 1:
            item_ids = np.repeat(item_ids, len(user_ids))
        
        predictions = []
        
        for user_id, item_id in zip(user_ids, item_ids):
            # Surprise expects string IDs internally, but we work with ints
            pred = self.model.predict(uid=str(user_id), iid=str(item_id))
            predictions.append(pred.est)
        
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
        user_ids = np.atleast_1d(user_ids)
        
        all_recommendations = []
        all_scores = []
        
        for user_id in user_ids:
            # Get all items
            all_items = list(self.item_to_idx.keys())
            
            # Filter seen items if needed
            if filter_seen and str(user_id) in [self.trainset.to_raw_uid(u) for u in self.trainset.all_users()]:
                try:
                    # Get items rated by user from trainset
                    user_inner_id = self.trainset.to_inner_uid(str(user_id))
                    seen_items = [int(self.trainset.to_raw_iid(i)) 
                                 for i, _ in self.trainset.ur[user_inner_id]]
                    candidate_items = [i for i in all_items if i not in seen_items]
                except:
                    # If user not in trainset, use all items
                    candidate_items = all_items
            else:
                candidate_items = all_items
            
            if len(candidate_items) == 0:
                recommendations = []
                scores = []
            else:
                # Predict scores for all candidate items
                predictions = []
                for item_id in candidate_items:
                    pred = self.model.predict(uid=str(user_id), iid=str(item_id))
                    predictions.append((item_id, pred.est))
                
                # Sort by predicted rating
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-N
                top_n = predictions[:n_recommendations]
                recommendations = [item_id for item_id, _ in top_n]
                scores = [score for _, score in top_n]
            
            all_recommendations.append(recommendations)
            all_scores.append(scores)
        
        if return_scores:
            return np.array(all_recommendations, dtype=object), np.array(all_scores, dtype=object)
        else:
            return np.array(all_recommendations, dtype=object)
    
    def _prepare_surprise_data(self, data: Union[pd.DataFrame, sparse.spmatrix]):
        """Convert data to Surprise Dataset format."""
        if isinstance(data, pd.DataFrame):
            df = data[['userId', 'movieId', 'rating']].copy()
            df['userId'] = df['userId'].astype(str)
            df['movieId'] = df['movieId'].astype(str)

            reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
            surprise_data = Dataset.load_from_df(df, reader)
            trainset = surprise_data.build_full_trainset()

        elif sparse.issparse(data):
            coo = data.tocoo()
            df = pd.DataFrame({
                'userId': coo.row.astype(str),
                'movieId': coo.col.astype(str),
                'rating': coo.data
            })
            reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
            surprise_data = Dataset.load_from_df(df, reader)
            trainset = surprise_data.build_full_trainset()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Create mappings for easier access
        self.user_to_idx = {int(trainset.to_raw_uid(u)): u for u in trainset.all_users()}
        self.item_to_idx = {int(trainset.to_raw_iid(i)): i for i in trainset.all_items()}
        
        self.n_users = trainset.n_users
        self.n_items = trainset.n_items
        
        return trainset
        
    def _store_mappings(self):
        """Store user and item mappings from trainset."""
        # Create reverse mappings
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
    
    def get_user_factors(self) -> np.ndarray:
        """
        Get user latent factors.
        
        Returns:
            User factors matrix (n_users x n_factors)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.pu
    
    def get_item_factors(self) -> np.ndarray:
        """
        Get item latent factors.
        
        Returns:
            Item factors matrix (n_items x n_factors)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.qi
    
    def get_user_biases(self) -> np.ndarray:
        """
        Get user biases.
        
        Returns:
            User biases array
        """
        if not self.is_fitted or not self.biased:
            return None
        
        return self.model.bu
    
    def get_item_biases(self) -> np.ndarray:
        """
        Get item biases.
        
        Returns:
            Item biases array
        """
        if not self.is_fitted or not self.biased:
            return None
        
        return self.model.bi
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items based on latent factors.
        
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
        
        # Get item's inner ID
        item_inner_id = self.trainset.to_inner_iid(str(item_id))
        
        # Get item factor
        item_factor = self.model.qi[item_inner_id]
        
        # Compute similarities with all items
        similarities = self.model.qi.dot(item_factor)
        
        # Get top-N similar items (excluding self)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        # Convert back to raw IDs
        similar_items = [int(self.trainset.to_raw_iid(idx)) for idx in similar_indices]
        similarity_scores = similarities[similar_indices]
        
        return np.array(similar_items), similarity_scores
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar users based on latent factors.
        
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
        
        # Get user's inner ID
        user_inner_id = self.trainset.to_inner_uid(str(user_id))
        
        # Get user factor
        user_factor = self.model.pu[user_inner_id]
        
        # Compute similarities with all users
        similarities = self.model.pu.dot(user_factor)
        
        # Get top-N similar users (excluding self)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        # Convert back to raw IDs
        similar_users = [int(self.trainset.to_raw_uid(idx)) for idx in similar_indices]
        similarity_scores = similarities[similar_indices]
        
        return np.array(similar_users), similarity_scores
    
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
        
        pred = self.predict([user_id], [item_id])[0]
        
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': pred,
            'model': 'SVD',
            'n_factors': self.n_factors
        }
        
        if self.biased:
            try:
                user_inner_id = self.trainset.to_inner_uid(str(user_id))
                item_inner_id = self.trainset.to_inner_iid(str(item_id))
                
                explanation['components'] = {
                    'global_mean': self.trainset.global_mean,
                    'user_bias': self.model.bu[user_inner_id],
                    'item_bias': self.model.bi[item_inner_id],
                    'interaction': pred - self.trainset.global_mean - 
                                 self.model.bu[user_inner_id] - self.model.bi[item_inner_id]
                }
                
                explanation['explanation'] = (
                    f"Predicted rating {pred:.2f} based on:\n"
                    f"- Global average: {explanation['components']['global_mean']:.2f}\n"
                    f"- User preference: {explanation['components']['user_bias']:.2f}\n"
                    f"- Item popularity: {explanation['components']['item_bias']:.2f}\n"
                    f"- User-item interaction: {explanation['components']['interaction']:.2f}"
                )
            except:
                explanation['explanation'] = f"Predicted rating {pred:.2f} based on {self.n_factors} latent factors"
        else:
            explanation['explanation'] = f"Predicted rating {pred:.2f} based on {self.n_factors} latent factors"
        
        return explanation