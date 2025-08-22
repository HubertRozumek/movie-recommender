"""Base class for recommendation models."""

import logging
import pickle
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

import pandas as pd
import numpy as np
from scipy import sparse
import mlflow

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models."""
    
    def __init__(self, 
                 model_name: str = "BaseRecommender",
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize base recommender.
        
        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose
        
        # Model state
        self.is_fitted = False
        self.n_users = None
        self.n_items = None
        
        # Mappings
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        # Training history
        self.training_history = {
            'train_time': None,
            'n_epochs': None,
            'metrics': {}
        }
        
        # Model parameters
        self.params = {}
        
        # Set random seed
        np.random.seed(random_state)
        
    @abstractmethod
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix], 
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'BaseRecommender':
        """
        Train the recommendation model.
        
        Args:
            train_data: Training data (DataFrame or sparse matrix)
            val_data: Validation data for early stopping (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def recommend(self, user_ids: Union[int, List[int], np.ndarray],
                  n_recommendations: int = 10,
                  filter_seen: bool = True,
                  return_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate top-N recommendations for users.
        
        Args:
            user_ids: User ID(s) to generate recommendations for
            n_recommendations: Number of recommendations per user
            filter_seen: Whether to filter out items user has already rated
            return_scores: Whether to return recommendation scores
            
        Returns:
            Array of recommended item IDs (and optionally scores)
        """
        pass
    
    def fit_predict(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
                   user_ids: Union[int, List[int], np.ndarray],
                   item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Fit model and make predictions.
        
        Args:
            train_data: Training data
            user_ids: User ID(s) for prediction
            item_ids: Item ID(s) for prediction
            
        Returns:
            Predicted ratings
        """
        self.fit(train_data)
        return self.predict(user_ids, item_ids)
    
    def _prepare_data(self, data: Union[pd.DataFrame, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Prepare data for model training.
        
        Args:
            data: Input data
            
        Returns:
            Sparse matrix representation
        """
        if isinstance(data, pd.DataFrame):
            # Assume DataFrame has columns: userId, movieId, rating
            if not self.user_to_idx:
                # Create mappings
                unique_users = data['userId'].unique()
                unique_items = data['movieId'].unique()
                
                self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
                self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
                self.item_to_idx = {i: j for j, i in enumerate(unique_items)}
                self.idx_to_item = {j: i for i, j in self.item_to_idx.items()}
                
                self.n_users = len(unique_users)
                self.n_items = len(unique_items)
            
            # Convert to sparse matrix
            user_indices = data['userId'].map(self.user_to_idx).values
            item_indices = data['movieId'].map(self.item_to_idx).values
            ratings = data['rating'].values
            
            matrix = sparse.csr_matrix(
                (ratings, (user_indices, item_indices)),
                shape=(self.n_users, self.n_items)
            )
            
            return matrix
        
        elif sparse.issparse(data):
            if not self.n_users:
                self.n_users, self.n_items = data.shape
                self.user_to_idx = {i: i for i in range(self.n_users)}
                self.idx_to_user = self.user_to_idx.copy()
                self.item_to_idx = {i: i for i in range(self.n_items)}
                self.idx_to_item = self.item_to_idx.copy()
            
            return data.tocsr()
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _validate_user_ids(self, user_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """Validate and convert user IDs to array."""
        if isinstance(user_ids, (int, np.integer)):
            user_ids = [user_ids]
        
        user_ids = np.asarray(user_ids)
        
        # Check if all users are known
        unknown_users = []
        for uid in user_ids:
            if uid not in self.user_to_idx:
                unknown_users.append(uid)
        
        if unknown_users:
            logger.warning(f"Unknown users: {unknown_users[:5]}{'...' if len(unknown_users) > 5 else ''}")
        
        return user_ids
    
    def _validate_item_ids(self, item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """Validate and convert item IDs to array."""
        if isinstance(item_ids, (int, np.integer)):
            item_ids = [item_ids]
        
        item_ids = np.asarray(item_ids)
        
        # Check if all items are known
        unknown_items = []
        for iid in item_ids:
            if iid not in self.item_to_idx:
                unknown_items.append(iid)
        
        if unknown_items:
            logger.warning(f"Unknown items: {unknown_items[:5]}{'...' if len(unknown_items) > 5 else ''}")
        
        return item_ids
    
    def save(self, path: Union[str, Path], save_format: str = 'pickle') -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            save_format: Format to save ('pickle', 'joblib', 'json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif save_format == 'joblib':
            import joblib
            joblib.dump(self, path)
        elif save_format == 'json':
            # Save only parameters (not the full model)
            model_dict = {
                'model_name': self.model_name,
                'params': self.params,
                'n_users': self.n_users,
                'n_items': self.n_items,
                'training_history': self.training_history
            }
            with open(path, 'w') as f:
                json.dump(model_dict, f, indent=2)
        else:
            raise ValueError(f"Unknown save format: {save_format}")
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], load_format: str = 'pickle') -> 'BaseRecommender':
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            load_format: Format to load ('pickle', 'joblib')
            
        Returns:
            Loaded model
        """
        path = Path(path)
        
        if load_format == 'pickle':
            with open(path, 'rb') as f:
                model = pickle.load(f)
        elif load_format == 'joblib':
            import joblib
            model = joblib.load(path)
        else:
            raise ValueError(f"Unknown load format: {load_format}")
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def log_to_mlflow(self, metrics: Dict[str, float] = None,
                     params: Dict[str, Any] = None,
                     artifacts: Dict[str, str] = None) -> None:
        """
        Log model to MLflow.
        
        Args:
            metrics: Metrics to log
            params: Parameters to log
            artifacts: Artifacts to log (name: path)
        """
        try:
            # Log parameters
            if params:
                mlflow.log_params(params)
            else:
                mlflow.log_params(self.params)
            
            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
            
            # Log training history
            if self.training_history['train_time']:
                mlflow.log_metric('train_time', self.training_history['train_time'])
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)
            
            # Log model
            mlflow.sklearn.log_model(self, "model")
            
            logger.info("Model logged to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar items.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            Tuple of (item IDs, similarity scores)
        """
        raise NotImplementedError("This model does not support item similarity")
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find similar users.
        
        Args:
            user_id: User ID
            n_similar: Number of similar users to return
            
        Returns:
            Tuple of (user IDs, similarity scores)
        """
        raise NotImplementedError("This model does not support user similarity")
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict[str, Any]:
        """
        Explain why an item was recommended to a user.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with explanation details
        """
        return {
            'user_id': user_id,
            'item_id': item_id,
            'predicted_rating': self.predict([user_id], [item_id])[0],
            'explanation': 'No explanation available for this model type'
        }
    
    def get_model_size(self) -> int:
        """
        Get model size in bytes.
        
        Returns:
            Model size in bytes
        """
        import sys
        return sys.getsizeof(self)
    
    def get_inference_time(self, n_predictions: int = 1000) -> float:
        """
        Measure average inference time.
        
        Args:
            n_predictions: Number of predictions to average over
            
        Returns:
            Average inference time in seconds
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Generate random user-item pairs
        user_ids = np.random.choice(list(self.user_to_idx.keys()), n_predictions)
        item_ids = np.random.choice(list(self.item_to_idx.keys()), n_predictions)
        
        # Measure time
        start_time = time.time()
        self.predict(user_ids, item_ids)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_predictions
        return avg_time
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.model_name}(n_users={self.n_users}, n_items={self.n_items}, fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()