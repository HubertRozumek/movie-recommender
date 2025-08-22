"""Neural Collaborative Filtering (NCF) model implementation."""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class NCFModel(BaseRecommender):
    """Neural Collaborative Filtering model using TensorFlow."""
    
    def __init__(self,
                 embedding_size: int = 64,
                 hidden_layers: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 batch_norm: bool = True,
                 learning_rate: float = 0.0005,
                 batch_size: int = 256,
                 epochs: int = 50,
                 num_negatives: int = 4,
                 l2_regularization: float = 0.0001,
                 **kwargs):
        """
        Initialize NCF model.
        
        Args:
            embedding_size: Size of embedding vectors.
            hidden_layers: List of hidden layer sizes for MLP.
            dropout_rate: Dropout rate.
            activation: Activation function.
            use_bias: Whether to use bias in layers.
            batch_norm: Whether to use batch normalization.
            learning_rate: Learning rate for optimizer.
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            num_negatives: Number of negative samples per positive.
            l2_regularization: L2 regularization factor.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(model_name="NCF", **kwargs)
        
        self.embedding_size = embedding_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_negatives = num_negatives
        self.l2_regularization = l2_regularization 
        
        # Model components
        self.model = None
        self.gmf_model = None
        self.mlp_model = None
        self.neumf_model = None
        
        # Training data
        self.train_matrix = None
        self.all_items = None
        
        # Store parameters
        self.params = {
            'embedding_size': embedding_size,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'l2_regularization': l2_regularization
        }
    
    def _build_gmf_model(self) -> Model:
        """Build Generalized Matrix Factorization model."""
        user_input = layers.Input(shape=(1,), name='gmf_user_input')
        item_input = layers.Input(shape=(1,), name='gmf_item_input')
        
        user_embedding = layers.Embedding(
            self.n_users,
            self.embedding_size,
            embeddings_regularizer=l2(self.l2_regularization),
            name='gmf_user_embedding'
        )(user_input)
        
        item_embedding = layers.Embedding(
            self.n_items,
            self.embedding_size,
            embeddings_regularizer=l2(self.l2_regularization),
            name='gmf_item_embedding'
        )(item_input)
        
        user_vector = layers.Flatten(name='gmf_user_flatten')(user_embedding)
        item_vector = layers.Flatten(name='gmf_item_flatten')(item_embedding)
        
        gmf_vector = layers.Multiply(name='gmf_multiply')([user_vector, item_vector])
        
        return Model(inputs=[user_input, item_input], outputs=gmf_vector, name='GMF')
    
    def _build_mlp_model(self) -> Model:
        """Build Multi-Layer Perceptron model."""
        user_input = layers.Input(shape=(1,), name='mlp_user_input')
        item_input = layers.Input(shape=(1,), name='mlp_item_input')
        
        user_embedding = layers.Embedding(
            self.n_users,
            self.embedding_size,
            embeddings_regularizer=l2(self.l2_regularization),
            name='mlp_user_embedding'
        )(user_input)
        
        item_embedding = layers.Embedding(
            self.n_items,
            self.embedding_size,
            embeddings_regularizer=l2(self.l2_regularization),
            name='mlp_item_embedding'
        )(item_input)
        
        user_vector = layers.Flatten(name='mlp_user_flatten')(user_embedding)
        item_vector = layers.Flatten(name='mlp_item_flatten')(item_embedding)
        
        mlp_vector = layers.Concatenate(name='mlp_concat')([user_vector, item_vector])
        
        for i, units in enumerate(self.hidden_layers):
            mlp_vector = layers.Dense(
                units,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_regularizer=l2(self.l2_regularization),
                name=f'mlp_layer_{i}'
            )(mlp_vector)
            
            if self.batch_norm:
                mlp_vector = layers.BatchNormalization(name=f'mlp_bn_{i}')(mlp_vector)
            
            if self.dropout_rate > 0:
                mlp_vector = layers.Dropout(self.dropout_rate, name=f'mlp_dropout_{i}')(mlp_vector)
        
        return Model(inputs=[user_input, item_input], outputs=mlp_vector, name='MLP')
    
    def _build_neumf_model(self) -> Model:
        """Build Neural Matrix Factorization model (GMF + MLP)."""
        self.gmf_model = self._build_gmf_model()
        self.mlp_model = self._build_mlp_model()
        
        user_input = layers.Input(shape=(1,), name='user_input')
        item_input = layers.Input(shape=(1,), name='item_input')
        
        gmf_output = self.gmf_model([user_input, item_input])
        mlp_output = self.mlp_model([user_input, item_input])
        
        concat_vector = layers.Concatenate(name='neumf_concat')([gmf_output, mlp_output])
        
        output = layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=l2(self.l2_regularization),
            name='prediction'
        )(concat_vector)
        
        return Model(inputs=[user_input, item_input], outputs=output, name='NeuMF')
    
    def fit(self, train_data: Union[pd.DataFrame, sparse.spmatrix],
            val_data: Optional[Union[pd.DataFrame, sparse.spmatrix]] = None,
            **kwargs) -> 'NCFModel':
        """Train NCF model."""
        if self.verbose:
            logger.info("Training NCF model...")
        
        start_time = time.time()
        
        self.train_matrix = self._prepare_data(train_data)
        self.all_items = set(range(self.n_items))
        
        self.neumf_model = self._build_neumf_model()
        
        self.neumf_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        n_positive_samples = len(self.train_matrix.data)
        steps_per_epoch = (n_positive_samples * (1 + self.num_negatives)) // self.batch_size
        
        train_generator = self._generate_training_data()
        
        val_generator = None
        validation_steps = None
        if val_data is not None:
            val_matrix = self._prepare_data(val_data)
            val_generator = self._generate_validation_data(val_matrix)
            validation_steps = (val_matrix.nnz * 2) // self.batch_size
        
        callbacks = []
        if val_data is not None:
            callbacks.extend([
                EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True,
                    verbose=1 if self.verbose else 0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
                    verbose=1 if self.verbose else 0
                )
            ])
        
        history = self.neumf_model.fit(
            train_generator,
            steps_per_epoch=max(1, steps_per_epoch),
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=max(1, validation_steps) if validation_steps else None,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        self.model = self.neumf_model
        self.is_fitted = True
        
        self.training_history['train_time'] = time.time() - start_time
        self.training_history['n_epochs'] = len(history.history['loss'])
        
        try:
            self.training_history['metrics'] = {
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history.get('accuracy', [0])[-1],
                'final_auc': history.history.get('auc', [0])[-1]
            }
            if 'val_loss' in history.history:
                self.training_history['metrics'].update({
                    'final_val_loss': history.history.get('val_loss', [None])[-1],
                    'final_val_accuracy': history.history.get('val_accuracy', [0])[-1],
                    'final_val_auc': history.history.get('val_auc', [0])[-1]
                })
        except (KeyError, IndexError) as e:
            logger.warning(f"Could not access training metrics: {e}")
            self.training_history['metrics'] = {'final_loss': 0.0, 'final_accuracy': 0.0, 'final_auc': 0.0}

        if self.verbose:
            logger.info(f"Training completed in {self.training_history['train_time']:.2f} seconds")
            logger.info(f"Final loss: {self.training_history['metrics']['final_loss']:.4f}")
        
        return self
    
    def predict(self, user_ids: Union[int, List[int], np.ndarray],
                item_ids: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Predict interaction scores for user-item pairs.
        
        Args:
            user_ids: User ID(s).
            item_ids: Item ID(s).
            
        Returns:
            Predicted interaction scores (probabilities between 0 and 1).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        user_ids = self._validate_user_ids(user_ids)
        item_ids = self._validate_item_ids(item_ids)
        
        if len(user_ids) != len(item_ids):
            if len(user_ids) == 1 and len(item_ids) > 1:
                user_ids = np.repeat(user_ids, len(item_ids))
            elif len(item_ids) == 1 and len(user_ids) > 1:
                item_ids = np.repeat(item_ids, len(user_ids))
            else:
                raise ValueError("user_ids and item_ids must have the same length or one must be scalar")
        
        user_indices, item_indices = [], []
        valid_indices_mask = []
        for i, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            user_idx = self.user_to_idx.get(uid)
            item_idx = self.item_to_idx.get(iid)
            if user_idx is not None and item_idx is not None:
                user_indices.append(user_idx)
                item_indices.append(item_idx)
                valid_indices_mask.append(i)
        
        predictions = np.full(len(user_ids), np.nan)
        
        if valid_indices_mask:
            user_indices = np.array(user_indices)
            item_indices = np.array(item_indices)
            
            raw_predictions = self.model.predict(
                [user_indices, item_indices],
                verbose=0
            ).flatten()
            
            predictions[valid_indices_mask] = raw_predictions

        return predictions

    def recommend(self, user_ids: Union[int, List[int], np.ndarray],
                  n_recommendations: int = 10,
                  filter_seen: bool = True,
                  return_scores: bool = False) -> Union[List, Tuple[List, List]]:
        """Generate top-N recommendations for users."""
        user_ids = self._validate_user_ids(user_ids)
        
        all_recommendations = []
        all_scores = []
        
        for user_id in user_ids:
            if user_id not in self.user_to_idx:
                recommendations = self._get_popular_items(n_recommendations)
                scores = np.zeros(len(recommendations))
            else:
                user_idx = self.user_to_idx[user_id]
                
                if filter_seen and self.train_matrix is not None:
                    seen_items = set(self.train_matrix[user_idx].nonzero()[1])
                    candidate_items = list(self.all_items - seen_items)
                else:
                    candidate_items = list(self.all_items)
                
                if not candidate_items:
                    recommendations, scores = [], []
                else:
                    user_array = np.full(len(candidate_items), user_idx)
                    item_array = np.array(candidate_items)
                    
                    scores_pred = self.model.predict(
                        [user_array, item_array], verbose=0, batch_size=1024
                    ).flatten()
                    
                    n_recs = min(n_recommendations, len(scores_pred))
                    top_indices = np.argsort(scores_pred)[-n_recs:][::-1]
                    
                    recommendations = [self.idx_to_item[candidate_items[i]] for i in top_indices]
                    scores = scores_pred[top_indices]
            
            all_recommendations.append(recommendations)
            all_scores.append(scores)
        
        if len(user_ids) == 1:
            all_recommendations = all_recommendations[0]
            all_scores = all_scores[0]
        
        return (all_recommendations, all_scores) if return_scores else all_recommendations
    
    def _generate_training_data(self):
        """Generate training data with negative sampling."""
        users, items = self.train_matrix.nonzero()
        user_seen_items = {u: set(self.train_matrix.getrow(u).indices) for u in np.unique(users)}
        
        while True:
            indices = np.random.permutation(len(users))
            
            for start_idx in range(0, len(users), self.batch_size // (1 + self.num_negatives)):
                end_idx = min(start_idx + self.batch_size // (1 + self.num_negatives), len(users))
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) == 0: continue

                batch_users, batch_items, batch_labels = [], [], []
                
                for idx in batch_indices:
                    user, item = users[idx], items[idx]
                    
                    batch_users.append(user)
                    batch_items.append(item)
                    batch_labels.append(1)
                    
                    seen = user_seen_items.get(user, set())
                    
                    n_candidates = self.num_negatives * 5
                    neg_candidates = np.random.randint(0, self.n_items, size=n_candidates)
                    neg_samples = [x for x in neg_candidates if x not in seen][:self.num_negatives]
                    
                    for neg_item in neg_samples:
                        batch_users.append(user)
                        batch_items.append(neg_item)
                        batch_labels.append(0)

                yield ([np.array(batch_users), np.array(batch_items)], np.array(batch_labels))

    def _generate_validation_data(self, val_matrix):
        """Generate validation data with one negative sample per positive."""
        users, items = val_matrix.nonzero()
        user_seen_items = {u: set(self.train_matrix.getrow(u).indices) for u in np.unique(users)}

        while True:
            indices = np.random.permutation(len(users))

            for i in range(0, len(users), self.batch_size // 2):
                batch_indices = indices[i : i + self.batch_size // 2]
                if len(batch_indices) == 0: continue

                batch_users, batch_items, batch_labels = [], [], []

                for idx in batch_indices:
                    user, item = users[idx], items[idx]
                    
                    # Positive sample
                    batch_users.append(user)
                    batch_items.append(item)
                    batch_labels.append(1)

                    # Negative sample
                    seen = user_seen_items.get(user, set())
                    neg_item = np.random.randint(self.n_items)
                    while neg_item in seen:
                        neg_item = np.random.randint(self.n_items)
                    
                    batch_users.append(user)
                    batch_items.append(neg_item)
                    batch_labels.append(0)

                yield ([np.array(batch_users), np.array(batch_items)], np.array(batch_labels))
    
    def _get_popular_items(self, n_items: int) -> List[int]:
        """Get most popular items."""
        if self.train_matrix is None or self.train_matrix.nnz == 0:
            return list(self.idx_to_item.values())[:n_items]
        
        item_counts = np.array(self.train_matrix.sum(axis=0)).flatten()
        n_items = min(n_items, len(item_counts))
        popular_indices = np.argsort(item_counts)[-n_items:][::-1]
        return [self.idx_to_item[idx] for idx in popular_indices]