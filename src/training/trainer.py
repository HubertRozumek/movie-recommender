"""Model trainer for recommendation systems."""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from tqdm import tqdm

from src.models.base_model import BaseRecommender
from src.evaluation.metrics import RecommenderMetrics
from src.data.splitter import DataSplitter

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Unified trainer for all recommendation models."""
    
    def __init__(self,
                 model: BaseRecommender,
                 metrics: Optional[RecommenderMetrics] = None,
                 mlflow_tracking: bool = True,
                 experiment_name: str = "movielens-benchmark",
                 save_dir: Union[str, Path] = "results/models"):
        """
        Initialize model trainer.
        
        Args:
            model: Recommendation model to train
            metrics: Metrics calculator
            mlflow_tracking: Whether to use MLflow tracking
            experiment_name: MLflow experiment name
            save_dir: Directory to save models
        """
        self.model = model
        self.metrics = metrics or RecommenderMetrics()
        self.mlflow_tracking = mlflow_tracking
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': {},
            'training_time': 0,
            'best_epoch': 0
        }
        
        # Setup MLflow
        if self.mlflow_tracking:
            self._setup_mlflow()
    
    def train(self,
             train_data: pd.DataFrame,
             val_data: Optional[pd.DataFrame] = None,
             test_data: Optional[pd.DataFrame] = None,
             epochs: int = 1,
             early_stopping: bool = True,
             patience: int = 5,
             monitor_metric: str = 'rmse',
             mode: str = 'min',
             verbose: bool = True,
             **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            epochs: Number of training epochs (for iterative models)
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            monitor_metric: Metric to monitor for early stopping
            mode: 'min' or 'max' for the monitored metric
            verbose: Whether to print progress
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        if verbose:
            logger.info(f"Starting training for {self.model.model_name}")
        
        # Start MLflow run
        if self.mlflow_tracking:
            mlflow.start_run(run_name=f"{self.model.model_name}_{time.strftime('%Y%m%d_%H%M%S')}")
            self._log_params()
        
        try:
            # Training loop
            start_time = time.time()
            best_metric = float('inf') if mode == 'min' else float('-inf')
            patience_counter = 0
            val_metrics = None
             
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Train model
                if epoch == 0 or hasattr(self.model, 'partial_fit'):
                    if verbose:
                        logger.info(f"Epoch {epoch + 1}/{epochs}")
                    
                    if hasattr(self.model, 'partial_fit'):
                        # For models that support incremental learning
                        self.model.partial_fit(train_data)
                    else:
                        # For models that train in one go
                        self.model.fit(train_data, val_data)
                
                # Evaluate on validation set
                if val_data is not None:
                    val_metrics = self._evaluate(val_data, prefix="val")
                    self.history['val_metrics'].append(val_metrics)
                    
                    # Check for improvement
                    # Make sure the metric name has the correct prefix
                    if monitor_metric.startswith("val_"):
                        metric_name = monitor_metric
                    else:
                        metric_name = f"val_{monitor_metric}"
                    
                    # Check if the metric exists in val_metrics
                    if metric_name in val_metrics:
                        current_metric = val_metrics[metric_name]
                        
                        if self._is_better(current_metric, best_metric, mode):
                            best_metric = current_metric
                            patience_counter = 0
                            self.history['best_epoch'] = epoch
                            
                            # Save best model
                            self._save_model(suffix="best")
                            
                            if verbose:
                                logger.info(f"Validation {monitor_metric} improved to {current_metric:.4f}")
                        else:
                            patience_counter += 1
                            
                            if early_stopping and patience_counter >= patience:
                                if verbose:
                                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                                break
                    else:
                        # Fallback if metric not found
                        if verbose:
                            logger.warning(f"Metric {metric_name} not found in validation metrics. Available: {list(val_metrics.keys())}")
                        early_stopping = False  # Disable early stopping if metric not available
                    
                    # Log to MLflow
                    if self.mlflow_tracking:
                        for metric_name, value in val_metrics.items():
                            if not np.isnan(value):  # Only log valid values
                                mlflow.log_metric(metric_name, value, step=epoch)
                
                # Evaluate on training set (sample for efficiency)
                train_sample = train_data.sample(min(10000, len(train_data)), random_state=42)
                train_metrics = self._evaluate(train_sample, prefix="train")
                self.history['train_metrics'].append(train_metrics)
                
                if self.mlflow_tracking:
                    for metric_name, value in train_metrics.items():
                        if not np.isnan(value):  # Only log valid values
                            mlflow.log_metric(metric_name, value, step=epoch)
                
                epoch_time = time.time() - epoch_start
                
                if verbose:
                    self._print_epoch_summary(epoch, train_metrics, val_metrics if val_data is not None else None, epoch_time)
            
            # Training completed
            self.history['training_time'] = time.time() - start_time
            
            # Load best model if early stopping was used
            if early_stopping and val_data is not None and patience_counter > 0:
                best_model_path = self.save_dir / f"{self.model.model_name}_best.pkl"
                if best_model_path.exists():
                    try:
                        self.model = BaseRecommender.load(best_model_path)
                        if verbose:
                            logger.info(f"Loaded best model from epoch {self.history['best_epoch'] + 1}")
                    except Exception as e:
                        logger.warning(f"Could not load best model: {e}")
            
            # Final evaluation on test set
            if test_data is not None:
                test_metrics = self._evaluate(test_data, prefix="test")
                self.history['test_metrics'] = test_metrics
                
                if self.mlflow_tracking:
                    for metric_name, value in test_metrics.items():
                        if not np.isnan(value):  # Only log valid values
                            mlflow.log_metric(metric_name, value)
                
                if verbose:
                    logger.info("\nTest Set Results:")
                    self.metrics.print_metrics(
                        {k.replace("test_", ""): v for k, v in test_metrics.items() if not np.isnan(v)},
                        title=f"Test Results - {self.model.model_name}"
                    )
            
            # Save final model
            self._save_model(suffix="final")
            
            # Log model to MLflow
            if self.mlflow_tracking:
                try:
                    mlflow.sklearn.log_model(self.model, "model")
                    mlflow.log_metric("training_time", self.history['training_time'])
                except Exception as e:
                    logger.warning(f"Could not log model to MLflow: {e}")
            
            if verbose:
                logger.info(f"Training completed in {self.history['training_time']:.2f} seconds")
            
            return self.history
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # End MLflow run
            if self.mlflow_tracking:
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def _evaluate(self, data: pd.DataFrame, prefix: str = "") -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            data: Evaluation data
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Get predictions for rating metrics
            y_true = data['rating'].values
            y_pred = self.model.predict(data['userId'].values, data['movieId'].values)
            
            # Handle NaN predictions
            valid_mask = ~np.isnan(y_pred)
            if not np.any(valid_mask):
                logger.warning("All predictions are NaN")
                return {f"{prefix}_rmse" if prefix else "rmse": np.nan,
                        f"{prefix}_mae" if prefix else "mae": np.nan}
            
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            # Calculate rating metrics
            metrics = self.metrics.calculate_rating_metrics(y_true_valid, y_pred_valid)
            
            # Get recommendations for ranking metrics (with error handling)
            try:
                unique_users = data['userId'].unique()[:100]  # Sample for efficiency
                recommendations, _ = self.model.recommend(unique_users, n_recommendations=20, return_scores=True)
                
                # Create user-items dictionary for ranking metrics
                user_items = {}
                for user in unique_users:
                    user_data = data[data['userId'] == user]
                    # Consider items with rating >= 4 as relevant
                    relevant_items = user_data[user_data['rating'] >= 4]['movieId'].tolist()
                    if relevant_items:
                        user_items[user] = relevant_items
                
                # Calculate ranking metrics
                if user_items and recommendations:
                    ranking_metrics = self.metrics.calculate_ranking_metrics(recommendations, user_items)
                    metrics.update(ranking_metrics)
            except Exception as e:
                logger.warning(f"Could not calculate ranking metrics: {e}")
                # Add default ranking metrics
                metrics.update({
                    'precision_at_10': np.nan,
                    'recall_at_10': np.nan,
                    'ndcg_at_10': np.nan
                })
            
            # Add prefix
            if prefix:
                metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return default metrics with NaN values
            default_metrics = {
                'rmse': np.nan,
                'mae': np.nan,
                'precision_at_10': np.nan,
                'recall_at_10': np.nan,
                'ndcg_at_10': np.nan
            }
            if prefix:
                default_metrics = {f"{prefix}_{k}": v for k, v in default_metrics.items()}
            return default_metrics
    
    def _is_better(self, current: float, best: float, mode: str) -> bool:
        """Check if current metric is better than best."""
        if np.isnan(current) or np.isnan(best):
            return not np.isnan(current) and np.isnan(best)
        
        if mode == 'min':
            return current < best
        else:
            return current > best
    
    def _save_model(self, suffix: str = ""):
        """Save model to disk."""
        try:
            model_name = f"{self.model.model_name}_{suffix}" if suffix else self.model.model_name
            model_path = self.save_dir / f"{model_name}.pkl"
            self.model.save(model_path)
            
            # Save training history
            history_path = self.save_dir / f"{model_name}_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                history_serializable = {}
                for key, value in self.history.items():
                    if isinstance(value, (list, dict)):
                        history_serializable[key] = self._make_serializable(value)
                    else:
                        history_serializable[key] = value
                json.dump(history_serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif np.isnan(obj) if isinstance(obj, (int, float)) else False:
            return None
        else:
            return obj
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.mlflow_tracking = False
    
    def _log_params(self):
        """Log model parameters to MLflow."""
        if not self.mlflow_tracking:
            return
        
        try:
            # Log model parameters
            if hasattr(self.model, 'params'):
                mlflow.log_params(self.model.params)
            
            # Log trainer parameters
            mlflow.log_param("model_name", self.model.model_name)
            if hasattr(self.model, 'random_state'):
                mlflow.log_param("random_state", self.model.random_state)
            
        except Exception as e:
            logger.warning(f"Failed to log parameters to MLflow: {e}")
    
    def _print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict], epoch_time: float):
        """Print summary of epoch results."""
        summary = f"Epoch {epoch + 1} - Time: {epoch_time:.2f}s"
        
        # Add key metrics
        if train_metrics:
            rmse = train_metrics.get('train_rmse')
            if rmse is not None and not np.isnan(rmse):
                summary += f" - Train RMSE: {rmse:.4f}"
        
        if val_metrics:
            rmse = val_metrics.get('val_rmse')
            precision = val_metrics.get('val_precision_at_10')
            if rmse is not None and not np.isnan(rmse):
                summary += f" - Val RMSE: {rmse:.4f}"
            if precision is not None and not np.isnan(precision):
                summary += f" - Val Precision@10: {precision:.4f}"
        
        print(summary)
    
    def cross_validate(self,
                      data: pd.DataFrame,
                      n_splits: int = 5,
                      stratified: bool = False,
                      **train_kwargs) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            data: Full dataset
            n_splits: Number of CV folds
            stratified: Whether to use stratified splitting
            **train_kwargs: Arguments to pass to train()
            
        Returns:
            Dictionary with lists of metrics for each fold
        """
        logger.info(f"Starting {n_splits}-fold cross-validation for {self.model.model_name}")
        
        # Create splitter
        splitter = DataSplitter(strategy='random')
        folds = splitter.get_k_folds(data, n_splits=n_splits, stratified=stratified)
        
        cv_results = {
            'fold_metrics': [],
            'avg_metrics': {},
            'std_metrics': {}
        }
        
        for fold_idx, fold_data in enumerate(folds):
            logger.info(f"Training fold {fold_idx + 1}/{n_splits}")
            
            # Train on fold
            fold_history = self.train(
                train_data=fold_data.train,
                val_data=fold_data.val,
                test_data=fold_data.test,
                verbose=False,
                **train_kwargs
            )
            
            # Store fold results
            cv_results['fold_metrics'].append(fold_history['test_metrics'])
        
        # Calculate average and std metrics
        all_metrics = {}
        for fold_metrics in cv_results['fold_metrics']:
            for metric_name, value in fold_metrics.items():
                if not np.isnan(value):  # Skip NaN values
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        for metric_name, values in all_metrics.items():
            if values:  # Only if we have valid values
                cv_results['avg_metrics'][metric_name] = np.mean(values)
                cv_results['std_metrics'][metric_name] = np.std(values)
        
        # Print CV results
        logger.info("\nCross-Validation Results:")
        logger.info("="*50)
        for metric_name in sorted(cv_results['avg_metrics'].keys()):
            avg = cv_results['avg_metrics'][metric_name]
            std = cv_results['std_metrics'][metric_name]
            logger.info(f"{metric_name:30s}: {avg:.4f} (+/- {std:.4f})")
        
        return cv_results