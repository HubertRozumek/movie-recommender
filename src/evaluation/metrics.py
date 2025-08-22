"""Evaluation metrics for recommendation systems."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class RecommenderMetrics:
    """Comprehensive metrics for evaluating recommendation systems."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize metrics calculator.
        
        Args:
            k_values: List of k values for ranking metrics
        """
        self.k_values = k_values
        self.metrics_history = []
    
    def calculate_all_metrics(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             recommendations: Optional[np.ndarray] = None,
                             user_items: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            recommendations: Recommended items per user (for ranking metrics)
            user_items: Dictionary of true items per user (for ranking metrics)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Rating prediction metrics
        metrics.update(self.calculate_rating_metrics(y_true, y_pred))
        
        # Ranking metrics (if recommendations provided)
        if recommendations is not None and user_items is not None:
            metrics.update(self.calculate_ranking_metrics(recommendations, user_items))
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_rating_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate rating prediction metrics.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            Dictionary of rating metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['rmse'] = self.rmse(y_true, y_pred)
        metrics['mae'] = self.mae(y_true, y_pred)
        metrics['mse'] = self.mse(y_true, y_pred)
        
        # Additional metrics
        metrics['r2'] = self.r2_score(y_true, y_pred)
        metrics['pearson_corr'] = self.pearson_correlation(y_true, y_pred)
        metrics['spearman_corr'] = self.spearman_correlation(y_true, y_pred)
        
        return metrics
    
    def calculate_ranking_metrics(self,
                                 recommendations: np.ndarray,
                                 user_items: Dict[int, List[int]]) -> Dict[str, float]:
        """
        Calculate ranking metrics.
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            
        Returns:
            Dictionary of ranking metrics
        """
        metrics = {}
        
        for k in self.k_values:
            # Precision@k
            precision = self.precision_at_k(recommendations, user_items, k)
            metrics[f'precision_at_{k}'] = precision
            
            # Recall@k
            recall = self.recall_at_k(recommendations, user_items, k)
            metrics[f'recall_at_{k}'] = recall
            
            # F1@k
            f1 = self.f1_at_k(precision, recall)
            metrics[f'f1_at_{k}'] = f1
            
            # NDCG@k
            ndcg = self.ndcg_at_k(recommendations, user_items, k)
            metrics[f'ndcg_at_{k}'] = ndcg
            
            # MAP@k
            map_score = self.map_at_k(recommendations, user_items, k)
            metrics[f'map_at_{k}'] = map_score
            
            # Hit Rate@k
            hit_rate = self.hit_rate_at_k(recommendations, user_items, k)
            metrics[f'hit_rate_at_{k}'] = hit_rate
        
        # Coverage
        metrics['coverage'] = self.coverage(recommendations)
        
        # Diversity
        metrics['diversity'] = self.diversity(recommendations)
        
        # Novelty
        metrics['novelty'] = self.novelty(recommendations, user_items)
        
        return metrics
    
    # Rating Prediction Metrics
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))
    
    def pearson_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if len(y_true) < 2:
            return 0.0
        corr, _ = stats.pearsonr(y_true, y_pred)
        return corr if not np.isnan(corr) else 0.0
    
    def spearman_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Spearman rank correlation coefficient."""
        if len(y_true) < 2:
            return 0.0
        corr, _ = stats.spearmanr(y_true, y_pred)
        return corr if not np.isnan(corr) else 0.0
    
    # Ranking Metrics
    
    def precision_at_k(self,
                      recommendations: np.ndarray,
                      user_items: Dict[int, List[int]],
                      k: int) -> float:
        """
        Precision at k.
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            k: Number of top recommendations to consider
            
        Returns:
            Average precision@k across users
        """
        precisions = []
        
        for user_id, true_items in user_items.items():
            if user_id >= len(recommendations):
                continue
                
            rec_items = recommendations[user_id][:k]
            if len(rec_items) == 0:
                continue
                
            n_relevant = len(set(rec_items) & set(true_items))
            precisions.append(n_relevant / len(rec_items))
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(self,
                   recommendations: np.ndarray,
                   user_items: Dict[int, List[int]],
                   k: int) -> float:
        """
        Recall at k.
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            k: Number of top recommendations to consider
            
        Returns:
            Average recall@k across users
        """
        recalls = []
        
        for user_id, true_items in user_items.items():
            if user_id >= len(recommendations) or len(true_items) == 0:
                continue
                
            rec_items = recommendations[user_id][:k]
            n_relevant = len(set(rec_items) & set(true_items))
            recalls.append(n_relevant / len(true_items))
        
        return np.mean(recalls) if recalls else 0.0
    
    def f1_at_k(self, precision: float, recall: float) -> float:
        """F1 score at k."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self,
                 recommendations: np.ndarray,
                 user_items: Dict[int, List[int]],
                 k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at k.
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            k: Number of top recommendations to consider
            
        Returns:
            Average NDCG@k across users
        """
        ndcgs = []
        
        for user_id, true_items in user_items.items():
            if user_id >= len(recommendations):
                continue
                
            rec_items = recommendations[user_id][:k]
            if len(rec_items) == 0:
                continue
            
            # Calculate DCG
            dcg = 0.0
            for i, item in enumerate(rec_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts from 0
            
            # Calculate IDCG (ideal DCG)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def map_at_k(self,
                recommendations: np.ndarray,
                user_items: Dict[int, List[int]],
                k: int) -> float:
        """
        Mean Average Precision at k.
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            k: Number of top recommendations to consider
            
        Returns:
            MAP@k across users
        """
        aps = []
        
        for user_id, true_items in user_items.items():
            if user_id >= len(recommendations):
                continue
                
            rec_items = recommendations[user_id][:k]
            if len(rec_items) == 0:
                continue
            
            # Calculate Average Precision for this user
            n_relevant = 0
            sum_precision = 0.0
            
            for i, item in enumerate(rec_items):
                if item in true_items:
                    n_relevant += 1
                    sum_precision += n_relevant / (i + 1)
            
            if n_relevant > 0:
                aps.append(sum_precision / min(len(true_items), k))
        
        return np.mean(aps) if aps else 0.0
    
    def hit_rate_at_k(self,
                     recommendations: np.ndarray,
                     user_items: Dict[int, List[int]],
                     k: int) -> float:
        """
        Hit Rate at k (percentage of users with at least one relevant item in top-k).
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of true relevant items per user
            k: Number of top recommendations to consider
            
        Returns:
            Hit rate@k
        """
        hits = 0
        n_users = 0
        
        for user_id, true_items in user_items.items():
            if user_id >= len(recommendations):
                continue
                
            rec_items = recommendations[user_id][:k]
            if len(set(rec_items) & set(true_items)) > 0:
                hits += 1
            n_users += 1
        
        return hits / n_users if n_users > 0 else 0.0
    
    def coverage(self, recommendations: np.ndarray) -> float:
        """
        Catalog coverage (percentage of items that are recommended).
        
        Args:
            recommendations: Recommended items per user
            
        Returns:
            Coverage percentage
        """
        all_items = set()
        recommended_items = set()
        
        for recs in recommendations:
            for item in recs:
                all_items.add(item)
                recommended_items.add(item)
        
        # This assumes we know the total catalog size
        # In practice, you'd pass this as a parameter
        return len(recommended_items) / len(all_items) if all_items else 0.0
    
    def diversity(self, recommendations: np.ndarray, 
                 item_features: Optional[np.ndarray] = None) -> float:
        """
        Average diversity of recommendations (1 - average similarity between recommended items).
        
        Args:
            recommendations: Recommended items per user
            item_features: Optional item feature matrix for similarity calculation
            
        Returns:
            Average diversity score
        """
        diversities = []
        
        for recs in recommendations:
            if len(recs) < 2:
                continue
            
            # Simple diversity: percentage of unique items
            diversity = len(set(recs)) / len(recs)
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def novelty(self, recommendations: np.ndarray,
               user_items: Dict[int, List[int]]) -> float:
        """
        Average novelty of recommendations (recommending less popular items).
        
        Args:
            recommendations: Recommended items per user
            user_items: Dictionary of items users have already interacted with
            
        Returns:
            Average novelty score
        """
        # Count item popularity
        item_counts = {}
        for items in user_items.values():
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        if not item_counts:
            return 0.0
        
        max_count = max(item_counts.values())
        novelties = []
        
        for user_id, recs in enumerate(recommendations):
            user_novelty = []
            for item in recs:
                # Novelty as inverse of popularity
                popularity = item_counts.get(item, 0) / max_count
                user_novelty.append(1 - popularity)
            
            if user_novelty:
                novelties.append(np.mean(user_novelty))
        
        return np.mean(novelties) if novelties else 0.0
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics display
        """
        print("\n" + "="*50)
        print(f" {title} ".center(50))
        print("="*50)
        
        # Group metrics by type
        rating_metrics = ['rmse', 'mae', 'mse', 'r2', 'pearson_corr', 'spearman_corr']
        
        print("\nRating Prediction Metrics:")
        print("-"*30)
        for metric in rating_metrics:
            if metric in metrics:
                print(f"  {metric.upper():15s}: {metrics[metric]:.4f}")
        
        print("\nRanking Metrics:")
        print("-"*30)
        for metric, value in metrics.items():
            if metric not in rating_metrics:
                print(f"  {metric:15s}: {value:.4f}")
        
        print("="*50 + "\n")
    
    def compare_models(self, model_metrics: Dict[str, Dict[str, float]]):
        """
        Compare metrics across multiple models.
        
        Args:
            model_metrics: Dictionary of model names to their metrics
        """
        # Create comparison DataFrame
        df = pd.DataFrame(model_metrics).T
        
        print("\n" + "="*60)
        print(" Model Comparison ".center(60))
        print("="*60)
        print(df.round(4))
        print("="*60)
        
        # Find best model for each metric
        print("\nBest Models by Metric:")
        print("-"*40)
        for metric in df.columns:
            if metric in ['rmse', 'mae', 'mse']:
                # Lower is better
                best_model = df[metric].idxmin()
            else:
                # Higher is better
                best_model = df[metric].idxmax()
            
            print(f"  {metric:20s}: {best_model} ({df.loc[best_model, metric]:.4f})")
        
        return df