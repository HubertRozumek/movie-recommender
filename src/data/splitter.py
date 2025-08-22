"""Data splitting strategies for train/validation/test sets."""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/validation/test splits."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    
    @property
    def sizes(self) -> Dict[str, int]:
        """Get sizes of each split."""
        return {
            'train': len(self.train),
            'val': len(self.val),
            'test': len(self.test)
        }
    
    @property
    def proportions(self) -> Dict[str, float]:
        """Get proportions of each split."""
        total = len(self.train) + len(self.val) + len(self.test)
        return {
            'train': len(self.train) / total,
            'val': len(self.val) / total,
            'test': len(self.test) / total
        }


class BaseSplitter(ABC):
    """Base class for data splitters."""
    
    def __init__(self, 
                 train_size: float = 0.7,
                 val_size: float = 0.15,
                 test_size: float = 0.15,
                 random_state: int = 42):
        """
        Initialize splitter.
        
        Args:
            train_size: Proportion for training set
            val_size: Proportion for validation set
            test_size: Proportion for test set
            random_state: Random seed
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split proportions must sum to 1.0"
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        
    @abstractmethod
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split ratings into train/val/test.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        pass
    
    def _log_split_info(self, split: DataSplit):
        """Log information about the split."""
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {split.sizes['train']} ({split.proportions['train']:.2%})")
        logger.info(f"  Val: {split.sizes['val']} ({split.proportions['val']:.2%})")
        logger.info(f"  Test: {split.sizes['test']} ({split.proportions['test']:.2%})")


class RandomSplitter(BaseSplitter):
    """Random splitting of ratings."""
    
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Randomly split ratings.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        logger.info("Performing random split")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            ratings,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (self.train_size + self.val_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        split = DataSplit(train=train, val=val, test=test)
        self._log_split_info(split)
        
        return split


class TemporalSplitter(BaseSplitter):
    """Temporal splitting based on timestamps."""
    
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split ratings based on timestamp.
        
        Args:
            ratings: Ratings dataframe (must have 'timestamp' column)
            
        Returns:
            DataSplit object
        """
        if 'timestamp' not in ratings.columns:
            raise ValueError("Temporal split requires 'timestamp' column")
        
        logger.info("Performing temporal split")
        
        # Sort by timestamp
        ratings = ratings.sort_values('timestamp')
        
        # Calculate split points
        n = len(ratings)
        train_end = int(n * self.train_size)
        val_end = int(n * (self.train_size + self.val_size))
        
        # Split
        train = ratings.iloc[:train_end]
        val = ratings.iloc[train_end:val_end]
        test = ratings.iloc[val_end:]
        
        split = DataSplit(train=train, val=val, test=test)
        self._log_split_info(split)
        
        # Log temporal information
        logger.info(f"  Train period: {train['timestamp'].min()} to {train['timestamp'].max()}")
        logger.info(f"  Val period: {val['timestamp'].min()} to {val['timestamp'].max()}")
        logger.info(f"  Test period: {test['timestamp'].min()} to {test['timestamp'].max()}")
        
        return split


class UserBasedSplitter(BaseSplitter):
    """Split by user - ensures all users appear in train set."""
    
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split ratings ensuring all users in train.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        logger.info("Performing user-based split")
        
        train_list = []
        val_list = []
        test_list = []
        
        # Split each user's ratings
        for user_id, user_ratings in ratings.groupby('userId'):
            n = len(user_ratings)
            
            if n < 3:
                # If user has less than 3 ratings, all go to train
                train_list.append(user_ratings)
            else:
                # Shuffle user's ratings
                user_ratings = user_ratings.sample(frac=1, random_state=self.random_state)
                
                # Calculate split points
                train_end = int(n * self.train_size)
                val_end = int(n * (self.train_size + self.val_size))
                
                # Split
                train_list.append(user_ratings.iloc[:train_end])
                if val_end > train_end:
                    val_list.append(user_ratings.iloc[train_end:val_end])
                if val_end < n:
                    test_list.append(user_ratings.iloc[val_end:])
        
        # Combine
        train = pd.concat(train_list, ignore_index=True)
        val = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
        test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        
        split = DataSplit(train=train, val=val, test=test)
        self._log_split_info(split)
        
        return split


class LeaveOneOutSplitter(BaseSplitter):
    """Leave-one-out splitting for each user."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize leave-one-out splitter.
        
        Args:
            random_state: Random seed
        """
        # For leave-one-out, we don't use train/val/test sizes
        super().__init__(train_size=0.8, val_size=0.1, test_size=0.1, 
                        random_state=random_state)
        
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Leave one out for validation and one for test per user.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        logger.info("Performing leave-one-out split")
        
        train_list = []
        val_list = []
        test_list = []
        
        for user_id, user_ratings in ratings.groupby('userId'):
            n = len(user_ratings)
            
            if n < 3:
                # If user has less than 3 ratings, all go to train
                train_list.append(user_ratings)
            else:
                # Sort by timestamp if available
                if 'timestamp' in user_ratings.columns:
                    user_ratings = user_ratings.sort_values('timestamp')
                else:
                    # Random shuffle
                    user_ratings = user_ratings.sample(frac=1, random_state=self.random_state)
                
                # Leave last two out
                train_list.append(user_ratings.iloc[:-2])
                val_list.append(user_ratings.iloc[-2:-1])
                test_list.append(user_ratings.iloc[-1:])
        
        # Combine
        train = pd.concat(train_list, ignore_index=True)
        val = pd.concat(val_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)
        
        split = DataSplit(train=train, val=val, test=test)
        self._log_split_info(split)
        
        return split


class StratifiedSplitter(BaseSplitter):
    """Stratified splitting based on rating values."""
    
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split ratings with stratification by rating value.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        logger.info("Performing stratified split")
        
        # Create stratification column (binned ratings)
        ratings['rating_bin'] = pd.cut(ratings['rating'], bins=5, labels=False)
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            ratings,
            test_size=self.test_size,
            stratify=ratings['rating_bin'],
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (self.train_size + self.val_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val['rating_bin'],
            random_state=self.random_state
        )
        
        # Remove temporary column
        for df in [train, val, test]:
            df.drop('rating_bin', axis=1, inplace=True)
        
        split = DataSplit(train=train, val=val, test=test)
        self._log_split_info(split)
        
        return split


class ColdStartSplitter(BaseSplitter):
    """Split to simulate cold-start scenarios."""
    
    def __init__(self,
                 cold_start_ratio: float = 0.2,
                 cold_start_type: str = 'user',
                 **kwargs):
        """
        Initialize cold-start splitter.
        
        Args:
            cold_start_ratio: Ratio of cold-start users/items
            cold_start_type: 'user' or 'item' cold-start
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.cold_start_ratio = cold_start_ratio
        self.cold_start_type = cold_start_type
        
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split with cold-start simulation.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        logger.info(f"Performing cold-start split ({self.cold_start_type})")
        
        if self.cold_start_type == 'user':
            return self._user_cold_start_split(ratings)
        elif self.cold_start_type == 'item':
            return self._item_cold_start_split(ratings)
        else:
            raise ValueError(f"Unknown cold start type: {self.cold_start_type}")
    
    def _user_cold_start_split(self, ratings: pd.DataFrame) -> DataSplit:
        """Split with user cold-start."""
        # Select cold-start users
        unique_users = ratings['userId'].unique()
        n_cold_users = int(len(unique_users) * self.cold_start_ratio)
        
        np.random.seed(self.random_state)
        cold_users = np.random.choice(unique_users, n_cold_users, replace=False)
        
        # Separate cold and warm users
        cold_ratings = ratings[ratings['userId'].isin(cold_users)]
        warm_ratings = ratings[~ratings['userId'].isin(cold_users)]
        
        # Split warm users normally
        warm_splitter = RandomSplitter(
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state
        )
        warm_split = warm_splitter.split(warm_ratings)
        
        # Cold users go to test
        test = pd.concat([warm_split.test, cold_ratings], ignore_index=True)
        
        split = DataSplit(
            train=warm_split.train,
            val=warm_split.val,
            test=test
        )
        
        self._log_split_info(split)
        logger.info(f"  Cold-start users: {n_cold_users}")
        
        return split
    
    def _item_cold_start_split(self, ratings: pd.DataFrame) -> DataSplit:
        """Split with item cold-start."""
        # Select cold-start items
        unique_items = ratings['movieId'].unique()
        n_cold_items = int(len(unique_items) * self.cold_start_ratio)
        
        np.random.seed(self.random_state)
        cold_items = np.random.choice(unique_items, n_cold_items, replace=False)
        
        # Separate cold and warm items
        cold_ratings = ratings[ratings['movieId'].isin(cold_items)]
        warm_ratings = ratings[~ratings['movieId'].isin(cold_items)]
        
        # Split warm items normally
        warm_splitter = RandomSplitter(
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state
        )
        warm_split = warm_splitter.split(warm_ratings)
        
        # Cold items go to test
        test = pd.concat([warm_split.test, cold_ratings], ignore_index=True)
        
        split = DataSplit(
            train=warm_split.train,
            val=warm_split.val,
            test=test
        )
        
        self._log_split_info(split)
        logger.info(f"  Cold-start items: {n_cold_items}")
        
        return split


class DataSplitter:
    """Main class for data splitting with multiple strategies."""
    
    STRATEGIES = {
        'random': RandomSplitter,
        'temporal': TemporalSplitter,
        'user_based': UserBasedSplitter,
        'leave_one_out': LeaveOneOutSplitter,
        'stratified': StratifiedSplitter,
        'cold_start_user': lambda **kwargs: ColdStartSplitter(cold_start_type='user', **kwargs),
        'cold_start_item': lambda **kwargs: ColdStartSplitter(cold_start_type='item', **kwargs)
    }
    
    def __init__(self, strategy: str = 'random', **kwargs):
        """
        Initialize data splitter.
        
        Args:
            strategy: Splitting strategy name
            **kwargs: Arguments for specific splitter
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(self.STRATEGIES.keys())}")
        
        splitter_class = self.STRATEGIES[strategy]
        if callable(splitter_class):
            self.splitter = splitter_class(**kwargs)
        else:
            self.splitter = splitter_class(**kwargs)
        
    def split(self, ratings: pd.DataFrame) -> DataSplit:
        """
        Split ratings data.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            DataSplit object
        """
        return self.splitter.split(ratings)
    
    def get_k_folds(self, ratings: pd.DataFrame, n_splits: int = 5,
                    stratified: bool = False) -> list:
        """
        Get k-fold cross-validation splits.
        
        Args:
            ratings: Ratings dataframe
            n_splits: Number of folds
            stratified: Whether to use stratified k-fold
            
        Returns:
            List of DataSplit objects
        """
        logger.info(f"Creating {n_splits}-fold cross-validation splits")
        
        if stratified:
            # Create stratification column
            ratings['rating_bin'] = pd.cut(ratings['rating'], bins=5, labels=False)
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                               random_state=self.splitter.random_state)
            splits = kf.split(ratings, ratings['rating_bin'])
            ratings.drop('rating_bin', axis=1, inplace=True)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True,
                      random_state=self.splitter.random_state)
            splits = kf.split(ratings)
        
        fold_splits = []
        for fold, (train_val_idx, test_idx) in enumerate(splits):
            # Get train+val and test
            train_val = ratings.iloc[train_val_idx]
            test = ratings.iloc[test_idx]
            
            # Split train+val into train and val
            val_size = 0.15  # Fixed validation size
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                random_state=self.splitter.random_state
            )
            
            fold_split = DataSplit(train=train, val=val, test=test)
            fold_splits.append(fold_split)
            
            logger.info(f"Fold {fold + 1}: Train={len(train)}, "
                       f"Val={len(val)}, Test={len(test)}")
        
        return fold_splits