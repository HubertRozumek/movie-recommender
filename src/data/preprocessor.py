"""Data preprocessing for MovieLens dataset."""

import logging
from typing import Optional, Tuple, Dict, List
import warnings

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from src.data.data_loader import MovieLensDataset

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess MovieLens data for recommendation models."""
    
    def __init__(self, 
                 min_user_ratings: int = 20,
                 min_item_ratings: int = 5,
                 normalize_ratings: bool = True,
                 remove_outliers: bool = True,
                 outlier_threshold: float = 3.5):
        """
        Initialize preprocessor.
        
        Args:
            min_user_ratings: Minimum ratings per user
            min_item_ratings: Minimum ratings per item
            normalize_ratings: Whether to normalize ratings
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outliers
        """
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        self.normalize_ratings = normalize_ratings
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.rating_scaler = None
        
        self.original_stats = {}
        self.processed_stats = {}
        
    def fit_transform(self, dataset: MovieLensDataset) -> MovieLensDataset:
        """
        Fit preprocessor and transform dataset.
        
        Args:
            dataset: Raw MovieLensDataset
            
        Returns:
            Preprocessed MovieLensDataset
        """
        logger.info("Starting data preprocessing...")
        
        # Store original statistics
        self.original_stats = dataset.get_statistics()
        
        # Create a copy to avoid modifying original
        ratings = dataset.ratings.copy()
        movies = dataset.movies.copy()
        users = dataset.users.copy() if dataset.users is not None else None
        tags = dataset.tags.copy() if dataset.tags is not None else None
        links = dataset.links.copy() if dataset.links is not None else None
        
        # Filter by minimum ratings
        ratings = self._filter_by_min_ratings(ratings)
        
        # Remove outliers
        if self.remove_outliers:
            ratings = self._remove_outliers(ratings)
        
        # Handle missing values
        ratings = self._handle_missing_values(ratings)
        movies = self._handle_missing_values(movies)
        
        # Encode user and item IDs
        ratings = self._encode_ids(ratings)
        
        # Normalize ratings
        if self.normalize_ratings:
            ratings = self._normalize_ratings(ratings)
        
        # Process movie features
        movies = self._process_movie_features(movies)
        
        # Process user features if available
        if users is not None:
            users = self._process_user_features(users)
        
        # Process tags if available
        if tags is not None:
            tags = self._process_tags(tags)
        
        # Create processed dataset
        processed_dataset = MovieLensDataset(
            ratings=ratings,
            movies=movies,
            users=users,
            tags=tags,
            links=links
        )
        
        # Store processed statistics
        self.processed_stats = processed_dataset.get_statistics()
        
        # Log preprocessing results
        self._log_preprocessing_results()
        
        return processed_dataset
    
    def transform(self, dataset: MovieLensDataset) -> MovieLensDataset:
        """
        Transform dataset using fitted preprocessor.
        
        Args:
            dataset: MovieLensDataset to transform
            
        Returns:
            Transformed MovieLensDataset
        """
        # Create a copy
        ratings = dataset.ratings.copy()
        movies = dataset.movies.copy()
        users = dataset.users.copy() if dataset.users is not None else None
        tags = dataset.tags.copy() if dataset.tags is not None else None
        links = dataset.links.copy() if dataset.links is not None else None
        
        # Apply transformations
        ratings = self._transform_ratings(ratings)
        movies = self._process_movie_features(movies)
        
        if users is not None:
            users = self._process_user_features(users)
        
        if tags is not None:
            tags = self._process_tags(tags)
        
        return MovieLensDataset(
            ratings=ratings,
            movies=movies,
            users=users,
            tags=tags,
            links=links
        )
    
    def _filter_by_min_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items by minimum number of ratings."""
        logger.info(f"Filtering by min ratings: users>={self.min_user_ratings}, "
                   f"items>={self.min_item_ratings}")
        
        initial_count = len(ratings)
        
        # Iterative filtering until convergence
        prev_count = 0
        iteration = 0
        
        while prev_count != len(ratings):
            prev_count = len(ratings)
            
            # Filter users
            user_counts = ratings['userId'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            ratings = ratings[ratings['userId'].isin(valid_users)]
            
            # Filter items
            item_counts = ratings['movieId'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_ratings].index
            ratings = ratings[ratings['movieId'].isin(valid_items)]
            
            iteration += 1
            
        logger.info(f"Filtering completed in {iteration} iterations. "
                   f"Removed {initial_count - len(ratings)} ratings "
                   f"({(initial_count - len(ratings)) / initial_count * 100:.2f}%)")
        
        return ratings
    
    def _remove_outliers(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Remove outlier ratings based on z-score."""
        logger.info(f"Removing outliers with z-score > {self.outlier_threshold}")
        
        # Calculate z-scores for each user's ratings
        outlier_mask = pd.Series(False, index=ratings.index)
        
        for user_id in ratings['userId'].unique():
            user_ratings = ratings[ratings['userId'] == user_id]['rating']
            if len(user_ratings) > 3:  # Need enough ratings for statistics
                z_scores = np.abs((user_ratings - user_ratings.mean()) / user_ratings.std())
                user_outliers = z_scores > self.outlier_threshold
                outlier_mask[user_ratings.index[user_outliers]] = True
        
        n_outliers = outlier_mask.sum()
        ratings = ratings[~outlier_mask]
        
        logger.info(f"Removed {n_outliers} outlier ratings "
                   f"({n_outliers / len(outlier_mask) * 100:.2f}%)")
        
        return ratings
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe."""
        if df.isnull().any().any():
            logger.info(f"Handling {df.isnull().sum().sum()} missing values")
            
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode or 'unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if not df[col].empty:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', 
                                  inplace=True)
        
        return df
    
    def _encode_ids(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Encode user and item IDs to consecutive integers."""
        logger.info("Encoding user and item IDs")
        
        # Fit encoders
        ratings['user_idx'] = self.user_encoder.fit_transform(ratings['userId'])
        ratings['item_idx'] = self.item_encoder.fit_transform(ratings['movieId'])
        
        logger.info(f"Encoded {len(self.user_encoder.classes_)} users and "
                   f"{len(self.item_encoder.classes_)} items")
        
        return ratings
    
    def _normalize_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Normalize ratings."""
        logger.info("Normalizing ratings")
        
        # Store original ratings
        ratings['rating_original'] = ratings['rating'].copy()
        
        # Initialize scaler
        self.rating_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Normalize
        ratings['rating_normalized'] = self.rating_scaler.fit_transform(
            ratings[['rating']].values
        ).flatten()
        
        # Also compute mean-centered ratings
        user_means = ratings.groupby('userId')['rating'].transform('mean')
        ratings['rating_centered'] = ratings['rating'] - user_means
        
        return ratings
    
    def _transform_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Transform ratings using fitted preprocessor."""
        # Encode IDs
        ratings['user_idx'] = self.user_encoder.transform(ratings['userId'])
        ratings['item_idx'] = self.item_encoder.transform(ratings['movieId'])
        
        # Normalize if needed
        if self.normalize_ratings and self.rating_scaler:
            ratings['rating_normalized'] = self.rating_scaler.transform(
                ratings[['rating']].values
            ).flatten()
        
        return ratings
    
    def _process_movie_features(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Process movie features."""
        logger.info("Processing movie features")
        
        # Process genres
        if 'genres' in movies.columns:
            movies = self._process_genres(movies)
        
        # Process year
        if 'year' in movies.columns:
            movies['year'].fillna(movies['year'].median(), inplace=True)
            movies['decade'] = (movies['year'] // 10) * 10
            
            # Create age feature
            current_year = pd.Timestamp.now().year
            movies['age'] = current_year - movies['year']
        
        # Process title
        if 'title' in movies.columns:
            # Extract features from title
            movies['title_length'] = movies['title'].str.len()
            movies['title_words'] = movies['title'].str.split().str.len()
        
        return movies
    
    def _process_genres(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Process genre information."""
        # Split genres
        genre_lists = movies['genres'].str.split('|')
        
        # Get all unique genres
        all_genres = set()
        for genres in genre_lists:
            if isinstance(genres, list):
                all_genres.update(genres)
        
        # Create binary columns for each genre
        for genre in all_genres:
            if genre and genre != '(no genres listed)':
                movies[f'genre_{genre}'] = movies['genres'].str.contains(
                    genre, na=False
                ).astype(int)
        
        # Count genres per movie
        movies['n_genres'] = genre_lists.apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        return movies
    
    def _process_user_features(self, users: pd.DataFrame) -> pd.DataFrame:
        """Process user features."""
        logger.info("Processing user features")
        
        # Process age groups
        if 'age' in users.columns:
            users['age_group'] = pd.cut(
                users['age'],
                bins=[0, 18, 25, 35, 45, 55, 100],
                labels=['<18', '18-25', '26-35', '36-45', '46-55', '56+']
            )
        
        # Process gender
        if 'gender' in users.columns:
            users['gender_encoded'] = LabelEncoder().fit_transform(users['gender'])
        
        # Process occupation
        if 'occupation' in users.columns:
            users['occupation_encoded'] = LabelEncoder().fit_transform(users['occupation'])
        
        return users
    
    def _process_tags(self, tags: pd.DataFrame) -> pd.DataFrame:
        """Process tags data."""
        logger.info("Processing tags")
        
        # Convert to lowercase
        tags['tag'] = tags['tag'].str.lower()
        
        # Remove special characters
        tags['tag'] = tags['tag'].str.replace('[^a-z0-9\s]', '', regex=True)
        
        # Count tag frequency
        tag_counts = tags['tag'].value_counts()
        
        # Filter rare tags
        min_tag_count = 5
        frequent_tags = tag_counts[tag_counts >= min_tag_count].index
        tags = tags[tags['tag'].isin(frequent_tags)]
        
        logger.info(f"Kept {len(frequent_tags)} frequent tags")
        
        return tags
    
    def _log_preprocessing_results(self):
        """Log preprocessing results."""
        logger.info("\n" + "="*50)
        logger.info("Preprocessing Results")
        logger.info("="*50)
        
        logger.info("Original dataset:")
        for key, value in self.original_stats.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        logger.info("\nProcessed dataset:")
        for key, value in self.processed_stats.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        logger.info("="*50)
    
    def create_interaction_matrix(self, ratings: pd.DataFrame,
                                 form: str = 'csr') -> sparse.spmatrix:
        """
        Create sparse interaction matrix.
        
        Args:
            ratings: Ratings dataframe
            form: Sparse matrix format ('csr', 'coo', 'dok')
            
        Returns:
            Sparse interaction matrix
        """
        n_users = ratings['user_idx'].max() + 1
        n_items = ratings['item_idx'].max() + 1
        
        if form == 'csr':
            matrix = sparse.csr_matrix(
                (ratings['rating'].values,
                 (ratings['user_idx'].values, ratings['item_idx'].values)),
                shape=(n_users, n_items)
            )
        elif form == 'coo':
            matrix = sparse.coo_matrix(
                (ratings['rating'].values,
                 (ratings['user_idx'].values, ratings['item_idx'].values)),
                shape=(n_users, n_items)
            )
        elif form == 'dok':
            matrix = sparse.dok_matrix((n_users, n_items))
            for _, row in ratings.iterrows():
                matrix[row['user_idx'], row['item_idx']] = row['rating']
        else:
            raise ValueError(f"Unknown sparse matrix format: {form}")
        
        return matrix
    
    def get_user_item_mappings(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Get user and item ID mappings.
        
        Returns:
            Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
        """
        user_to_idx = dict(zip(self.user_encoder.classes_, 
                              range(len(self.user_encoder.classes_))))
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        
        item_to_idx = dict(zip(self.item_encoder.classes_,
                              range(len(self.item_encoder.classes_))))
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        
        return user_to_idx, idx_to_user, item_to_idx, idx_to_item