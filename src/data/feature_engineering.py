"""Feature engineering for recommendation models."""

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Create features for recommendation models."""
    
    def __init__(self,
                 create_user_features: bool = True,
                 create_item_features: bool = True,
                 create_interaction_features: bool = True,
                 use_dimensionality_reduction: bool = False,
                 n_components: int = 50):
        """
        Initialize feature engineer.
        
        Args:
            create_user_features: Whether to create user features
            create_item_features: Whether to create item features
            create_interaction_features: Whether to create interaction features
            use_dimensionality_reduction: Whether to reduce feature dimensions
            n_components: Number of components for dimensionality reduction
        """
        self.create_user_features = create_user_features
        self.create_item_features = create_item_features
        self.create_interaction_features = create_interaction_features
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.n_components = n_components
        
        self.user_feature_names = []
        self.item_feature_names = []
        self.interaction_feature_names = []
        
        self.tfidf_vectorizer = None
        self.user_pca = None
        self.item_pca = None
        
    def fit_transform(self,
                     ratings: pd.DataFrame,
                     movies: pd.DataFrame,
                     users: Optional[pd.DataFrame] = None,
                     tags: Optional[pd.DataFrame] = None) -> Dict:
        """
        Fit feature engineer and transform data.
        
        Args:
            ratings: Ratings dataframe
            movies: Movies dataframe
            users: Users dataframe (optional)
            tags: Tags dataframe (optional)
            
        Returns:
            Dictionary with feature matrices
        """
        logger.info("Starting feature engineering...")
        
        features = {}
        
        # Create user features
        if self.create_user_features:
            user_features = self._create_user_features(ratings, users, tags)
            features['user_features'] = user_features
            logger.info(f"Created {user_features.shape[1]} user features")
        
        # Create item features
        if self.create_item_features:
            item_features = self._create_item_features(ratings, movies, tags)
            features['item_features'] = item_features
            logger.info(f"Created {item_features.shape[1]} item features")
        
        # Create interaction features
        if self.create_interaction_features:
            interaction_features = self._create_interaction_features(ratings)
            features['interaction_features'] = interaction_features
            logger.info(f"Created {len(interaction_features)} interaction features")
        
        # Apply dimensionality reduction if needed
        if self.use_dimensionality_reduction:
            features = self._apply_dimensionality_reduction(features)
        
        return features
    
    def transform(self,
                 ratings: pd.DataFrame,
                 movies: pd.DataFrame,
                 users: Optional[pd.DataFrame] = None,
                 tags: Optional[pd.DataFrame] = None) -> Dict:
        """
        Transform data using fitted feature engineer.
        
        Args:
            ratings: Ratings dataframe
            movies: Movies dataframe
            users: Users dataframe (optional)
            tags: Tags dataframe (optional)
            
        Returns:
            Dictionary with feature matrices
        """
        features = {}
        
        if self.create_user_features:
            user_features = self._transform_user_features(ratings, users, tags)
            features['user_features'] = user_features
        
        if self.create_item_features:
            item_features = self._transform_item_features(ratings, movies, tags)
            features['item_features'] = item_features
        
        if self.create_interaction_features:
            interaction_features = self._create_interaction_features(ratings)
            features['interaction_features'] = interaction_features
        
        return features
    
    def _create_user_features(self,
                            ratings: pd.DataFrame,
                            users: Optional[pd.DataFrame],
                            tags: Optional[pd.DataFrame]) -> np.ndarray:
        """Create user features."""
        user_ids = ratings['userId'].unique()
        n_users = len(user_ids)
        features_list = []
        
        # Rating statistics per user
        user_stats = ratings.groupby('userId').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'timestamp': ['min', 'max'] if 'timestamp' in ratings.columns else []
        }).fillna(0)
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        features_list.append(user_stats.values)
        self.user_feature_names.extend(user_stats.columns.tolist())
        
        # Rating distribution features
        rating_dist = ratings.pivot_table(
            index='userId',
            columns='rating',
            values='movieId',
            aggfunc='count',
            fill_value=0
        )
        rating_dist = rating_dist.div(rating_dist.sum(axis=1), axis=0)  # Normalize
        features_list.append(rating_dist.values)
        self.user_feature_names.extend([f'rating_dist_{r}' for r in rating_dist.columns])
        
        # Temporal features
        if 'timestamp' in ratings.columns:
            temporal_features = self._create_temporal_user_features(ratings)
            features_list.append(temporal_features)
        
        # Demographic features
        if users is not None:
            demo_features = self._create_demographic_features(users, user_ids)
            features_list.append(demo_features)
        
        # Tag-based features
        if tags is not None:
            tag_features = self._create_user_tag_features(tags, user_ids)
            features_list.append(tag_features)
        
        # Genre preferences
        genre_prefs = self._create_genre_preferences(ratings, user_ids)
        if genre_prefs is not None:
            features_list.append(genre_prefs)
        
        # Concatenate all features
        user_features = np.hstack(features_list)
        
        # Handle any NaN values
        user_features = np.nan_to_num(user_features, 0)
        
        return user_features
    
    def _create_item_features(self,
                            ratings: pd.DataFrame,
                            movies: pd.DataFrame,
                            tags: Optional[pd.DataFrame]) -> np.ndarray:
        """Create item features."""
        item_ids = ratings['movieId'].unique()
        n_items = len(item_ids)
        features_list = []
        
        # Rating statistics per item
        item_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max']
        }).fillna(0)
        item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns.values]
        features_list.append(item_stats.values)
        self.item_feature_names.extend(item_stats.columns.tolist())
        
        # Popularity features
        popularity = self._create_popularity_features(ratings, item_ids)
        features_list.append(popularity)
        
        # Content features from movies
        if movies is not None:
            content_features = self._create_content_features(movies, item_ids)
            features_list.append(content_features)
        
        # Tag-based features
        if tags is not None:
            tag_features = self._create_item_tag_features(tags, item_ids)
            features_list.append(tag_features)
        
        # Temporal features
        if 'timestamp' in ratings.columns:
            temporal_features = self._create_temporal_item_features(ratings)
            features_list.append(temporal_features)
        
        # Concatenate all features
        item_features = np.hstack(features_list)
        
        # Handle any NaN values
        item_features = np.nan_to_num(item_features, 0)
        
        return item_features
    
    def _create_interaction_features(self, ratings: pd.DataFrame) -> Dict:
        """Create interaction-level features."""
        interaction_features = {}
        
        # Time since user's first rating
        if 'timestamp' in ratings.columns:
            user_first_rating = ratings.groupby('userId')['timestamp'].min()
            ratings['time_since_first'] = (
                ratings['timestamp'] - ratings['userId'].map(user_first_rating)
            ).dt.total_seconds() / 86400  # Convert to days
            interaction_features['time_since_first'] = ratings['time_since_first'].values
        
        # User rating minus user mean
        user_means = ratings.groupby('userId')['rating'].mean()
        ratings['rating_vs_user_mean'] = (
            ratings['rating'] - ratings['userId'].map(user_means)
        )
        interaction_features['rating_vs_user_mean'] = ratings['rating_vs_user_mean'].values
        
        # Item rating minus item mean
        item_means = ratings.groupby('movieId')['rating'].mean()
        ratings['rating_vs_item_mean'] = (
            ratings['rating'] - ratings['movieId'].map(item_means)
        )
        interaction_features['rating_vs_item_mean'] = ratings['rating_vs_item_mean'].values
        
        # User-item interaction count (how many times user rated similar items)
        item_similarity_count = self._calculate_item_similarity_count(ratings)
        interaction_features['item_similarity_count'] = item_similarity_count
        
        # Rating position in user's history
        ratings['user_rating_position'] = ratings.groupby('userId').cumcount()
        interaction_features['user_rating_position'] = ratings['user_rating_position'].values
        
        # Rating position in item's history
        ratings['item_rating_position'] = ratings.groupby('movieId').cumcount()
        interaction_features['item_rating_position'] = ratings['item_rating_position'].values
        
        self.interaction_feature_names = list(interaction_features.keys())
        
        return interaction_features
    
    def _create_temporal_user_features(self, ratings: pd.DataFrame) -> np.ndarray:
        """Create temporal features for users."""
        user_temporal = ratings.groupby('userId').agg({
            'timestamp': [
                lambda x: (x.max() - x.min()).days,  # Active period
                lambda x: len(x.dt.dayofweek.unique()),  # Unique days of week
                lambda x: len(x.dt.month.unique()),  # Unique months
                lambda x: x.dt.hour.mean() if hasattr(x.dt, 'hour') else 0,  # Avg hour
            ]
        })
        
        user_temporal.columns = ['active_days', 'unique_weekdays', 
                                'unique_months', 'avg_hour']
        self.user_feature_names.extend(user_temporal.columns.tolist())
        
        return user_temporal.values
    
    def _create_temporal_item_features(self, ratings: pd.DataFrame) -> np.ndarray:
        """Create temporal features for items."""
        item_temporal = ratings.groupby('movieId').agg({
            'timestamp': [
                lambda x: (pd.Timestamp.now() - x.max()).days,  # Days since last rating
                lambda x: (x.max() - x.min()).days,  # Lifespan
                lambda x: x.dt.dayofweek.value_counts().std(),  # Weekday distribution
            ]
        })
        
        item_temporal.columns = ['days_since_last', 'lifespan', 'weekday_std']
        self.item_feature_names.extend(item_temporal.columns.tolist())
        
        return item_temporal.fillna(0).values
    
    def _create_demographic_features(self,
                                    users: pd.DataFrame,
                                    user_ids: np.ndarray) -> np.ndarray:
        """Create demographic features."""
        # One-hot encode categorical features
        features_list = []
        
        if 'gender' in users.columns:
            gender_encoded = pd.get_dummies(users['gender'], prefix='gender')
            features_list.append(gender_encoded.values)
            self.user_feature_names.extend(gender_encoded.columns.tolist())
        
        if 'age' in users.columns:
            # Age groups
            age_groups = pd.cut(users['age'], bins=[0, 18, 25, 35, 45, 55, 100])
            age_encoded = pd.get_dummies(age_groups, prefix='age_group')
            features_list.append(age_encoded.values)
            self.user_feature_names.extend(age_encoded.columns.tolist())
        
        if 'occupation' in users.columns:
            # Top occupations only (to limit dimensions)
            top_occupations = users['occupation'].value_counts().head(10).index
            occ_filtered = users['occupation'].apply(
                lambda x: x if x in top_occupations else 'other'
            )
            occ_encoded = pd.get_dummies(occ_filtered, prefix='occupation')
            features_list.append(occ_encoded.values)
            self.user_feature_names.extend(occ_encoded.columns.tolist())
        
        if features_list:
            return np.hstack(features_list)
        else:
            return np.zeros((len(user_ids), 1))
    
    def _create_content_features(self,
                                movies: pd.DataFrame,
                                item_ids: np.ndarray) -> np.ndarray:
        """Create content-based features for items."""
        features_list = []
        
        # Genre features
        genre_cols = [col for col in movies.columns if col.startswith('genre_')]
        if genre_cols:
            genre_features = movies[genre_cols].values
            features_list.append(genre_features)
            self.item_feature_names.extend(genre_cols)
        
        # Year features
        if 'year' in movies.columns:
            year_features = movies[['year']].fillna(movies['year'].median()).values
            decade_features = pd.get_dummies(
                (movies['year'] // 10) * 10,
                prefix='decade'
            ).values
            features_list.append(year_features)
            features_list.append(decade_features)
            self.item_feature_names.append('year')
            self.item_feature_names.extend([f'decade_{i}' for i in range(decade_features.shape[1])])
        
        # Title features
        if 'title' in movies.columns:
            # Title length
            title_length = movies['title'].str.len().values.reshape(-1, 1)
            features_list.append(title_length)
            self.item_feature_names.append('title_length')
            
            # TF-IDF on titles
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english'
                )
                title_tfidf = self.tfidf_vectorizer.fit_transform(movies['title'])
            else:
                title_tfidf = self.tfidf_vectorizer.transform(movies['title'])
            
            features_list.append(title_tfidf.toarray())
            self.item_feature_names.extend([f'title_tfidf_{i}' for i in range(50)])
        
        if features_list:
            return np.hstack(features_list)
        else:
            return np.zeros((len(item_ids), 1))
    
    def _create_user_tag_features(self,
                                 tags: pd.DataFrame,
                                 user_ids: np.ndarray) -> np.ndarray:
        """Create tag-based features for users."""
        # Count unique tags per user
        user_tag_counts = tags.groupby('userId')['tag'].nunique()
        
        # Create tag preference vector using TF-IDF
        user_tags = tags.groupby('userId')['tag'].apply(lambda x: ' '.join(x))
        
        vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
        user_tag_matrix = vectorizer.fit_transform(user_tags.values)
        
        # Align with user_ids
        tag_features = np.zeros((len(user_ids), user_tag_matrix.shape[1] + 1))
        
        for i, uid in enumerate(user_ids):
            if uid in user_tag_counts.index:
                tag_features[i, 0] = user_tag_counts[uid]
            if uid in user_tags.index:
                idx = user_tags.index.get_loc(uid)
                tag_features[i, 1:] = user_tag_matrix[idx].toarray()
        
        self.user_feature_names.append('n_tags')
        self.user_feature_names.extend([f'user_tag_{i}' for i in range(30)])
        
        return tag_features
    
    def _create_item_tag_features(self,
                                 tags: pd.DataFrame,
                                 item_ids: np.ndarray) -> np.ndarray:
        """Create tag-based features for items."""
        # Count tags per item
        item_tag_counts = tags.groupby('movieId')['tag'].count()
        
        # Create tag vector using TF-IDF
        item_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
        
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        item_tag_matrix = vectorizer.fit_transform(item_tags.values)
        
        # Align with item_ids
        tag_features = np.zeros((len(item_ids), item_tag_matrix.shape[1] + 1))
        
        for i, iid in enumerate(item_ids):
            if iid in item_tag_counts.index:
                tag_features[i, 0] = item_tag_counts[iid]
            if iid in item_tags.index:
                idx = item_tags.index.get_loc(iid)
                tag_features[i, 1:] = item_tag_matrix[idx].toarray()
        
        self.item_feature_names.append('n_tags')
        self.item_feature_names.extend([f'item_tag_{i}' for i in range(50)])
        
        return tag_features
    
    def _create_popularity_features(self,
                                   ratings: pd.DataFrame,
                                   item_ids: np.ndarray) -> np.ndarray:
        """Create popularity features for items."""
        # Rating count
        rating_counts = ratings['movieId'].value_counts()
        
        # Rating count percentile
        percentiles = rating_counts.rank(pct=True)
        
        # Trending score (recent vs overall)
        if 'timestamp' in ratings.columns:
            recent_date = ratings['timestamp'].max() - pd.Timedelta(days=30)
            recent_counts = ratings[ratings['timestamp'] > recent_date]['movieId'].value_counts()
            trending_score = (recent_counts / (rating_counts + 1)).fillna(0)
        else:
            trending_score = pd.Series(0, index=item_ids)
        
        # Align with item_ids
        popularity_features = np.zeros((len(item_ids), 3))
        
        for i, iid in enumerate(item_ids):
            if iid in rating_counts.index:
                popularity_features[i, 0] = rating_counts[iid]
                popularity_features[i, 1] = percentiles[iid]
                if iid in trending_score.index:
                    popularity_features[i, 2] = trending_score[iid]
        
        self.item_feature_names.extend(['rating_count', 'popularity_percentile', 'trending_score'])
        
        return popularity_features
    
    def _create_genre_preferences(self,
                                 ratings: pd.DataFrame,
                                 user_ids: np.ndarray) -> Optional[np.ndarray]:
        """Create genre preference features for users."""
        # This would require joining with movies data
        # Placeholder for now
        return None
    
    def _calculate_item_similarity_count(self, ratings: pd.DataFrame) -> np.ndarray:
        """Calculate how many similar items a user has rated."""
        # Simplified version - count items in same rating range
        rating_ranges = pd.cut(ratings['rating'], bins=[0, 2, 3.5, 5], labels=['low', 'mid', 'high'])
        
        similarity_counts = []
        for _, row in ratings.iterrows():
            user_id = row['userId']
            rating_range = rating_ranges[row.name]
            
            # Count user's other ratings in same range
            user_ratings = ratings[ratings['userId'] == user_id]
            user_ranges = rating_ranges[user_ratings.index]
            count = (user_ranges == rating_range).sum() - 1  # Exclude current
            
            similarity_counts.append(count)
        
        return np.array(similarity_counts)
    
    def _apply_dimensionality_reduction(self, features: Dict) -> Dict:
        """Apply dimensionality reduction to features."""
        logger.info(f"Applying dimensionality reduction to {self.n_components} components")
        
        if 'user_features' in features:
            if self.user_pca is None:
                self.user_pca = TruncatedSVD(n_components=min(self.n_components, 
                                                             features['user_features'].shape[1]))
                features['user_features'] = self.user_pca.fit_transform(features['user_features'])
            else:
                features['user_features'] = self.user_pca.transform(features['user_features'])
            
            logger.info(f"User features reduced to {features['user_features'].shape[1]} dimensions")
        
        if 'item_features' in features:
            if self.item_pca is None:
                self.item_pca = TruncatedSVD(n_components=min(self.n_components,
                                                             features['item_features'].shape[1]))
                features['item_features'] = self.item_pca.fit_transform(features['item_features'])
            else:
                features['item_features'] = self.item_pca.transform(features['item_features'])
            
            logger.info(f"Item features reduced to {features['item_features'].shape[1]} dimensions")
        
        return features
    
    def _transform_user_features(self,
                                ratings: pd.DataFrame,
                                users: Optional[pd.DataFrame],
                                tags: Optional[pd.DataFrame]) -> np.ndarray:
        """Transform user features using fitted engineer."""
        # Similar to _create_user_features but using fitted transformers
        return self._create_user_features(ratings, users, tags)
    
    def _transform_item_features(self,
                                ratings: pd.DataFrame,
                                movies: pd.DataFrame,
                                tags: Optional[pd.DataFrame]) -> np.ndarray:
        """Transform item features using fitted engineer."""
        # Similar to _create_item_features but using fitted transformers
        return self._create_item_features(ratings, movies, tags)