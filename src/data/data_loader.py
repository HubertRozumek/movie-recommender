"""Data loader for MovieLens datasets."""

import os
import logging
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class MovieLensDataset:
    """Container for MovieLens dataset components."""
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: Optional[pd.DataFrame] = None
    tags: Optional[pd.DataFrame] = None
    links: Optional[pd.DataFrame] = None
    
    @property
    def n_users(self) -> int:
        """Get number of unique users."""
        return self.ratings['userId'].nunique()
    
    @property
    def n_items(self) -> int:
        """Get number of unique items."""
        return self.ratings['movieId'].nunique()
    
    @property
    def n_ratings(self) -> int:
        """Get total number of ratings."""
        return len(self.ratings)
    
    @property
    def sparsity(self) -> float:
        """Calculate sparsity of the rating matrix."""
        return 1 - (self.n_ratings / (self.n_users * self.n_items))
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'n_ratings': self.n_ratings,
            'sparsity': self.sparsity,
            'rating_min': self.ratings['rating'].min(),
            'rating_max': self.ratings['rating'].max(),
            'rating_mean': self.ratings['rating'].mean(),
            'rating_std': self.ratings['rating'].std(),
            'ratings_per_user': self.n_ratings / self.n_users,
            'ratings_per_item': self.n_ratings / self.n_items
        }


class DataLoader:
    """Load and manage MovieLens datasets."""
    
    DATASETS = {
        'ml-100k': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'ratings_file': 'ml-100k/u.data',
            'movies_file': 'ml-100k/u.item',
            'delimiter': '\t',
            'encoding': 'latin-1'
        },
        'ml-1m': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            'ratings_file': 'ml-1m/ratings.dat',
            'movies_file': 'ml-1m/movies.dat',
            'users_file': 'ml-1m/users.dat',
            'delimiter': '::',
            'encoding': 'latin-1'
        },
        'ml-10m': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
            'ratings_file': 'ml-10M100K/ratings.dat',
            'movies_file': 'ml-10M100K/movies.dat',
            'tags_file': 'ml-10M100K/tags.dat',
            'delimiter': '::',
            'encoding': 'utf-8'
        },
        'ml-20m': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip',
            'ratings_file': 'ml-20m/ratings.csv',
            'movies_file': 'ml-20m/movies.csv',
            'tags_file': 'ml-20m/tags.csv',
            'links_file': 'ml-20m/links.csv',
            'delimiter': ',',
            'encoding': 'utf-8'
        },
        'ml-latest-small': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'ratings_file': 'ml-latest-small/ratings.csv',
            'movies_file': 'ml-latest-small/movies.csv',
            'tags_file': 'ml-latest-small/tags.csv',
            'links_file': 'ml-latest-small/links.csv',
            'delimiter': ',',
            'encoding': 'utf-8'
        },
        'ml-latest': {
            'url': 'http://files.grouplens.org/datasets/movielens/ml-latest.zip',
            'ratings_file': 'ml-latest/ratings.csv',
            'movies_file': 'ml-latest/movies.csv',
            'tags_file': 'ml-latest/tags.csv',
            'links_file': 'ml-latest/links.csv',
            'delimiter': ',',
            'encoding': 'utf-8'
        }
    }
    
    def __init__(self, data_dir: Union[str, Path] = 'data/raw'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory to store/load data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download MovieLens dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'ml-100k', 'ml-1m')
            force: Force re-download even if exists
            
        Returns:
            Path to the extracted dataset
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(self.DATASETS.keys())}")
        
        dataset_info = self.DATASETS[dataset_name]
        zip_path = self.data_dir / f"{dataset_name}.zip"
        extract_path = self.data_dir / dataset_name
        
        # Check if already extracted
        if extract_path.exists() and not force:
            logger.info(f"Dataset {dataset_name} already exists at {extract_path}")
            return extract_path
        
        # Download if needed
        if not zip_path.exists() or force:
            logger.info(f"Downloading {dataset_name} from {dataset_info['url']}")
            self._download_file(dataset_info['url'], zip_path)
        
        # Extract
        logger.info(f"Extracting {dataset_name} to {self.data_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        return extract_path
    
    def load_dataset(self, dataset_name: str = 'ml-latest-small', 
                    download: bool = True) -> MovieLensDataset:
        """
        Load MovieLens dataset.
        
        Args:
            dataset_name: Name of the dataset
            download: Whether to download if not exists
            
        Returns:
            MovieLensDataset object
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.DATASETS[dataset_name]
        
        # Download if needed
        if download:
            self.download_dataset(dataset_name)
        
        # Load ratings
        ratings_path = self.data_dir / dataset_info['ratings_file']
        logger.info(f"Loading ratings from {ratings_path}")
        ratings = self._load_ratings(ratings_path, dataset_name)
        
        # Load movies
        movies_path = self.data_dir / dataset_info['movies_file']
        logger.info(f"Loading movies from {movies_path}")
        movies = self._load_movies(movies_path, dataset_name)
        
        # Load optional components
        users = None
        if 'users_file' in dataset_info:
            users_path = self.data_dir / dataset_info['users_file']
            if users_path.exists():
                logger.info(f"Loading users from {users_path}")
                users = self._load_users(users_path, dataset_name)
        
        tags = None
        if 'tags_file' in dataset_info:
            tags_path = self.data_dir / dataset_info['tags_file']
            if tags_path.exists():
                logger.info(f"Loading tags from {tags_path}")
                tags = self._load_tags(tags_path, dataset_name)
        
        links = None
        if 'links_file' in dataset_info:
            links_path = self.data_dir / dataset_info['links_file']
            if links_path.exists():
                logger.info(f"Loading links from {links_path}")
                links = self._load_links(links_path, dataset_name)
        
        dataset = MovieLensDataset(
            ratings=ratings,
            movies=movies,
            users=users,
            tags=tags,
            links=links
        )
        
        # Log statistics
        stats = dataset.get_statistics()
        logger.info(f"Dataset loaded: {stats}")
        
        return dataset
    
    def _download_file(self, url: str, dest_path: Path):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _load_ratings(self, path: Path, dataset_name: str) -> pd.DataFrame:
        """Load ratings file."""
        info = self.DATASETS[dataset_name]
        
        if dataset_name == 'ml-100k':
            # Special handling for ml-100k
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=['userId', 'movieId', 'rating', 'timestamp'],
                encoding=info['encoding']
            )
        elif dataset_name in ['ml-1m', 'ml-10m']:
            # Special handling for dat files
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=['userId', 'movieId', 'rating', 'timestamp'],
                encoding=info['encoding'],
                engine='python'
            )
        else:
            # CSV files
            df = pd.read_csv(path, encoding=info['encoding'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Ensure correct data types
        df['userId'] = df['userId'].astype(np.int32)
        df['movieId'] = df['movieId'].astype(np.int32)
        df['rating'] = df['rating'].astype(np.float32)
        
        return df
    
    def _load_movies(self, path: Path, dataset_name: str) -> pd.DataFrame:
        """Load movies file."""
        info = self.DATASETS[dataset_name]
        
        if dataset_name == 'ml-100k':
            # Special handling for ml-100k
            columns = ['movieId', 'title', 'release_date', 'video_release_date',
                      'imdb_url'] + [f'genre_{i}' for i in range(19)]
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=columns,
                encoding=info['encoding']
            )
            # Combine genre columns
            genre_cols = [col for col in df.columns if col.startswith('genre_')]
            df['genres'] = df[genre_cols].apply(
                lambda x: '|'.join([str(i) for i, v in enumerate(x) if v == 1]),
                axis=1
            )
            df = df[['movieId', 'title', 'genres']]
        elif dataset_name in ['ml-1m', 'ml-10m']:
            # Special handling for dat files
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=['movieId', 'title', 'genres'],
                encoding=info['encoding'],
                engine='python'
            )
        else:
            # CSV files
            df = pd.read_csv(path, encoding=info['encoding'])
        
        # Ensure correct data types
        df['movieId'] = df['movieId'].astype(np.int32)
        
        # Extract year from title if present
        df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype('float')
        
        return df
    
    def _load_users(self, path: Path, dataset_name: str) -> pd.DataFrame:
        """Load users file."""
        info = self.DATASETS[dataset_name]
        
        if dataset_name == 'ml-1m':
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
                encoding=info['encoding'],
                engine='python'
            )
        else:
            df = pd.read_csv(path, encoding=info['encoding'])
        
        df['userId'] = df['userId'].astype(np.int32)
        return df
    
    def _load_tags(self, path: Path, dataset_name: str) -> pd.DataFrame:
        """Load tags file."""
        info = self.DATASETS[dataset_name]
        
        if dataset_name == 'ml-10m':
            df = pd.read_csv(
                path,
                sep=info['delimiter'],
                names=['userId', 'movieId', 'tag', 'timestamp'],
                encoding=info['encoding'],
                engine='python'
            )
        else:
            df = pd.read_csv(path, encoding=info['encoding'])
        
        df['userId'] = df['userId'].astype(np.int32)
        df['movieId'] = df['movieId'].astype(np.int32)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def _load_links(self, path: Path, dataset_name: str) -> pd.DataFrame:
        """Load links file."""
        info = self.DATASETS[dataset_name]
        df = pd.read_csv(path, encoding=info['encoding'])
        df['movieId'] = df['movieId'].astype(np.int32)
        return df
    
    def save_processed(self, dataset: MovieLensDataset, 
                      save_dir: Union[str, Path] = 'data/processed'):
        """
        Save processed dataset.
        
        Args:
            dataset: MovieLensDataset object
            save_dir: Directory to save processed data
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficient storage
        dataset.ratings.to_parquet(save_path / 'ratings.parquet')
        dataset.movies.to_parquet(save_path / 'movies.parquet')
        
        if dataset.users is not None:
            dataset.users.to_parquet(save_path / 'users.parquet')
        if dataset.tags is not None:
            dataset.tags.to_parquet(save_path / 'tags.parquet')
        if dataset.links is not None:
            dataset.links.to_parquet(save_path / 'links.parquet')
        
        logger.info(f"Dataset saved to {save_path}")
    
    def load_processed(self, load_dir: Union[str, Path] = 'data/processed') -> MovieLensDataset:
        """
        Load processed dataset.
        
        Args:
            load_dir: Directory with processed data
            
        Returns:
            MovieLensDataset object
        """
        load_path = Path(load_dir)
        
        ratings = pd.read_parquet(load_path / 'ratings.parquet')
        movies = pd.read_parquet(load_path / 'movies.parquet')
        
        users = None
        users_path = load_path / 'users.parquet'
        if users_path.exists():
            users = pd.read_parquet(users_path)
        
        tags = None
        tags_path = load_path / 'tags.parquet'
        if tags_path.exists():
            tags = pd.read_parquet(tags_path)
        
        links = None
        links_path = load_path / 'links.parquet'
        if links_path.exists():
            links = pd.read_parquet(links_path)
        
        return MovieLensDataset(
            ratings=ratings,
            movies=movies,
            users=users,
            tags=tags,
            links=links
        )