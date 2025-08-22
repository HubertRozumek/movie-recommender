#!/usr/bin/env python
"""Script to download and prepare MovieLens dataset."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download and prepare MovieLens dataset."""
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        default='ml-latest-small',
        choices=['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-latest-small', 'ml-latest'],
        help='MovieLens dataset version to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory to save raw data'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess the data after downloading'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split the data into train/val/test'
    )
    parser.add_argument(
        '--split-strategy',
        type=str,
        default='random',
        choices=['random', 'temporal', 'user_based', 'stratified'],
        help='Strategy for splitting data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if data exists'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize data loader
        logger.info(f"Initializing data loader for {args.dataset}")
        data_loader = DataLoader(data_dir=args.data_dir)
        
        # Download dataset
        logger.info(f"Downloading {args.dataset} dataset...")
        data_loader.download_dataset(args.dataset, force=args.force)
        
        # Load dataset
        logger.info(f"Loading {args.dataset} dataset...")
        dataset = data_loader.load_dataset(args.dataset, download=False)
        
        # Print dataset statistics
        stats = dataset.get_statistics()
        logger.info("Dataset Statistics:")
        logger.info(f"  Users: {stats['n_users']:,}")
        logger.info(f"  Items: {stats['n_items']:,}")
        logger.info(f"  Ratings: {stats['n_ratings']:,}")
        logger.info(f"  Sparsity: {stats['sparsity']:.2%}")
        logger.info(f"  Rating range: {stats['rating_min']:.1f} - {stats['rating_max']:.1f}")
        logger.info(f"  Rating mean: {stats['rating_mean']:.2f} ± {stats['rating_std']:.2f}")
        
        # Preprocess if requested
        if args.preprocess:
            logger.info("Preprocessing data...")
            preprocessor = DataPreprocessor(
                min_user_ratings=20,
                min_item_ratings=5,
                normalize_ratings=True,
                remove_outliers=True
            )
            
            processed_dataset = preprocessor.fit_transform(dataset)
            
            # Save processed data
            logger.info(f"Saving processed data to {args.processed_dir}")
            data_loader.save_processed(processed_dataset, save_dir=args.processed_dir)
            
            # Print processed statistics
            processed_stats = processed_dataset.get_statistics()
            logger.info("Processed Dataset Statistics:")
            logger.info(f"  Users: {processed_stats['n_users']:,}")
            logger.info(f"  Items: {processed_stats['n_items']:,}")
            logger.info(f"  Ratings: {processed_stats['n_ratings']:,}")
            logger.info(f"  Sparsity: {processed_stats['sparsity']:.2%}")
            
            dataset = processed_dataset
        
        # Split if requested
        if args.split:
            logger.info(f"Splitting data using {args.split_strategy} strategy...")
            splitter = DataSplitter(
                strategy=args.split_strategy,
                train_size=0.7,
                val_size=0.15,
                test_size=0.15,
                random_state=42
            )
            
            data_split = splitter.split(dataset.ratings)
            
            # Save splits
            split_dir = Path(args.processed_dir) / 'splits'
            split_dir.mkdir(parents=True, exist_ok=True)
            
            data_split.train.to_parquet(split_dir / 'train.parquet')
            data_split.val.to_parquet(split_dir / 'val.parquet')
            data_split.test.to_parquet(split_dir / 'test.parquet')
            
            logger.info("Data Split Summary:")
            logger.info(f"  Train: {len(data_split.train):,} ({data_split.proportions['train']:.1%})")
            logger.info(f"  Val: {len(data_split.val):,} ({data_split.proportions['val']:.1%})")
            logger.info(f"  Test: {len(data_split.test):,} ({data_split.proportions['test']:.1%})")
        
        logger.info("Data download and preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())