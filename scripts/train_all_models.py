#!/usr/bin/env python
"""Script to train all recommendation models."""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import mlflow

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.collaborative_filtering.user_based import UserBasedCF
from src.models.collaborative_filtering.item_based import ItemBasedCF
from src.models.matrix_factorization.svd_model import SVDModel
from src.models.matrix_factorization.nmf_model import NMFModel
from src.models.matrix_factorization.als_model import ALSModel
from src.models.deep_learning.ncf_model import NCFModel
from src.models.hybrid.lightfm_model import LightFMModel
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import RecommenderMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model registry
MODEL_CLASSES = {
    'user_based': UserBasedCF,
    'item_based': ItemBasedCF,
    'svd': SVDModel,
    'nmf': NMFModel,
    'als': ALSModel,
    'ncf': NCFModel,
    'lightfm': LightFMModel
}

MODEL_SECTIONS = {
    'user_based': 'collaborative_filtering',
    'item_based': 'collaborative_filtering',
    'svd': 'matrix_factorization',
    'nmf': 'matrix_factorization',
    'als': 'matrix_factorization',
    'ncf': 'deep_learning',
    'lightfm': 'hybrid'
}

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_dir: str = 'data/processed') -> tuple:
    """Load preprocessed data splits."""
    split_dir = Path(data_dir) / 'splits'
    
    if not split_dir.exists():
        logger.warning(f"Split directory {split_dir} not found. Loading full dataset...")
        
        # Load and split data
        data_loader = DataLoader()
        dataset = data_loader.load_dataset('ml-latest-small')
        
        # Preprocess
        preprocessor = DataPreprocessor()
        dataset = preprocessor.fit_transform(dataset)
        
        # Split
        splitter = DataSplitter(strategy='random')
        data_split = splitter.split(dataset.ratings)
        
        return data_split.train, data_split.val, data_split.test
    
    # Load splits
    train_data = pd.read_parquet(split_dir / 'train.parquet')
    val_data = pd.read_parquet(split_dir / 'val.parquet')
    test_data = pd.read_parquet(split_dir / 'test.parquet')
    
    return train_data, val_data, test_data


def train_model(model_name: str, 
                model_config: Dict[str, Any],
                train_data: pd.DataFrame,
                val_data: pd.DataFrame,
                test_data: pd.DataFrame,
                save_dir: str = 'results/models') -> Dict[str, float]:
    """Train a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}")
    
    # Get model class
    if model_name not in MODEL_CLASSES:
        logger.error(f"Unknown model: {model_name}")
        return {}
    
    model_class = MODEL_CLASSES[model_name]
    
    # Initialize model with config
    model = model_class()     
    model.params = model_config.get('parameters', {})

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        metrics=RecommenderMetrics(),
        mlflow_tracking=True,
        experiment_name='movielens-benchmark',
        save_dir=save_dir
    )
    
    # Train model
    training_config = model_config.get('training', {})
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=training_config.get('epochs', 1),
        early_stopping=training_config.get('early_stopping', {}).get('enabled', True),
        patience=training_config.get('early_stopping', {}).get('patience', 5),
        monitor_metric=training_config.get('early_stopping', {}).get('monitor', 'val_rmse'),
        verbose=True
    )
    
    return history['test_metrics']


def main():
    """Train all recommendation models."""
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=list(MODEL_CLASSES.keys()) + ['all'],
        help='Model to train (or "all" for all models)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='movielens-benchmark',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Setup MLflow
        if not args.no_mlflow:
            mlflow.set_experiment(args.experiment)
        
        # Load data
        logger.info("Loading data...")
        train_data, val_data, test_data = load_data(args.data_dir)
        logger.info(f"Data loaded: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
        
        # Determine which models to train
        if args.model == 'all' or args.model is None:
            models_to_train = list(MODEL_CLASSES.keys())
        else:
            models_to_train = [args.model]
        
        # Train models
        results = {}
        for model_name in models_to_train:
            # Check if model is enabled in config
            #model_config = config.get(model_name, config.get('collaborative_filtering', {}).get(model_name))
            section = MODEL_SECTIONS.get(model_name)
            model_config = config.get(section, {}).get(model_name)   
            params = model_config.get('parameters', {})

            if not model_config:
                logger.warning(f"No configuration found for {model_name}, skipping...")
                continue
            
            if not model_config.get('enabled', True):
                logger.info(f"Model {model_name} is disabled in config, skipping...")
                continue
            
            # Train model
            test_metrics = train_model(
                model_name=model_name,
                model_config=model_config,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                save_dir=args.save_dir
            )
            
            results[model_name] = test_metrics
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info(" Training Summary ".center(60))
        logger.info("="*60)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name}:")
            for metric_name, value in sorted(metrics.items()):
                if 'test_' in metric_name:
                    clean_name = metric_name.replace('test_', '')
                    logger.info(f"  {clean_name:20s}: {value:.4f}")
        
        logger.info("\nAll models trained successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())