"""Main Gradio application for MovieLens Recommendation System."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import DataLoader
from src.models.base_model import BaseRecommender
from src.evaluation.metrics import RecommenderMetrics

logger = logging.getLogger(__name__)

# Global variables for models and data
MODELS = {}
DATA = None
MOVIES_DF = None
METRICS_CALCULATOR = RecommenderMetrics()


def load_models(models_dir: str = "results/models") -> Dict[str, BaseRecommender]:
    """Load trained models from directory."""
    models = {}
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.warning(f"Models directory {models_dir} does not exist")
        return models
    
    for model_file in models_path.glob("*_final.pkl"):
        try:
            model_name = model_file.stem.replace("_final", "")
            model = BaseRecommender.load(model_file)
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")
    
    return models


def load_data(dataset_name: str = "ml-latest-small") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens data."""
    data_loader = DataLoader()
    dataset = data_loader.load_dataset(dataset_name)
    return dataset.ratings, dataset.movies


def get_user_history(user_id: int, n_items: int = 10) -> pd.DataFrame:
    """Get user's rating history."""
    if DATA is None:
        return pd.DataFrame()
    
    user_data = DATA[DATA['userId'] == user_id].copy()
    
    if MOVIES_DF is not None:
        user_data = user_data.merge(MOVIES_DF[['movieId', 'title', 'genres']], on='movieId', how='left')
    
    user_data = user_data.sort_values('timestamp', ascending=False).head(n_items)
    
    return user_data[['movieId', 'title', 'genres', 'rating', 'timestamp']]


def get_recommendations(user_id: int, model_name: str, n_recommendations: int = 10) -> pd.DataFrame:
    """Get recommendations for a user."""
    if model_name not in MODELS:
        return pd.DataFrame({"Error": ["Model not loaded"]})
    
    model = MODELS[model_name]
    
    try:
        # Get recommendations
        recommendations, scores = model.recommend(
            [user_id],
            n_recommendations=n_recommendations,
            filter_seen=True,
            return_scores=True
        )
        
        # Convert to DataFrame
        rec_df = pd.DataFrame({
            'movieId': recommendations[0],
            'predicted_score': scores[0]
        })
        
        # Add movie information
        if MOVIES_DF is not None:
            rec_df = rec_df.merge(MOVIES_DF[['movieId', 'title', 'genres', 'year']], 
                                 on='movieId', how='left')
        
        rec_df['rank'] = range(1, len(rec_df) + 1)
        
        return rec_df[['rank', 'movieId', 'title', 'genres', 'year', 'predicted_score']]
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return pd.DataFrame({"Error": [str(e)]})


def compare_models(user_id: int, n_recommendations: int = 10) -> pd.DataFrame:
    """Compare recommendations from different models."""
    results = []
    
    for model_name in MODELS.keys():
        recs = get_recommendations(user_id, model_name, n_recommendations)
        if not recs.empty and 'Error' not in recs.columns:
            recs['model'] = model_name
            results.append(recs)
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame({"Error": ["No models available"]})


def plot_user_stats(user_id: int) -> go.Figure:
    """Create visualization of user statistics."""
    if DATA is None:
        return go.Figure()
    
    user_data = DATA[DATA['userId'] == user_id]
    
    if user_data.empty:
        return go.Figure().add_annotation(text="No data for this user")
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Rating Distribution', 'Ratings Over Time', 
                       'Genre Preferences', 'Rating Activity'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'histogram'}]]
    )
    
    # Rating distribution
    rating_counts = user_data['rating'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=rating_counts.index, y=rating_counts.values, name='Ratings'),
        row=1, col=1
    )
    
    # Ratings over time
    if 'timestamp' in user_data.columns:
        fig.add_trace(
            go.Scatter(x=user_data['timestamp'], y=user_data['rating'], 
                      mode='markers', name='Rating Timeline'),
            row=1, col=2
        )
    
    # Genre preferences (if available)
    if MOVIES_DF is not None and 'genres' in MOVIES_DF.columns:
        user_movies = user_data.merge(MOVIES_DF[['movieId', 'genres']], on='movieId')
        
        # Split genres and count
        genre_list = []
        for genres in user_movies['genres'].dropna():
            genre_list.extend(genres.split('|'))
        
        if genre_list:
            genre_counts = pd.Series(genre_list).value_counts().head(10)
            fig.add_trace(
                go.Bar(x=genre_counts.values, y=genre_counts.index, 
                      orientation='h', name='Genres'),
                row=2, col=1
            )
    
    # Rating activity histogram
    fig.add_trace(
        go.Histogram(x=user_data['rating'], nbinsx=10, name='Rating Frequency'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, 
                     title=f"User {user_id} Statistics")
    
    return fig


def plot_model_comparison() -> go.Figure:
    """Create model comparison visualization."""
    if not MODELS:
        return go.Figure().add_annotation(text="No models loaded")
    
    # Load metrics if available
    metrics_data = []
    
    for model_name in MODELS.keys():
        metrics_file = Path(f"results/models/{model_name}_final_history.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                history = json.load(f)
                if 'test_metrics' in history:
                    metrics = history['test_metrics']
                    metrics['model'] = model_name
                    metrics_data.append(metrics)
    
    if not metrics_data:
        return go.Figure().add_annotation(text="No metrics data available")
    
    df = pd.DataFrame(metrics_data)
    
    # Create bar chart for key metrics
    metrics_to_plot = ['rmse', 'mae', 'precision_at_10', 'recall_at_10', 'ndcg_at_10']
    available_metrics = [m for m in metrics_to_plot if f"test_{m}" in df.columns]
    
    if not available_metrics:
        return go.Figure().add_annotation(text="No metrics to plot")
    
    fig = go.Figure()
    
    for metric in available_metrics:
        col_name = f"test_{metric}"
        if col_name in df.columns:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=df['model'],
                y=df[col_name],
                text=df[col_name].round(4),
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode='group',
        height=500
    )
    
    return fig


def create_app():
    """Create Gradio application."""
    global MODELS, DATA, MOVIES_DF
    
    # Load models and data
    logger.info("Loading models and data...")
    MODELS = load_models()
    DATA, MOVIES_DF = load_data()
    logger.info(f"Loaded {len(MODELS)} models")
    
    # Create Gradio interface
    with gr.Blocks(title="MovieLens Recommendation System", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎬 MovieLens Recommendation System Benchmark
            
            Interactive dashboard for testing and comparing recommendation algorithms on the MovieLens dataset.
            """
        )
        
        with gr.Tab("Get Recommendations"):
            with gr.Row():
                with gr.Column(scale=1):
                    user_id_input = gr.Number(
                        label="User ID",
                        value=1,
                        precision=0,
                        info="Enter a user ID to get recommendations"
                    )
                    model_select = gr.Dropdown(
                        label="Select Model",
                        choices=list(MODELS.keys()) if MODELS else [],
                        value=list(MODELS.keys())[0] if MODELS else None
                    )
                    n_recs_slider = gr.Slider(
                        label="Number of Recommendations",
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5
                    )
                    recommend_btn = gr.Button("Get Recommendations", variant="primary")
                
                with gr.Column(scale=2):
                    recommendations_output = gr.Dataframe(
                        label="Recommendations",
                        headers=["Rank", "Movie ID", "Title", "Genres", "Year", "Score"],
                        datatype=["number", "number", "str", "str", "number", "number"]
                    )
            
            with gr.Row():
                user_history_output = gr.Dataframe(
                    label="User's Recent Ratings",
                    headers=["Movie ID", "Title", "Genres", "Rating", "Timestamp"]
                )
            
            recommend_btn.click(
                fn=lambda u, m, n: (
                    get_recommendations(int(u), m, n),
                    get_user_history(int(u), n)
                ),
                inputs=[user_id_input, model_select, n_recs_slider],
                outputs=[recommendations_output, user_history_output]
            )
        
        with gr.Tab("Compare Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    compare_user_input = gr.Number(
                        label="User ID",
                        value=1,
                        precision=0
                    )
                    compare_n_recs = gr.Slider(
                        label="Number of Recommendations",
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=5
                    )
                    compare_btn = gr.Button("Compare All Models", variant="primary")
                
                with gr.Column(scale=3):
                    comparison_output = gr.Dataframe(
                        label="Model Comparison",
                        headers=["Rank", "Movie ID", "Title", "Genres", "Year", "Score", "Model"]
                    )
            
            compare_btn.click(
                fn=lambda u, n: compare_models(int(u), n),
                inputs=[compare_user_input, compare_n_recs],
                outputs=comparison_output
            )
        
        with gr.Tab("User Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    analysis_user_input = gr.Number(
                        label="User ID",
                        value=1,
                        precision=0
                    )
                    analyze_btn = gr.Button("Analyze User", variant="primary")
                
                with gr.Column(scale=3):
                    user_stats_plot = gr.Plot(label="User Statistics")
            
            analyze_btn.click(
                fn=lambda u: plot_user_stats(int(u)),
                inputs=analysis_user_input,
                outputs=user_stats_plot
            )
        
        with gr.Tab("Model Performance"):
            gr.Markdown("## Model Performance Metrics")
            
            performance_plot = gr.Plot(label="Performance Comparison")
            refresh_btn = gr.Button("Refresh Metrics")
            
            refresh_btn.click(
                fn=plot_model_comparison,
                inputs=[],
                outputs=performance_plot
            )
            
            # Load initial plot
            performance_plot.value = plot_model_comparison()
        
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About This Application
                
                This is a comprehensive benchmark system for recommendation algorithms on the MovieLens dataset.
                
                ### Features:
                - **Multiple Algorithms**: Collaborative Filtering, Matrix Factorization, Deep Learning, and Hybrid models
                - **Interactive Testing**: Test recommendations for any user
                - **Model Comparison**: Compare recommendations from different models
                - **User Analysis**: Visualize user preferences and rating patterns
                - **Performance Metrics**: View comprehensive evaluation metrics
                
                ### Available Models:
                - User-Based Collaborative Filtering
                - Item-Based Collaborative Filtering
                - SVD (Singular Value Decomposition)
                - NMF (Non-negative Matrix Factorization)
                - NCF (Neural Collaborative Filtering)
                - LightFM (Hybrid Factorization)
                
                ### Metrics:
                - **Rating Prediction**: RMSE, MAE, R²
                - **Ranking**: Precision@K, Recall@K, NDCG@K, MAP@K
                - **Beyond Accuracy**: Coverage, Diversity, Novelty
                
                ### Dataset:
                Using MovieLens dataset with ratings from users on movies.
                """
            )
    
    return app


def launch_app(share: bool = False, port: int = 7860):
    """Launch the Gradio application."""
    app = create_app()
    app.launch(share=share, server_port=port, server_name="0.0.0.0")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch MovieLens Recommendation System Dashboard")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    launch_app(share=args.share, port=args.port)


if __name__ == "__main__":
    main()