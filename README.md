# Local Installation

```bash
# Clone the repository
git clone https://github.com/HubertRozumek/movie-recommender.git
cd movielens-recommendation-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Download MovieLens dataset
python scripts/download_data.py
```

# Docker

```bash
# Build image
docker build -t movielens-benchmark -f docker/Dockerfile .

# Run container
docker run -p 7860:7860 -p 5000:5000 movielens-benchmark

# Or using Docker Compose
docker-compose -f docker/docker-compose.yml up
```

# Train a Single Model

```bash
python scripts/train_all_models.py --model svd --config config/model_config.yaml
```

# Run Full Benchmark

```bash
python scripts/run_benchmark.py --all-models --metrics all
```

# Launch Gradio Application

```bash
python app/main.py
# Application available at: http://localhost:7860
```

# MLflow UI

```bash
mlflow ui --backend-store-uri experiments/mlruns
# UI available at: http://localhost:5000
```