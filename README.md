# Breast Cancer Wisconsin

A modular and maintainable machine learning pipeline for breast cancer classification.

## Project Structure

```
project/
│
├── src/                      # Source code
│   ├── preprocessing/        # Data preprocessing and feature engineering
│   ├── models/              # Model training and evaluation
│   ├── utils/               # Utility functions
│   └── pipeline/            # Pipeline orchestration
│
├── configs/                  # Configuration files
├── tests/                   # Unit tests
├── scripts/                 # Training and inference scripts
├── data/                    # Data directory
└── models/                  # Saved models and states
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-pipeline.git
cd ml-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training:
```bash
python scripts/train.py
```

2. Inference:
```bash
python scripts/predict.py
```

## Configuration

Modify `configs/config.yaml` to adjust:
- Preprocessing parameters
- Model hyperparameters
- Training settings

Example configuration:
```yaml
preprocessing:
  random_state: 42
  handle_outliers: true
  power_transform: true

model:
  random_state: 42
  n_cv_folds: 5
  n_iter: 20
```

## Development

1. Running tests:
```bash
pytest tests/
```

2. Adding new features:
- Add preprocessing steps in `src/preprocessing/feature_engineering.py`
- Add new models in `src/models/model_trainer.py`
- Update configuration in `configs/config.yaml`

## Project Components

### Feature Engineering
- Handles missing values
- Removes outliers
- Creates engineered features
- Applies necessary transformations

### Model Training
- Supports multiple models
- Performs hyperparameter tuning
- Evaluates model performance
- Selects best model

### Pipeline
- Orchestrates end-to-end process
- Manages state between training and inference
- Handles configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License