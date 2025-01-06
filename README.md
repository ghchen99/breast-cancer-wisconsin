# Breast Cancer Wisconsin

## Project Introduction
This project focuses on predicting whether a breast cancer tumor is malignant or benign based on features computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. The dataset describes characteristics of cell nuclei present in the image, such as:

- **Radius**: Mean of distances from center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter** and **Area**.
- **Smoothness**: Local variation in radius lengths.
- **Compactness**: Calculated as: (perimeter² / area - 1.0).
- **Concavity**: Severity of concave portions of the contour.
- **Concave Points**: Number of concave portions of the contour.
- **Symmetry** and **Fractal Dimension**: "Coastline approximation" - 1.

By analysing these features, the goal is to develop a machine learning pipeline to predict tumor diagnosis accurately.

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