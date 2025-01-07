# Breast Cancer Diagnosis Prediction

A machine learning project that predicts breast cancer diagnosis (malignant vs benign) using cell nuclei characteristics from fine needle aspirate (FNA) images.

## Project Introduction
This project builds a Random Forest classifier to predict breast cancer diagnoses based on cell nuclei measurements. The model analyses various features like radius, texture, perimeter, area, smoothness, etc., extracted from digitised images of FNA samples.

### Key Features
- Automated cancer diagnosis prediction with ~98% accuracy
- Comprehensive feature analysis and selection
- Model performance visualization and interpretation
- SHAP (SHapley Additive exPlanations) value analysis for model interpretability


## Project Structure

```
├── data/
│   └── cell-data.csv         # Processed dataset
├── models/
│   └── cancer_diagnosis_model.joblib  # Trained model
├── notebooks/
│   └── randomforest.ipynb    # Main analysis notebook
└── results/
    ├── boxplot_analysis.png
    ├── confusion_matrix.png
    ├── correlation_matrix.png
    ├── feature_importance.png
    ├── learning_curves.png
    ├── roc_balanced_vs_unbalanced.png
    └── shap_values_distribution.png
```

## Dataset

The dataset contains measurements from digitised images of FNA samples, including:
- Cell nuclei characteristics (30 features)
- Binary classification (Malignant/Benign)
- 569 instances

Features include measurements of:

- **Radius**: Mean of distances from center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter** and **Area**.
- **Smoothness**: Local variation in radius lengths.
- **Compactness**: Calculated as: (perimeter² / area - 1.0).
- **Concavity**: Severity of concave portions of the contour.
- **Concave Points**: Number of concave portions of the contour.
- **Symmetry** and **Fractal Dimension**: "Coastline approximation" - 1.

By analysing these features, the goal is to develop a machine learning pipeline to predict tumor diagnosis accurately.

## Model Performance

The Random Forest classifier achieves:
- ROC-AUC Score: ~0.99
- Accuracy: ~97%
- Precision: ~98%
- Recall: ~96%

## Key Findings

1. Most important features for diagnosis:
   - Concave points (worst)
   - Area (worst)
   - Perimeter (worst)

2. Feature correlations:
   - Strong correlations between radius, perimeter, and area measurements
   - SE (standard error) measurements showed lower predictive power

3. Model behavior:
   - Excellent separation between malignant and benign cases
   - Robust performance across different cross-validation splits
   - High confidence in predictions with clear decision boundaries

## Setup and Usage

1. Clone the repository:
```bash
git clone git clone https://github.com/ghchen99/breast-cancer-wisconsin.git
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

4. Run the notebook:
```bash
jupyter notebook notebooks/randomforest.ipynb
```

## Future Improvements

1. Feature engineering:
    - Explore polynomial features
    - Investigate feature ratios

2. Model enhancements:
    - Experiment with other algorithms (XGBoost, LightGBM)
    - Implement ensemble methods

3. Deployment:
    - Create API endpoint
    - Develop web interface

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License