# Breast Cancer Diagnosis Prediction

A comprehensive machine learning project that predicts breast cancer diagnosis (malignant vs benign) using cell nuclei characteristics from fine needle aspirate (FNA) images, with an interactive web interface for real-time predictions.

## Project Introduction

This project consists of two main components:
1. A Random Forest classifier trained on cell nuclei measurements
2. A web application interface for making real-time predictions

The model analyzes various features like radius, texture, perimeter, area, smoothness, etc., extracted from digitized images of FNA samples.

## Project Structure

```
├── data/
│   └── cell-data.csv                  # Processed dataset
├── models/
│   └── cancer_diagnosis_model.joblib   # Trained model
├── notebooks/
│   └── randomforest.ipynb             # Model training notebook
├── results/                           # Analysis visualizations
│   ├── boxplot_analysis.png
│   ├── confusion_matrix.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── learning_curves.png
│   ├── roc_balanced_vs_unbalanced.png
│   └── shap_values_distribution.png
└── app/                              # Web application
    ├── backend/
    │   ├── app.py                    # Flask server
    │   ├── models/                   # Model directory
    │   └── requirements.txt          # Python dependencies
    └── frontend/
        ├── src/
        │   ├── components/
        │   │   ├── DiagnosisForm.jsx
        │   │   └── PredictionResult.jsx
        │   ├── App.jsx
        │   ├── main.jsx
        │   └── index.css
        ├── package.json
        └── index.html
```

## Features

### Machine Learning Model
- ~98% prediction accuracy
- Comprehensive feature analysis
- Model performance visualization
- SHAP value analysis for interpretability

### Web Application
- User-friendly interface for entering measurements
- Real-time input validation
- Detailed prediction results including:
  - Binary classification (Benign/Malignant)
  - Confidence levels
  - Probability distribution
  - Visual result representation
- Responsive design for desktop and mobile

## Dataset

The dataset contains measurements from digitized FNA images:
- 30 cell nuclei characteristics features
- Binary classification (Malignant/Benign)
- 569 instances

Features measured include:
- **Radius**: Mean distances from center to perimeter points
- **Texture**: Gray-scale values standard deviation
- **Perimeter** and **Area**
- **Smoothness**: Local radius length variations
- **Compactness**: (perimeter² / area - 1.0)
- **Concavity**: Contour concave portions severity
- **Concave Points**: Number of concave contour portions
- **Symmetry** and **Fractal Dimension**

## Setup and Usage

### 1. Model Training Environment

```bash
# Clone the repository
git clone https://github.com/ghchen99/breast-cancer-wisconsin.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ML requirements
pip install -r requirements.txt

# Run the training notebook
jupyter notebook notebooks/randomforest.ipynb
```

### 2. Web Application

#### Backend Setup
```bash
cd app/backend
pip install -r requirements.txt
python app.py  # Starts Flask server on port 5000
```

#### Frontend Setup
```bash
cd app/frontend
npm install
npm run dev   # Starts development server on port 5173
```

The application will be available at http://localhost:5173

## Model Performance

The Random Forest classifier achieves:
- ROC-AUC Score: ~0.99
- Accuracy: ~97%
- Precision: ~98%
- Recall: ~96%

## Key Findings

1. Most important predictive features:
   - Concave points (worst)
   - Area (worst)
   - Perimeter (worst)

2. Feature correlations:
   - Strong correlations between radius, perimeter, and area
   - SE measurements showed lower predictive power

3. Model behavior:
   - Excellent malignant/benign separation
   - Robust cross-validation performance
   - Clear decision boundaries

## Technical Stack

### Machine Learning
- Python
- scikit-learn
- pandas
- numpy
- SHAP

### Web Application
#### Backend
- Flask
- Flask-CORS
- joblib

#### Frontend
- React
- Vite
- Tailwind CSS
- Lucide React

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Frontend Development Notes

After initial setup, several configuration files are automatically generated:
- `package-lock.json`: Dependency version lock
- `eslint.config.js`: Code linting rules
- `postcss.config.js`: CSS processing
- `tailwind.config.js`: Tailwind CSS settings
- `vite.config.js`: Build and development server

## Security Considerations

- Input validation on both frontend and backend
- CORS protection
- Error message sanitization
- No sensitive data storage

## License

MIT License

## Support

For support, please open an issue in the GitHub repository.