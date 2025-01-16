from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained pipeline (includes both StandardScaler and RandomForestClassifier)
pipeline = joblib.load('models/cancer_diagnosis_model.joblib')

# Define the expected feature order (matches the order used in training)
FEATURE_ORDER = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def validate_input_ranges(data):
    """Validate input values are within reasonable ranges based on representative sample data analysis"""
    validation_rules = {
        # Mean values
        'radius_mean': (10.07, 22.51),
        'texture_mean': (15.26, 24.21),
        'perimeter_mean': (65.13, 149.65),
        'area_mean': (233.14, 1484.66),
        'smoothness_mean': (0.069, 0.124),
        'compactness_mean': (0.01, 0.276),
        'concavity_mean': (0.01, 0.351),
        'concave points_mean': (0.01, 0.158),
        'symmetry_mean': (0.128, 0.222),
        'fractal_dimension_mean': (0.051, 0.076),
        
        # Standard Error values
        'radius_se': (0.15, 1.012),
        'texture_se': (0.289, 2.310),
        'perimeter_se': (0.735, 5.360),
        'area_se': (10.0, 130.44),
        'smoothness_se': (0.0039, 0.0114),
        'compactness_se': (0.01, 0.0772),
        'concavity_se': (0.001, 0.1014),
        'concave points_se': (0.0006, 0.0274),
        'symmetry_se': (0.0085, 0.0282),
        'fractal_dimension_se': (0.001, 0.0075),
        
        # Worst values
        'radius_worst': (9.95, 25.59),
        'texture_worst': (17.42, 34.05),
        'perimeter_worst': (71.73, 163.14),
        'area_worst': (189.64, 1929.16),
        'smoothness_worst': (0.089, 0.184),
        'compactness_worst': (0.01, 0.893),
        'concavity_worst': (0.01, 1.162),
        'concave points_worst': (0.01, 0.311),
        'symmetry_worst': (0.182, 0.366),
        'fractal_dimension_worst': (0.044, 0.132)
    }

    for feature in FEATURE_ORDER:
        if feature in validation_rules:
            min_val, max_val = validation_rules[feature]
            value = data[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Feature {feature} with value {value} is outside expected "
                    f"range [{min_val}, {max_val}]"
                )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert input data to DataFrame with correct feature order
        features_dict = {feature: float(data[feature]) for feature in FEATURE_ORDER}
        
        # Basic validation
        validate_input_ranges(features_dict)
        
        # Create DataFrame with single row
        df = pd.DataFrame([features_dict])
        
        # Make prediction using the pipeline
        # Pipeline will handle the scaling internally
        prediction_proba = pipeline.predict_proba(df)[0]
        
        # Get detailed prediction info
        diagnosis = 'Malignant' if prediction_proba[1] > 0.5 else 'Benign'
        confidence = float(max(prediction_proba))
        
        # Determine confidence level
        if confidence >= 0.9:
            confidence_level = 'Very High'
        elif confidence >= 0.8:
            confidence_level = 'High'
        elif confidence >= 0.7:
            confidence_level = 'Moderate'
        else:
            confidence_level = 'Low'
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'malignant_probability': float(prediction_proba[1]),
            'benign_probability': float(prediction_proba[0]),
            'warning': 'Low confidence prediction, consider additional testing.' if confidence < 0.8 else None
        })
        
    except KeyError as e:
        return jsonify({
            'error': f'Missing feature: {str(e)}',
            'required_features': FEATURE_ORDER
        }), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)