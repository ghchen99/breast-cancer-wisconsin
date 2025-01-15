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
    """Validate input values are within reasonable ranges based on training data"""
    # These ranges are approximate based on the original dataset
    validation_rules = {
        'radius': (6.0, 30.0),
        'texture': (9.0, 40.0),
        'perimeter': (40.0, 190.0),
        'area': (140.0, 2600.0),
        'smoothness': (0.05, 0.16),
        'compactness': (0.02, 0.35),
        'concavity': (0.0, 0.5),
        'concave_points': (0.0, 0.2),
        'symmetry': (0.1, 0.3),
        'fractal_dimension': (0.05, 0.1)
    }

    for feature in FEATURE_ORDER:
        base_feature = feature.split('_')[0]
        if base_feature in validation_rules:
            min_val, max_val = validation_rules[base_feature]
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