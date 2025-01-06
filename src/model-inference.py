import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Dict, Any, List, Tuple

class ModelInference:
    """
    Class for running inference on trained breast cancer classification models.
    Handles model loading, data preprocessing, and prediction generation.
    """
    
    def __init__(self, models_path: str = 'models/'):
        """
        Initialize ModelInference with path configurations.
        
        Args:
            models_path (str): Path to directory containing trained models
        """
        self.models_path = models_path
        self.models = {}
        self.label_encoder = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.load_transformers()
        
    def load_transformers(self) -> None:
        """Load all necessary transformers including label encoder."""
        try:
            self.label_encoder = joblib.load(os.path.join(self.models_path, 'label_encoder.joblib'))
            logging.info("Successfully loaded label encoder")
        except Exception as e:
            logging.error(f"Error loading label encoder: {str(e)}")
            raise

    def load_models(self) -> Dict[str, Any]:
        """
        Load all available trained models from the models directory.
        
        Returns:
            Dictionary containing loaded models
        """
        try:
            model_files = [f for f in os.listdir(self.models_path) 
                         if f.endswith('_model.joblib')]
            
            if not model_files:
                raise FileNotFoundError("No trained models found in the specified directory")
            
            for model_file in model_files:
                model_name = model_file.replace('_model.joblib', '')
                model_path = os.path.join(self.models_path, model_file)
                
                self.models[model_name] = joblib.load(model_path)
                logging.info(f"Loaded model: {model_name}")
                
            return self.models
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise
            
    def load_model_summary(self) -> pd.DataFrame:
        """
        Load and return the model performance summary.
        
        Returns:
            DataFrame containing model performance metrics
        """
        try:
            summary_path = os.path.join(self.models_path, 'model_summary.csv')
            if not os.path.exists(summary_path):
                raise FileNotFoundError("Model summary file not found")
                
            summary_df = pd.read_csv(summary_path)
            return summary_df
            
        except Exception as e:
            logging.error(f"Error loading model summary: {str(e)}")
            raise
            
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match training data format.
        
        Args:
            input_data (pd.DataFrame): Raw input data
            
        Returns:
            Preprocessed DataFrame ready for model inference
        """
        try:
            # Remove any unnecessary columns
            features = [col for col in input_data.columns 
                       if col not in ['id', 'diagnosis']]
            
            # Ensure all required features are present
            required_features = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
                'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 
                'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 
                'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 
                'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
                'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
                'fractal_dimension_worst', 'radius_to_perimeter_ratio', 'area_to_perimeter_ratio', 
                'radius_worst_to_mean', 'area_worst_to_mean', 'concavity_worst_to_mean', 'circularity'
            ]
            
            missing_features = set(required_features) - set(features)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Return preprocessed features in correct order
            return input_data[required_features]
            
        except Exception as e:
            logging.error(f"Error preprocessing input data: {str(e)}")
            raise
            
    def get_prediction(self, 
                      model: Any, 
                      X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using specified model.
        
        Args:
            model: Trained model object
            X: Preprocessed feature DataFrame
            
        Returns:
            Tuple of (predictions, prediction probabilities)
        """
        try:
            if self.label_encoder is None:
                raise ValueError("Label encoder not loaded")
                
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Convert numeric predictions to labels
            prediction_labels = self.label_encoder.inverse_transform(predictions)
            
            return prediction_labels, probabilities
            
        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise
            
    def run_inference(self, 
                     input_data: pd.DataFrame, 
                     model_name: str = None) -> Dict[str, Any]:
        """
        Run inference on input data using specified or all available models.
        
        Args:
            input_data: Input DataFrame containing features
            model_name: Optional specific model to use (if None, uses all models)
            
        Returns:
            Dictionary containing predictions and probabilities for each model
        """
        try:
            # Load models if not already loaded
            if not self.models:
                self.load_models()
                
            # Preprocess input data
            X = self.preprocess_input(input_data)
            
            results = {}
            models_to_use = {model_name: self.models[model_name]} if model_name else self.models
            
            for name, model in models_to_use.items():
                predictions, probabilities = self.get_prediction(model, X)
                
                results[name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'prediction_df': pd.DataFrame({
                        'id': input_data['id'] if 'id' in input_data.columns else range(len(X)),
                        'predicted_class': predictions,
                        'probability_benign': probabilities[:, 0],
                        'probability_malignant': probabilities[:, 1]
                    })
                }
                
            return results
            
        except Exception as e:
            logging.error(f"Error during inference: {str(e)}")
            raise
            
    def get_ensemble_prediction(self, 
                              results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate ensemble predictions by combining results from multiple models.
        
        Args:
            results: Dictionary containing predictions from multiple models
            
        Returns:
            DataFrame with ensemble predictions
        """
        try:
            if self.label_encoder is None:
                raise ValueError("Label encoder not loaded")
                
            # Get predictions from all models
            all_probas = np.array([result['probabilities'] for result in results.values()])
            
            # Average probabilities across models
            ensemble_probas = np.mean(all_probas, axis=0)
            ensemble_predictions = (ensemble_probas[:, 1] > 0.5).astype(int)
            ensemble_labels = self.label_encoder.inverse_transform(ensemble_predictions)
            
            # Create ensemble results DataFrame
            ensemble_df = pd.DataFrame({
                'id': results[list(results.keys())[0]]['prediction_df']['id'],
                'predicted_class': ensemble_labels,
                'probability_benign': ensemble_probas[:, 0],
                'probability_malignant': ensemble_probas[:, 1]
            })
            
            return ensemble_df
            
        except Exception as e:
            logging.error(f"Error generating ensemble predictions: {str(e)}")
            raise

def main():
    """
    Example usage of ModelInference class.
    """
    try:
        # Initialize inference class
        inference = ModelInference(models_path='models/')
        
        # Print available classes from label encoder
        if inference.label_encoder is not None:
            print("\nAvailable classes:", inference.label_encoder.classes_)
        
        # Load example data (modify path as needed)
        input_data = pd.read_csv('data/inference_samples.csv')
        
        # Run inference using all available models
        results = inference.run_inference(input_data)
        
        # Generate ensemble predictions
        ensemble_predictions = inference.get_ensemble_prediction(results)
        
        # Print results
        print("\nModel Predictions:")
        for model_name, result in results.items():
            print(f"\n{model_name.upper()} Results:")
            print(result['prediction_df'])
            
        print("\nEnsemble Predictions:")
        print(ensemble_predictions)
        
        # Load and print model performance summary
        summary_df = inference.load_model_summary()
        print("\nModel Performance Summary:")
        print(summary_df)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()