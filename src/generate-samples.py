import pandas as pd
import numpy as np
import joblib
import logging
from scipy.stats import zscore

class TestDataGenerator:
    """
    Generates test data matching the feature engineering pipeline used in training.
    Ensures consistency with the FeatureEngineering class processing.
    """
    def __init__(self, random_state=42, models_path='models/'):
        self.random_state = random_state
        self.models_path = models_path
        np.random.seed(random_state)
        
        # Load saved transformers
        self.load_transformers()
        
    def load_transformers(self):
        """Load transformers used during training."""
        try:
            self.power_transformer = joblib.load(f'{self.models_path}/power_transformer.joblib')
            self.standard_scaler = joblib.load(f'{self.models_path}/standard_scaler.joblib')
            self.median_values = joblib.load(f'{self.models_path}/median_values.joblib')
            self.label_encoder = joblib.load(f'{self.models_path}/label_encoder.joblib')
        except Exception as e:
            logging.error(f"Error loading transformers: {str(e)}")
            raise
            
    def get_label_classes(self):
        """Get the classes from the label encoder."""
        if self.label_encoder is not None:
            return self.label_encoder.classes_
        return None
    
    def encode_labels(self, labels):
        """Encode labels using the loaded label encoder."""
        if self.label_encoder is not None:
            return self.label_encoder.transform(labels)
        raise ValueError("Label encoder not loaded")
    
    def decode_labels(self, encoded_labels):
        """Decode labels using the loaded label encoder."""
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(encoded_labels)
        raise ValueError("Label encoder not loaded")
    
    def _generate_base_features(self, n_samples):
        """Generate base features with realistic ranges."""
        return pd.DataFrame({
            'id': [f'TEST{i:03d}' for i in range(1, n_samples + 1)],
            'radius_mean': np.random.uniform(6, 28, n_samples),
            'texture_mean': np.random.uniform(9, 40, n_samples),
            'perimeter_mean': np.random.uniform(43, 190, n_samples),
            'area_mean': np.random.uniform(140, 2550, n_samples),
            'smoothness_mean': np.random.uniform(0.05, 0.16, n_samples),
            'compactness_mean': np.random.uniform(0.02, 0.35, n_samples),
            'concavity_mean': np.random.uniform(0, 0.43, n_samples),
            'concave points_mean': np.random.uniform(0, 0.2, n_samples),
            'symmetry_mean': np.random.uniform(0.1, 0.3, n_samples),
            'fractal_dimension_mean': np.random.uniform(0.05, 0.1, n_samples),
            'radius_se': np.random.uniform(0.1, 2.9, n_samples),
            'texture_se': np.random.uniform(0.3, 4.9, n_samples),
            'perimeter_se': np.random.uniform(0.7, 22, n_samples),
            'area_se': np.random.uniform(6.8, 542, n_samples),
            'smoothness_se': np.random.uniform(0.002, 0.03, n_samples),
            'compactness_se': np.random.uniform(0.002, 0.14, n_samples),
            'concavity_se': np.random.uniform(0, 0.4, n_samples),
            'concave points_se': np.random.uniform(0, 0.05, n_samples),
            'symmetry_se': np.random.uniform(0.008, 0.08, n_samples),
            'fractal_dimension_se': np.random.uniform(0.001, 0.03, n_samples),
            'radius_worst': np.random.uniform(7.9, 36, n_samples),
            'texture_worst': np.random.uniform(12, 50, n_samples),
            'perimeter_worst': np.random.uniform(50, 250, n_samples),
            'area_worst': np.random.uniform(185, 4255, n_samples),
            'smoothness_worst': np.random.uniform(0.07, 0.22, n_samples),
            'compactness_worst': np.random.uniform(0.03, 1.06, n_samples),
            'concavity_worst': np.random.uniform(0, 1.25, n_samples),
            'concave points_worst': np.random.uniform(0, 0.29, n_samples),
            'symmetry_worst': np.random.uniform(0.15, 0.66, n_samples),
            'fractal_dimension_worst': np.random.uniform(0.055, 0.21, n_samples)
        })

    def handle_missing_values(self, df):
        """Handle any missing values using stored median values."""
        for col in df.columns:
            if col in self.median_values and df[col].isnull().any():
                df[col] = df[col].fillna(self.median_values[col])
        return df

    def apply_power_transform(self, df):
        """Apply power transformation to handle skewness."""
        if self.power_transformer is not None:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            transform_cols = [col for col in numerical_cols 
                          if col in self.power_transformer.feature_names_in_]
            
            if transform_cols:
                df[transform_cols] = self.power_transformer.transform(df[transform_cols])
        return df

    def handle_outliers(self, df, threshold=3.0):
        """Handle outliers using z-score method."""
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'id']
        
        for col in numerical_cols:
            if col in self.median_values:
                z_scores = zscore(df[col])
                df.loc[abs(z_scores) > threshold, col] = self.median_values[col]
        return df

    def create_key_ratios(self, df):
        """Create key ratio features matching FeatureEngineering class."""
        # Basic ratios
        df['radius_to_perimeter_ratio'] = df['radius_mean'] / df['perimeter_mean']
        df['area_to_perimeter_ratio'] = df['area_mean'] / df['perimeter_mean']
        return df

    def create_core_statistical_features(self, df):
        """Create core statistical features matching FeatureEngineering class."""
        key_features = ['radius', 'area', 'concavity']
        
        for feature in key_features:
            mean_col = f'{feature}_mean'
            worst_col = f'{feature}_worst'
            if all(col in df.columns for col in [mean_col, worst_col]):
                df[f'{feature}_worst_to_mean'] = df[worst_col] / df[mean_col]
        return df

    def create_shape_features(self, df):
        """Create shape-related features matching FeatureEngineering class."""
        # Calculate circularity
        df['circularity'] = (4 * np.pi * df['area_mean']) / (df['perimeter_mean'] ** 2)
        return df

    def apply_scaling(self, df):
        """Apply standard scaling using stored scaler."""
        if self.standard_scaler is not None:
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col != 'id']
            df[numerical_cols] = self.standard_scaler.transform(df[numerical_cols])
        return df

    def generate_data(self, n_samples=10, include_labels=False):
        """
        Generate complete test dataset following the exact same pipeline as training.
        """
        try:
            # Generate base features
            df = self._generate_base_features(n_samples)
            
            # Apply preprocessing and feature engineering
            df = self.handle_missing_values(df)
            df = self.apply_power_transform(df)
            df = self.handle_outliers(df)
            df = self.create_key_ratios(df)
            df = self.create_core_statistical_features(df)
            df = self.create_shape_features(df)
            df = self.apply_scaling(df)
            
            if include_labels:
                # Generate random labels from actual classes
                classes = self.get_label_classes()
                if classes is not None:
                    labels = np.random.choice(classes, size=n_samples)
                    df['diagnosis'] = labels
                    
            return df
            
        except Exception as e:
            logging.error(f"Error generating test data: {str(e)}")
            raise

def main():
    """Generate and save test samples with engineered features."""
    try:
        # Create test data generator
        generator = TestDataGenerator()
        
        # Generate test samples
        df = generator.generate_data(n_samples=100)
        
        # Save to CSV
        df.to_csv('data/inference_samples.csv', index=False)
        
        print("Created inference_samples.csv with the following data:")
        print("\nShape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        print("\nFeature statistics:")
        print(df.describe().round(3))
        
    except Exception as e:
        print(f"Error generating test data: {str(e)}")
        raise

if __name__ == "__main__":
    main()