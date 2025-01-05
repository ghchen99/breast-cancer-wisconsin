import logging
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, PolynomialFeatures
from scipy.stats import skew, zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

class DataPipeline:
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.scalers = {}
        self.encoders = {}
        self.transformers = {}
        self.poly = None
        self.top_interactions = None
        self.median_values = {}
        self.cols_to_drop = []
        
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets before any preprocessing."""
        logging.info("Splitting data into train and test sets")
        
        X = df.drop(['diagnosis', 'id'], axis=1)  # Dropping target and unique identifier columns
        y = df['diagnosis']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def clean_missing_values(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> tuple:
        """Handle missing values using only training data statistics."""
        logging.info("Cleaning missing values")
        
        if test_df is None:
            test_df = pd.DataFrame()  # Empty DataFrame for single dataset processing
        
        # Drop columns with more than 30% missing values in training set
        if not self.cols_to_drop:  # Only compute during training
            threshold = 0.3 * len(train_df)
            self.cols_to_drop = [col for col in train_df.columns 
                               if train_df[col].isnull().sum() > threshold]
            
        train_df = train_df.drop(columns=self.cols_to_drop)
        if not test_df.empty:
            test_df = test_df.drop(columns=self.cols_to_drop)
        
        # Fill missing values using training data medians
        for col in train_df.columns:
            if train_df[col].isnull().any():
                if col not in self.median_values:  # Only compute during training
                    self.median_values[col] = train_df[col].median()
                
                train_df[col] = train_df[col].fillna(self.median_values[col])
                if not test_df.empty:
                    test_df[col] = test_df[col].fillna(self.median_values[col])
        
        return (train_df, test_df) if not test_df.empty else train_df
    
    def detect_skewness(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> tuple:
        """Correct skewed features using PowerTransformer."""
        logging.info("Correcting skewed features")
        
        if test_df is None:
            test_df = pd.DataFrame()
        
        numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        skewed_cols = [col for col in numerical_cols 
                      if abs(skew(train_df[col])) > 0.5 and col not in ['diagnosis']]
        
        if skewed_cols:
            if 'power' not in self.transformers:  # Only fit during training
                self.transformers['power'] = PowerTransformer(method='yeo-johnson')
                train_df[skewed_cols] = self.transformers['power'].fit_transform(train_df[skewed_cols])
            else:
                train_df[skewed_cols] = self.transformers['power'].transform(train_df[skewed_cols])
            
            if not test_df.empty:
                test_df[skewed_cols] = self.transformers['power'].transform(test_df[skewed_cols])
        
        return (train_df, test_df) if not test_df.empty else train_df
    
    def handle_outliers(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, 
                       threshold: float = 3.0) -> tuple:
        """Handle outliers in numerical features."""
        logging.info("Handling outliers")
        
        if test_df is None:
            test_df = pd.DataFrame()
        
        numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
        numerical_cols = [col for col in numerical_cols if col not in ['diagnosis', 'id']]
        
        for col in numerical_cols:
            if col not in self.median_values:  # Only compute during training
                z_scores = zscore(train_df[col])
                self.median_values[col] = train_df[col].median()
                train_df.loc[abs(z_scores) > threshold, col] = self.median_values[col]
            else:
                z_scores = zscore(train_df[col])
                train_df.loc[abs(z_scores) > threshold, col] = self.median_values[col]
            
            if not test_df.empty:
                z_scores = zscore(test_df[col])
                test_df.loc[abs(z_scores) > threshold, col] = self.median_values[col]
        
        return (train_df, test_df) if not test_df.empty else train_df
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features between different measurements."""
        logging.info("Creating ratio features")
        
        df['radius_to_perimeter_ratio'] = df['radius_mean'] / df['perimeter_mean']
        df['area_to_perimeter_ratio'] = df['area_mean'] / df['perimeter_mean']
        df['compactness_to_concavity_ratio'] = df['compactness_mean'] / (df['concavity_mean'] + 1e-6)
        df['symmetry_to_fractal_ratio'] = df['symmetry_mean'] / (df['fractal_dimension_mean'] + 1e-6)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features comparing mean, SE, and worst values."""
        logging.info("Creating statistical features")
        
        base_features = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                        'compactness', 'concavity', 'concave points', 'symmetry',
                        'fractal_dimension']
        
        for feature in base_features:
            mean_col = f'{feature}_mean'
            se_col = f'{feature}_se'
            worst_col = f'{feature}_worst'
            
            if all(col in df.columns for col in [mean_col, se_col, worst_col]):
                df[f'{feature}_cv'] = df[se_col] / df[mean_col]
                df[f'{feature}_worst_to_mean'] = df[worst_col] / df[mean_col]
                df[f'{feature}_range'] = df[worst_col] - df[mean_col]
        
        return df
    
    def create_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to cell shape and geometry."""
        logging.info("Creating shape features")
        
        df['circularity_mean'] = (4 * np.pi * df['area_mean']) / (df['perimeter_mean'] ** 2)
        df['roundness_mean'] = (4 * df['area_mean']) / (np.pi * (2 * df['radius_mean']) ** 2)
        df['area_radius_deviation'] = df['area_mean'] / (np.pi * df['radius_mean'] ** 2)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, target=None, degree: int = 2) -> pd.DataFrame:
        """Create polynomial interaction features for mean measurements."""
        logging.info(f"Creating interaction features with degree {degree}")
        
        mean_features = [col for col in df.columns if col.endswith('_mean') 
                        and col not in ['id', 'diagnosis']]
        
        if self.poly is None:  # Only fit during training
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = self.poly.fit_transform(df[mean_features])
            
            if target is not None:
                feature_names = self.poly.get_feature_names_out(mean_features)
                interaction_features = pd.DataFrame(
                    poly_features[:, len(mean_features):],
                    columns=feature_names[len(mean_features):],
                    index=df.index
                )
                correlations = abs(interaction_features.corrwith(target))
                self.top_interactions = correlations.nlargest(10).index
                interaction_features = interaction_features[self.top_interactions]
            else:
                interaction_features = pd.DataFrame(
                    poly_features[:, len(mean_features):],
                    columns=self.poly.get_feature_names_out(mean_features)[len(mean_features):],
                    index=df.index
                )
        else:
            poly_features = self.poly.transform(df[mean_features])
            interaction_features = pd.DataFrame(
                poly_features[:, len(mean_features):],
                columns=self.poly.get_feature_names_out(mean_features)[len(mean_features):],
                index=df.index
            )
            if self.top_interactions is not None:
                interaction_features = interaction_features[self.top_interactions]
        
        return pd.concat([df, interaction_features], axis=1)
    
    def create_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture the complexity and variability of measurements."""
        logging.info("Creating complexity features")
        
        complexity_features = ['compactness_mean', 'concavity_mean', 'fractal_dimension_mean']
        df['overall_complexity'] = df[complexity_features].mean(axis=1)
        
        variability_features = [col for col in df.columns if col.endswith('_se')]
        df['measurement_variability'] = df[variability_features].mean(axis=1)
        
        worst_features = [col for col in df.columns if col.endswith('_worst')]
        df['severity_score'] = df[worst_features].mean(axis=1)
        
        return df
    
    def process_data(self, df: pd.DataFrame, is_training: bool = True) -> tuple:
        """Main processing pipeline."""
        try:
            if is_training:
                logging.info("Processing training data")
                # First split the data
                X_train, X_test, y_train, y_test = self.split_data(df)
                
                # Encode target labels (diagnosis) as numeric
                self.encoders['diagnosis'] = LabelEncoder()
                y_train = self.encoders['diagnosis'].fit_transform(y_train)
                y_test = self.encoders['diagnosis'].transform(y_test)

                # Convert y_train and y_test back to pandas Series
                y_train = pd.Series(y_train, index=X_train.index)
                y_test = pd.Series(y_test, index=X_test.index)

                # Clean training data
                X_train, X_test = self.clean_missing_values(X_train, X_test)
                X_train, X_test = self.detect_skewness(X_train, X_test)
                X_train, X_test = self.handle_outliers(X_train, X_test)
                
                # Feature engineering
                X_train = self.create_ratio_features(X_train)
                X_train = self.create_statistical_features(X_train)
                X_train = self.create_shape_features(X_train)
                X_train = self.create_complexity_features(X_train)
                X_train = self.create_interaction_features(X_train, y_train)
                
                X_test = self.create_ratio_features(X_test)
                X_test = self.create_statistical_features(X_test)
                X_test = self.create_shape_features(X_test)
                X_test = self.create_complexity_features(X_test)
                X_test = self.create_interaction_features(X_test)

                # Final scaling
                numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
                numerical_cols = [col for col in numerical_cols if col != 'id']
                
                self.scalers['standard'] = StandardScaler()
                X_train[numerical_cols] = self.scalers['standard'].fit_transform(X_train[numerical_cols])
                X_test[numerical_cols] = self.scalers['standard'].transform(X_test[numerical_cols])
                
                return X_train, X_test, y_train, y_test
            
            else:
                logging.info("Processing prediction data")
                # Apply all transformations in sequence
                X_new = self.clean_missing_values(df.copy())
                X_new = self.detect_skewness(X_new)
                X_new = self.handle_outliers(X_new)
                X_new = self.create_ratio_features(X_new)
                X_new = self.create_statistical_features(X_new)
                X_new = self.create_shape_features(X_new)
                X_new = self.create_complexity_features(X_new)
                X_new = self.create_interaction_features(X_new)

                # Apply scaling
                numerical_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
                numerical_cols = [col for col in numerical_cols if col != 'id']
                X_new[numerical_cols] = self.scalers['standard'].transform(X_new[numerical_cols])
                
                return X_new
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise



def main():
    """Main execution function."""
    try:
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Load raw data
        df = pd.read_csv('data/raw_dataset.csv')
        logging.info("Dataset loaded successfully")
        df['id'] = df['id'].astype(str)
        
        # Initialize and run pipeline
        pipeline = DataPipeline(random_state=42, test_size=0.2)
        X_train, X_test, y_train, y_test = pipeline.process_data(df)
        
        # Save processed datasets
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
