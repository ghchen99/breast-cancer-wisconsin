import logging
import warnings
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from scipy.stats import skew, zscore
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self, random_state=42, test_size=0.2, models_path='models'):
        self.random_state = random_state
        self.test_size = test_size
        self.models_path = models_path
        self.power_transformer = None
        self.standard_scaler = None
        self.encoders = {}
        self.median_values = {}
        self.cols_to_drop = []
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            
    def save_transformers(self):
        """Save all transformers and values to disk."""
        logging.info("Saving transformers and values to disk")
        if self.power_transformer:
            joblib.dump(self.power_transformer, f'{self.models_path}/power_transformer.joblib')
        if self.standard_scaler:
            joblib.dump(self.standard_scaler, f'{self.models_path}/standard_scaler.joblib')
        if self.median_values:
            joblib.dump(self.median_values, f'{self.models_path}/median_values.joblib')
            
    def load_transformers(self):
        """Load all transformers and values from disk."""
        logging.info("Loading transformers and values from disk")
        try:
            self.power_transformer = joblib.load(f'{self.models_path}/power_transformer.joblib')
            self.standard_scaler = joblib.load(f'{self.models_path}/standard_scaler.joblib')
            self.median_values = joblib.load(f'{self.models_path}/median_values.joblib')
        except FileNotFoundError as e:
            logging.warning(f"Could not load transformers: {str(e)}")
            raise
        
    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train and test sets before any preprocessing."""
        logging.info("Splitting data into train and test sets")
        
        X = df.drop(['diagnosis', 'id'], axis=1)
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
            test_df = pd.DataFrame()
        
        # Drop columns with more than 30% missing values in training set
        if not self.cols_to_drop:
            threshold = 0.3 * len(train_df)
            self.cols_to_drop = [col for col in train_df.columns 
                               if train_df[col].isnull().sum() > threshold]
            
        train_df = train_df.drop(columns=self.cols_to_drop)
        if not test_df.empty:
            test_df = test_df.drop(columns=self.cols_to_drop)
        
        # Fill missing values using training data medians
        for col in train_df.columns:
            if train_df[col].isnull().any():
                if col not in self.median_values:
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
            if self.power_transformer is None:
                self.power_transformer = PowerTransformer(method='yeo-johnson')
                train_df[skewed_cols] = self.power_transformer.fit_transform(train_df[skewed_cols])
            else:
                train_df[skewed_cols] = self.power_transformer.transform(train_df[skewed_cols])
            
            if not test_df.empty:
                test_df[skewed_cols] = self.power_transformer.transform(test_df[skewed_cols])
        
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
            if col not in self.median_values:
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
    
    def create_key_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create only the most important ratio features."""
        logging.info("Creating key ratio features")
        
        df['radius_to_perimeter_ratio'] = df['radius_mean'] / df['perimeter_mean']
        df['area_to_perimeter_ratio'] = df['area_mean'] / df['perimeter_mean']
        
        return df
    
    def create_core_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential statistical features."""
        logging.info("Creating core statistical features")
        
        key_features = ['radius', 'area', 'concavity']
        
        for feature in key_features:
            mean_col = f'{feature}_mean'
            worst_col = f'{feature}_worst'
            
            if all(col in df.columns for col in [mean_col, worst_col]):
                df[f'{feature}_worst_to_mean'] = df[worst_col] / df[mean_col]
        
        return df
    
    def create_basic_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental shape features."""
        logging.info("Creating basic shape features")
        
        df['circularity'] = (4 * np.pi * df['area_mean']) / (df['perimeter_mean'] ** 2)
        
        return df
    
    def process_data(self, df: pd.DataFrame, is_training: bool = True) -> tuple:
        """Main processing pipeline with reduced feature engineering."""
        try:
            if is_training:
                logging.info("Processing training data")
                X_train, X_test, y_train, y_test = self.split_data(df)
                
                self.encoders['diagnosis'] = LabelEncoder()
                y_train = self.encoders['diagnosis'].fit_transform(y_train)
                y_test = self.encoders['diagnosis'].transform(y_test)

                y_train = pd.Series(y_train, index=X_train.index)
                y_test = pd.Series(y_test, index=X_test.index)

                X_train, X_test = self.clean_missing_values(X_train, X_test)
                X_train, X_test = self.detect_skewness(X_train, X_test)
                X_train, X_test = self.handle_outliers(X_train, X_test)
                
                X_train = self.create_key_ratios(X_train)
                X_train = self.create_core_statistical_features(X_train)
                X_train = self.create_basic_shape_features(X_train)
                
                X_test = self.create_key_ratios(X_test)
                X_test = self.create_core_statistical_features(X_test)
                X_test = self.create_basic_shape_features(X_test)

                numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
                numerical_cols = [col for col in numerical_cols if col != 'id']
                
                self.standard_scaler = StandardScaler()
                X_train[numerical_cols] = self.standard_scaler.fit_transform(X_train[numerical_cols])
                X_test[numerical_cols] = self.standard_scaler.transform(X_test[numerical_cols])
                
                # Save transformers after fitting
                self.save_transformers()
                
                return X_train, X_test, y_train, y_test
            
            else:
                logging.info("Processing prediction data")
                # Load transformers if not already loaded
                if self.power_transformer is None or self.standard_scaler is None:
                    self.load_transformers()
                    
                X_new = self.clean_missing_values(df.copy())
                X_new = self.detect_skewness(X_new)
                X_new = self.handle_outliers(X_new)
                X_new = self.create_key_ratios(X_new)
                X_new = self.create_core_statistical_features(X_new)
                X_new = self.create_basic_shape_features(X_new)

                numerical_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
                numerical_cols = [col for col in numerical_cols if col != 'id']
                X_new[numerical_cols] = self.standard_scaler.transform(X_new[numerical_cols])
                
                return X_new
                
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
        
        df = pd.read_csv('data/raw_dataset.csv')
        logging.info("Dataset loaded successfully")
        df['id'] = df['id'].astype(str)
        
        pipeline = FeatureEngineering(random_state=42, test_size=0.2)
        X_train, X_test, y_train, y_test = pipeline.process_data(df)
        
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
        
        print("\nProcessing Summary:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of features: {X_train.shape[1]}")
        
        logging.info("Data processing completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()