import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging
import warnings
import joblib
import os
import time
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelSelector:
    """
    A class for automated model selection and evaluation.
    Handles multiple classification models with hyperparameter tuning,
    cross-validation, and comprehensive performance evaluation.
    """
    
    def __init__(self, random_state: int = 42, n_cv_folds: int = 5, n_iter: int = 20):
        """
        Initialize ModelSelector with configuration parameters.
        
        Args:
            random_state (int): Seed for reproducibility
            n_cv_folds (int): Number of cross-validation folds
            n_iter (int): Number of iterations for randomized search
        """
        self.config = {
            'random_state': random_state,
            'n_cv_folds': n_cv_folds,
            'n_iter': n_iter,
            'data_path': 'data/',
            'models_path': 'models/'
        }
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_params = None
        self.best_score = 0
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and validate the preprocessed train and test datasets.
        
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
            
        Raises:
            FileNotFoundError: If required data files are missing
            ValueError: If data validation fails
        """
        try:
            data_path = self.config['data_path']
            files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
            
            # Check if files exist
            for file in files:
                if not os.path.exists(os.path.join(data_path, file)):
                    raise FileNotFoundError(f"Missing required file: {file}")
            
            X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
            X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
            y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).iloc[:, 0]
            y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).iloc[:, 0]
            
            # Validate data
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Feature mismatch between train and test sets")
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError("Sample count mismatch in training set")
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError("Sample count mismatch in test set")
                
            # Check for missing values
            if X_train.isnull().any().any() or X_test.isnull().any().any():
                logging.warning("Missing values detected in features")
                
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize models with their hyperparameter search spaces.
        
        Returns:
            Dictionary containing model configurations and their parameter spaces
        """
        models = {
            'logistic': {
                'model': LogisticRegression(random_state=self.config['random_state'], 
                                          max_iter=1000),
                'params': {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l2'],
                    'solver': ['lbfgs'],
                    'class_weight': ['balanced', None]
                }
            },
            'rf': {
                'model': RandomForestClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'gbm': {
                'model': GradientBoostingClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'validation_fraction': [0.1],
                    'n_iter_no_change': [10],
                    'tol': [1e-4]
                }
            },
            'svm': {
                'model': SVC(random_state=self.config['random_state'], 
                           probability=True),
                'params': {
                    'C': np.logspace(-3, 3, 7),
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None]
                }
            }
        }
        return models

    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray, 
                      y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary containing various performance metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
        }

    def analyze_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from the model if available.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame containing feature importance scores or None if not available
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return None
                
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            return feature_importance.sort_values('importance', ascending=False)
            
        except Exception as e:
            logging.warning(f"Could not calculate feature importance: {str(e)}")
            return None

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models with performance monitoring.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            Dictionary containing comprehensive results for each model
        """
        cv = StratifiedKFold(n_splits=self.config['n_cv_folds'], 
                            shuffle=True, 
                            random_state=self.config['random_state'])
        models = self.initialize_models()
        results = {}
        
        for name, model_info in models.items():
            start_time = time.time()
            logging.info(f"Training {name}")
            try:
                search = RandomizedSearchCV(
                    estimator=model_info['model'],
                    param_distributions=model_info['params'],
                    n_iter=self.config['n_iter'],
                    cv=cv,
                    scoring=['roc_auc', 'precision', 'recall', 'f1'],
                    refit='roc_auc',
                    random_state=self.config['random_state'],
                    n_jobs=-1
                )
                
                # Fit with progress monitoring
                search.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Get predictions
                y_pred = search.predict(X_test)
                y_prob = search.predict_proba(X_test)
                
                # Store comprehensive results
                results[name] = {
                    'best_params': search.best_params_,
                    'cv_scores': {
                        metric: scores.mean() 
                        for metric, scores in search.cv_results_.items() 
                        if metric.startswith('mean_test_')
                    },
                    'test_metrics': self.evaluate_model(y_test, y_pred, y_prob),
                    'model': search.best_estimator_,
                    'training_time': training_time,
                    'feature_importance': self.analyze_feature_importance(
                        search.best_estimator_, 
                        X_train.columns.tolist()
                    )
                }
                
                # Update best model if necessary
                if results[name]['cv_scores']['mean_test_roc_auc'] > self.best_score:
                    self.best_score = results[name]['cv_scores']['mean_test_roc_auc']
                    self.best_model = search.best_estimator_
                    self.best_params = search.best_params_
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue
        
        return results

    def visualize_results(self, results: Dict[str, Dict[str, Any]], save_path: str = None):
        """
        Create visualizations of model performance.
        
        Args:
            results: Dictionary containing model results
            save_path: Path to save the visualizations
        """
        # Model comparison plot
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame([
            {'model': model, 'metric': metric, 'value': value}
            for model, result in results.items()
            for metric, value in result['test_metrics'].items()
        ])
        
        sns.barplot(data=metrics_df, x='model', y='value', hue='metric')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'model_comparison.png'))
        plt.close()
        
        # Feature importance plots
        for name, result in results.items():
            if result['feature_importance'] is not None:
                plt.figure(figsize=(10, 6))
                importance_df = result['feature_importance'].head(20)  # Top 20 features
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title(f'Top 20 Feature Importance - {name}')
                
                if save_path:
                    plt.savefig(os.path.join(save_path, f'{name}_feature_importance.png'))
                plt.close()

    def save_models(self, results: Dict[str, Dict[str, Any]], path: str = None):
        """
        Save trained models and results.
        
        Args:
            results: Dictionary containing model results
            path: Path to save the models and results
        """
        if path is None:
            path = self.config['models_path']
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save each model
        for name, result in results.items():
            model_path = os.path.join(path, f'{name}_model.joblib')
            joblib.dump(result['model'], model_path)
            
        # Save results summary
        summary = {name: {
            'cv_scores': result['cv_scores'],
            'test_metrics': result['test_metrics'],
            'best_params': result['best_params'],
            'training_time': result['training_time']
        } for name, result in results.items()}
        
        # Convert summary to DataFrame for CSV export
        summary_rows = []
        for model_name, model_results in summary.items():
            # CV scores
            for metric, value in model_results['cv_scores'].items():
                summary_rows.append({
                    'model': model_name,
                    'metric_type': 'cv',
                    'metric': metric,
                    'value': value
                })
            
            # Test metrics
            for metric, value in model_results['test_metrics'].items():
                summary_rows.append({
                    'model': model_name,
                    'metric_type': 'test',
                    'metric': metric,
                    'value': value
                })
                
            # Training time
            summary_rows.append({
                'model': model_name,
                'metric_type': 'performance',
                'metric': 'training_time',
                'value': model_results['training_time']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(path, 'model_summary.csv'), index=False)
        
        # Save parameters separately
        params_df = pd.DataFrame([{
            'model': name,
            'parameters': str(result['best_params'])
        } for name, result in results.items()])
        params_df.to_csv(os.path.join(path, 'best_parameters.csv'), index=False)
        
        # Save feature importance
        for name, result in results.items():
            if result['feature_importance'] is not None:
                result['feature_importance'].to_csv(
                    os.path.join(path, f'{name}_feature_importance.csv'),
                    index=False
                )

def load_model(self, model_name: str, path: str = None) -> Any:
    """
    Load a saved model.
    
    Args:
        model_name: Name of the model to load
        path: Path where models are saved
        
    Returns:
        Loaded model object
    """
    if path is None:
        path = self.config['models_path']
    
    model_path = os.path.join(path, f'{model_name}_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    return joblib.load(model_path)

def predict(self, model: Any, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Features to predict on
        
    Returns:
        Tuple of (predictions, prediction probabilities)
    """
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        return y_pred, y_prob
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise

def get_model_summary(self, path: str = None) -> pd.DataFrame:
    """
    Load and return the model summary.
    
    Args:
        path: Path where summary is saved
        
    Returns:
        DataFrame containing model summary
    """
    if path is None:
        path = self.config['models_path']
        
    summary_path = os.path.join(path, 'model_summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
        
    return pd.read_csv(summary_path)


def main():
    """Main execution function."""
    try:
        # Initialize model selector
        selector = ModelSelector(random_state=42, n_cv_folds=5, n_iter=20)
        
        # Load data
        X_train, X_test, y_train, y_test = selector.load_data()
        logging.info("Data loaded successfully")
        
        # Check class distribution
        class_dist = pd.Series(y_train).value_counts(normalize=True)
        logging.info(f"Class distribution in training set:\n{class_dist}")
        
        # Train and evaluate models
        results = selector.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        if not results:
            logging.error("No models were successfully trained")
            return
            
        # Create visualizations
        selector.visualize_results(results, save_path='results/metrics/')
        
        # Save models and results
        selector.save_models(results, path='models/')
        
        # Print comprehensive summary
        print("\nModel Performance Summary:")
        for name, result in results.items():
            print(f"\n{name.upper()} Results:")
            print(f"Best CV Score (ROC-AUC): {result['cv_scores']['mean_test_roc_auc']:.4f}")
            print(f"Training Time: {result['training_time']:.2f} seconds")
            print("\nTest Metrics:")
            for metric, value in result['test_metrics'].items():
                print(f"  {metric}: {value:.4f}")
            print("\nBest Parameters:")
            for param, value in result['best_params'].items():
                print(f"  {param}: {value}")
            
            if result['feature_importance'] is not None:
                print("\nTop 5 Important Features:")
                for _, row in result['feature_importance'].head().iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"\nBest Overall Model: {type(selector.best_model).__name__}")
        print(f"Best CV Score: {selector.best_score:.4f}")
        
        logging.info("Model selection completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during model selection: {str(e)}")
        raise

if __name__ == "__main__":
    main()