import logging
import warnings
import os
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataset_loader import DatasetLoader

# Constants
RESULTS_DIR = "results/figures"
FIGURE_FORMAT = "png"
PLOT_SIZE = {
    'small': (10, 6),
    'medium': (12, 10),
    'large': (15, 6)
}

@dataclass
class PlotConfig:
    """Configuration for plot settings."""
    n_cols: int = 3
    figsize_multiplier: tuple = (15, 5)
    rotation: int = 45

class EDAProcessor:
    """Class to handle Exploratory Data Analysis operations."""
    
    def __init__(self, output_dir: str = RESULTS_DIR):
        self.output_dir = output_dir
        self._setup_environment()
        
    def _setup_environment(self) -> None:
        """Set up logging and create necessary directories."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        warnings.filterwarnings('ignore')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_plot(self, plot_name: str) -> None:
        """Save the current plot to the output directory."""
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{plot_name}.{FIGURE_FORMAT}')
        plt.savefig(output_path)
        logging.info(f"Plot saved as '{output_path}'")
        plt.close()

    def analyze_missing_values(self, df: pd.DataFrame) -> None:
        """Analyze and visualize missing values in the dataset."""
        logging.info("Analyzing missing values.")
        missing_values = df.isnull().sum()
        
        if not missing_values.any():
            logging.info("No missing values found.")
            print("\nNo missing values found in the dataset.")
            return
            
        logging.info("Missing values detected.")
        print("\nMissing Values Analysis:")
        print(missing_values[missing_values > 0])
        
        plt.figure(figsize=PLOT_SIZE['small'])
        missing_values.plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xticks(rotation=PlotConfig.rotation)
        self.save_plot('missing_values_by_feature')

    def analyze_feature_distributions(self, df: pd.DataFrame) -> None:
        """Analyze and visualize feature distributions."""
        logging.info("Analyzing feature distributions.")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        n_rows = (len(numerical_cols) + PlotConfig.n_cols - 1) // PlotConfig.n_cols
        plt.figure(figsize=(
            PlotConfig.figsize_multiplier[0],
            PlotConfig.figsize_multiplier[1] * n_rows
        ))
        
        for idx, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, PlotConfig.n_cols, idx)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=PlotConfig.rotation)
        
        self.save_plot('feature_distributions')
        
        print("\nNumerical Features Statistics:")
        print(df[numerical_cols].describe())

    def analyze_correlations(self, df: pd.DataFrame) -> None:
        """Analyze and visualize feature correlations."""
        logging.info("Analyzing correlations.")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=PLOT_SIZE['medium'])
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            square=True
        )
        plt.title('Feature Correlation Heatmap')
        plt.xticks(rotation=PlotConfig.rotation)
        plt.yticks(rotation=PlotConfig.rotation)
        self.save_plot('correlation_heatmap')
        
        self._print_top_correlations(correlation_matrix)

    def _print_top_correlations(self, correlation_matrix: pd.DataFrame, top_n: int = 10) -> None:
        """Print the top N feature correlations."""
        logging.info("Identifying top correlations.")
        correlations = correlation_matrix.unstack()
        correlations = correlations[correlations != 1.0]
        top_correlations = correlations.sort_values(ascending=False)[:top_n]
        print(f"\nTop {top_n} Feature Correlations:")
        print(top_correlations)

    def analyze_target_relationship(self, df: pd.DataFrame, target_col: str) -> None:
        """Analyze relationships between features and target variable."""
        logging.info(f"Analyzing relationships with target variable: {target_col}")
        numerical_cols = [
            col for col in df.select_dtypes(include=['int64', 'float64']).columns 
            if col != target_col
        ]
        
        n_rows = (len(numerical_cols) + PlotConfig.n_cols - 1) // PlotConfig.n_cols
        plt.figure(figsize=(
            PlotConfig.figsize_multiplier[0],
            PlotConfig.figsize_multiplier[1] * n_rows
        ))
        
        for idx, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, PlotConfig.n_cols, idx)
            sns.boxplot(data=df, x=target_col, y=col)
            plt.title(f'{col} by {target_col}')
        
        self.save_plot(f'target_relationship_{target_col}')
        
        print(f"\nTarget Variable ({target_col}) Distribution:")
        print(df[target_col].value_counts(normalize=True))

    def analyze_outliers(self, df: pd.DataFrame) -> None:
        """Analyze and visualize outliers in the dataset."""
        logging.info("Analyzing outliers.")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        plt.figure(figsize=PLOT_SIZE['large'])
        sns.boxplot(data=df[numerical_cols])
        plt.title('Outlier Detection Using Box Plots')
        plt.xticks(rotation=PlotConfig.rotation)
        self.save_plot('outlier_detection')
        
        self._print_outlier_statistics(df, numerical_cols)

    def _print_outlier_statistics(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Calculate and print outlier statistics for given columns."""
        print("\nOutlier Analysis:")
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
            
            if len(outliers) > 0:
                logging.warning(f"Outliers detected in {col}.")
                print(f"\n{col}:")
                print(f"Number of outliers: {len(outliers)}")
                print(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")

    def perform_eda(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """Perform complete exploratory data analysis."""
        self.analyze_missing_values(df)
        self.analyze_feature_distributions(df)
        self.analyze_correlations(df)
        
        if target_col and target_col in df.columns:
            self.analyze_target_relationship(df, target_col)
        
        self.analyze_outliers(df)

def main():
    """Main execution function for EDA."""
    try:
        logging.info("Starting EDA process.")
        
        # Load and process dataset
        processor = DatasetLoader()
        df, output_path = processor.process()
        df = pd.read_csv(output_path)
        df['id'] = df['id'].astype(str)
        
        # Print initial dataset information
        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print("\nFeature Information:")
        print(df.info())
        
        # Perform EDA
        eda_processor = EDAProcessor()
        eda_processor.perform_eda(df, target_col='diagnosis')
        
        logging.info("EDA process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()