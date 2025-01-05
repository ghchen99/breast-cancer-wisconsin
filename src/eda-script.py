import logging
import warnings
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings('ignore')

# Create the directory for saving plots if it doesn't exist
if not os.path.exists("results/figures"):
    os.makedirs("results/figures")

def save_plot(plot_name: str) -> None:
    """
    Save the current plot to the 'results/figures' directory.
    
    Args:
        plot_name (str): Name of the plot file to save.
    """
    plt.tight_layout()
    plt.savefig(f'results/figures/{plot_name}.png')
    logging.info(f"Plot saved as 'results/figures/{plot_name}.png'")

def analyze_missing_values(df: pd.DataFrame) -> None:
    logging.info("Analyzing missing values.")
    missing_values = df.isnull().sum()
    if missing_values.any():
        logging.info("Missing values detected.")
        print("\nMissing Values Analysis:")
        print(missing_values[missing_values > 0])
        
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        missing_values.plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xticks(rotation=45)
        save_plot('missing_values_by_feature')
    else:
        logging.info("No missing values found.")
        print("\nNo missing values found in the dataset.")

def analyze_feature_distributions(df: pd.DataFrame) -> None:
    logging.info("Analyzing feature distributions.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create distribution plots
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    save_plot('feature_distributions')
    
    # Calculate basic statistics
    logging.info("Calculating descriptive statistics.")
    print("\nNumerical Features Statistics:")
    print(df[numerical_cols].describe())

def analyze_correlations(df: pd.DataFrame) -> None:
    logging.info("Analyzing correlations.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    save_plot('correlation_heatmap')
    
    # Print highest correlations
    logging.info("Identifying top correlations.")
    print("\nTop 10 Feature Correlations:")
    correlations = correlation_matrix.unstack()
    correlations = correlations[correlations != 1.0]
    correlations = correlations.sort_values(ascending=False)[:10]
    print(correlations)

def analyze_target_relationship(df: pd.DataFrame, target_col: str) -> None:
    logging.info(f"Analyzing relationships with target variable: {target_col}.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    # Create box plots
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} by {target_col}')
    
    save_plot(f'target_relationship_{target_col}')
    
    # Print target value distribution
    logging.info("Displaying target variable distribution.")
    print(f"\nTarget Variable ({target_col}) Distribution:")
    print(df[target_col].value_counts(normalize=True))

def analyze_outliers(df: pd.DataFrame) -> None:
    logging.info("Analyzing outliers.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create box plots for outlier detection
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[numerical_cols])
    plt.title('Outlier Detection Using Box Plots')
    plt.xticks(rotation=45)
    save_plot('outlier_detection')
    
    # Calculate and print outlier statistics
    print("\nOutlier Analysis:")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        if len(outliers) > 0:
            logging.warning(f"Outliers detected in {col}.")
            print(f"\n{col}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")

def perform_eda(df: pd.DataFrame) -> None:
    """Perform all exploratory data analysis steps."""
    analyze_missing_values(df)
    analyze_feature_distributions(df)
    analyze_correlations(df)
    
    if 'diagnosis' in df.columns:
        analyze_target_relationship(df, 'diagnosis')
    
    analyze_outliers(df)

def main():
    """Main execution function for EDA."""
    try:
        logging.info("Starting EDA process.")
        df = pd.read_csv('data/raw_dataset.csv')
        logging.info("Dataset loaded successfully")
        
        df['id'] = df['id'].astype(str)

        logging.info("Dataset loaded successfully.")
        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print("\nFeature Information:")
        print(df.info())
        
        # Perform EDA using the new function
        perform_eda(df)

    except Exception as e:
        logging.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()
