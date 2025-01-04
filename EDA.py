import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from DataLoader import download_dataset, load_csv_files
import warnings
warnings.filterwarnings('ignore')

def analyze_missing_values(df: pd.DataFrame) -> None:
    """
    Analyze and visualize missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nMissing Values Analysis:")
        print(missing_values[missing_values > 0])
        
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        missing_values.plot(kind='bar')
        plt.title('Missing Values by Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo missing values found in the dataset.")

def analyze_feature_distributions(df: pd.DataFrame) -> None:
    """
    Analyze and visualize the distribution of numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
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
    
    plt.tight_layout()
    plt.show()
    
    # Calculate basic statistics
    print("\nNumerical Features Statistics:")
    print(df[numerical_cols].describe())

def analyze_correlations(df: pd.DataFrame) -> None:
    """
    Analyze and visualize feature correlations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
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
    plt.tight_layout()
    plt.show()
    
    # Print highest correlations
    print("\nTop 10 Feature Correlations:")
    correlations = correlation_matrix.unstack()
    correlations = correlations[correlations != 1.0]
    correlations = correlations.sort_values(ascending=False)[:10]
    print(correlations)

def analyze_target_relationship(df: pd.DataFrame, target_col: str) -> None:
    """
    Analyze and visualize relationships between features and target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Name of target column
    """
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
    
    plt.tight_layout()
    plt.show()
    
    # Print target value distribution
    print(f"\nTarget Variable ({target_col}) Distribution:")
    print(df[target_col].value_counts(normalize=True))

def analyze_outliers(df: pd.DataFrame) -> None:
    """
    Analyze and visualize outliers in numerical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create box plots for outlier detection
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[numerical_cols])
    plt.title('Outlier Detection Using Box Plots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calculate and print outlier statistics
    print("\nOutlier Analysis:")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        if len(outliers) > 0:
            print(f"\n{col}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Percentage of outliers: {(len(outliers) / len(df) * 100):.2f}%")

def main():
    """Main execution function for EDA."""
    try:
        # Load the dataset
        dataset_path = download_dataset()
        dataframes = load_csv_files(dataset_path)
        
        # Assuming the main dataset is the first one
        df = list(dataframes.values())[0]
        
        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print("\nFeature Information:")
        print(df.info())
        
        # Perform various analyses
        analyze_missing_values(df)
        analyze_feature_distributions(df)
        analyze_correlations(df)
        
        # Assuming 'diagnosis' is the target column - adjust if different
        if 'diagnosis' in df.columns:
            analyze_target_relationship(df, 'diagnosis')
        
        analyze_outliers(df)
        
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()