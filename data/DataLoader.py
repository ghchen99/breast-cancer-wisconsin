import os
import kagglehub
import pandas as pd
from typing import Dict


def download_dataset() -> str:
    """
    Download the breast cancer dataset from Kaggle.
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    try:
        path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        print(f"Dataset downloaded successfully to: {path}")
        return path
    except Exception as e:
        raise Exception(f"Failed to download dataset: {str(e)}")


def load_csv_files(directory_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the specified directory into DataFrames.
    
    Args:
        directory_path (str): Path to directory containing CSV files
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping file names to DataFrames
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all CSV files
    files = os.listdir(directory_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory_path}")
    
    # Load each CSV file into a DataFrame
    dataframes = {}
    for csv_file in csv_files:
        try:
            csv_path = os.path.join(directory_path, csv_file)
            key = os.path.splitext(csv_file)[0]
            dataframes[key] = pd.read_csv(csv_path)
            print(f"Loaded {csv_file} successfully")
        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            continue
    
    return dataframes


def display_dataframes(dataframes: Dict[str, pd.DataFrame], n_rows: int = 5) -> None:
    """
    Display the first n rows of each DataFrame.
    
    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of DataFrames to display
        n_rows (int): Number of rows to display for each DataFrame
    """
    for key, df in dataframes.items():
        print(f"\nDataFrame: {key}")
        print(f"Shape: {df.shape}")
        print(df.head(n_rows))
        print("-" * 80)


def main():
    """Main execution function."""
    try:
        # Download and load the dataset
        dataset_path = download_dataset()
        dataframes = load_csv_files(dataset_path)
        
        # Display the loaded data
        display_dataframes(dataframes)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()