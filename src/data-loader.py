import os
import kagglehub
import pandas as pd


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


def load_csv_file(file_path: str) -> pd.DataFrame:
    """
    Load a single CSV file into a DataFrame.
    
    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Drop columns with names starting with 'Unnamed'
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        print(f"Loaded {file_path} successfully")
        return df
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame as a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved DataFrame to {output_path}")


def display_dataframe(df: pd.DataFrame, n_rows: int = 5) -> None:
    """
    Display the first n rows of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to display
        n_rows (int): Number of rows to display
    """
    print(f"\nDataFrame Shape: {df.shape}")
    print(df.head(n_rows))
    print("-" * 80)


def main():
    """Main execution function."""
    try:
        # Download the dataset
        dataset_path = download_dataset()

        # Assume there is only one CSV file in the directory
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset directory")

        # Load the single CSV file
        csv_file_path = os.path.join(dataset_path, csv_files[0])
        df = load_csv_file(csv_file_path)

        # Display the loaded data
        display_dataframe(df)

        # Save the DataFrame to a separate directory
        output_path = os.path.join("data", "raw.csv")
        save_dataframe(df, output_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
