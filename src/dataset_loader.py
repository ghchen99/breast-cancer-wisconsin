import os
from typing import Tuple
import kagglehub
import pandas as pd

# Constants
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "raw.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

class DatasetLoader:
    def __init__(self):
        self.dataset_path = None
        self.output_path = OUTPUT_PATH

    def download_dataset(self) -> str:
        """
        Download the breast cancer dataset from Kaggle.
        
        Returns:
            str: Path to the downloaded dataset directory
        """
        try:
            self.dataset_path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
            print(f"Dataset downloaded successfully to: {self.dataset_path}")
            return self.dataset_path
        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")

    def find_csv_file(self) -> str:
        """
        Find the first CSV file in the dataset directory.
        
        Returns:
            str: Path to the CSV file
        
        Raises:
            FileNotFoundError: If no CSV files are found
        """
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset directory")
        
        return os.path.join(self.dataset_path, csv_files[0])

    @staticmethod
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
            df = pd.read_csv(file_path)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            print(f"Loaded {file_path} successfully")
            return df
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")

    def save_dataframe(self, df: pd.DataFrame) -> None:
        """
        Save a DataFrame as a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved DataFrame to {self.output_path}")

    @staticmethod
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

    def process(self) -> Tuple[pd.DataFrame, str]:
        """
        Execute the complete data processing pipeline.
        
        Returns:
            Tuple[pd.DataFrame, str]: Processed DataFrame and output path
        """
        try:
            self.download_dataset()
            csv_file_path = self.find_csv_file()
            df = self.load_csv_file(csv_file_path)
            self.display_dataframe(df)
            self.save_dataframe(df)
            return df, self.output_path
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise


def main():
    """Main execution function."""
    processor = DatasetLoader()
    df, output_path = processor.process()
    print(f"\nProcessing complete. Data saved to: {output_path}")

if __name__ == "__main__":
    main()