import logging
import warnings
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data_preprocessing.data_loader import download_dataset, load_csv_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings('ignore')

# Create the directory for saving cleaned data if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

def save_cleaned_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save the cleaned dataset to the 'data' directory.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        file_name (str): Name of the output file.
    """
    output_path = f'data/{file_name}.csv'
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned dataset saved as '{output_path}'")


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values by removing columns or imputing values."""
    logging.info("Cleaning missing values.")

    # Drop columns with more than 30% missing values
    threshold = 0.3 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining missing values with column median (excluding 'id' and 'diagnosis')
    for col in df.columns:
        if col not in ['id', 'diagnosis'] and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df


def clean_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on 'id'."""
    logging.info("Removing duplicate rows.")
    df = df.drop_duplicates(subset='id')
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    logging.info("Encoding categorical variables.")
    if 'diagnosis' in df.columns:
        le = LabelEncoder()
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numerical features (excluding 'id' and 'diagnosis')."""
    logging.info("Scaling numerical features.")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'diagnosis']]
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Perform all cleaning steps."""
    df = clean_missing_values(df)
    df = clean_duplicates(df)
    df = encode_categorical(df)
    df = scale_features(df)
    return df


def main():
    """Main execution function for data cleaning."""
    try:
        logging.info("Starting data cleaning process.")
        dataset_path = download_dataset()
        dataframes = load_csv_files(dataset_path)
        df = list(dataframes.values())[0]

        logging.info("Dataset loaded successfully.")
        print("Dataset Overview:")
        print(f"Shape: {df.shape}")
        print("\nFeature Information:")
        print(df.info())

        # Perform data cleaning
        cleaned_df = clean_dataset(df)

        # Save cleaned dataset
        save_cleaned_data(cleaned_df, 'cleaned_dataset')

        logging.info("Data cleaning process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during data cleaning: {str(e)}")


if __name__ == "__main__":
    main()
