import logging
from src.data_preprocessing.eda_script import perform_eda
from src.data_preprocessing.data_loader import download_dataset, load_csv_files

def main():
    """Main execution function for EDA."""
    try:
        logging.info("Starting EDA process.")
        dataset_path = download_dataset()
        dataframes = load_csv_files(dataset_path)
        df = list(dataframes.values())[0]
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