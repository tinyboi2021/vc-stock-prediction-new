import pandas as pd
import logging
import yaml
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_to_raw(input_path: str, output_path: str):
    """
    Validates the final merged dataset and officially ingests it into the raw folder.
    """
    logging.info(f"Initiating final data ingestion from {input_path}...")
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing {input_path}. Did the final merge step fail?")

    # 1. Load the data to perform a final validation check
    df = pd.read_csv(input_path)
    
    # 2. Log vital statistics for your MLOps tracking
    date_col = df.columns[0]
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    logging.info("--- INGESTION VALIDATION REPORT ---")
    logging.info(f"Total Rows: {len(df)}")
    logging.info(f"Total Columns: {len(df.columns)}")
    logging.info(f"Date Range: {min_date} to {max_date}")
    logging.info("-----------------------------------")
    
    # 3. Officially move it to the raw folder
    logging.info(f"Saving validated dataset to {output_path}...")
    
    # We use shutil.copy2 to explicitly copy the file to preserve metadata, 
    # or you could use df.to_csv if you wanted to reset the index. 
    # df.to_csv is safer to guarantee pure CSV formatting.
    df.to_csv(output_path, index=False)
    
    logging.info("✅ Data Ingestion Complete. Ready for preprocessing.")

if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["data_ingestion"]
            
        ingest_to_raw(params["input_path"], params["output_path"])
        
    except FileNotFoundError:
        logging.error("params.yaml not found. Please ensure it is in the project root.")
        exit(1)
    except KeyError as e:
        logging.error(f"Missing key in params.yaml: {e}")
        exit(1)