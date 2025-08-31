import pandas as pd
import pmdarima as pm
from joblib import dump
import re
import os

def create_clean_basename(crop_name):
    base_name = crop_name.replace('(INR/Quintal)', '')
    temp_name = re.sub(r'[^a-zA-Z0-9]+', '_', base_name)
    clean_base = temp_name.strip('_')
    return f"{clean_base}"

def train_and_save_model(file_path, crop_name):
    try:
        
        print(f"--- Processing: {crop_name} ---")
        data = pd.read_csv(file_path, index_col='Month', parse_dates=True)
        data.index.freq = 'MS'

      
        print("Training SARIMA model with auto_arima...")
        model = pm.auto_arima(data[crop_name],
                              start_p=1, start_q=1,
                              test='adf', max_p=3, max_q=3, m=12,
                              d=None, seasonal=True, start_P=0,
                              D=1, trace=False, error_action='ignore',
                              suppress_warnings=True, stepwise=True)

        print("Model training complete.")
        print(model.summary())

       
        output_folder = "Trained_Models/SARIMA_MODELS"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        clean_name = create_clean_basename(crop_name)
        model_save_path = os.path.join(output_folder, f"sarima_model_{clean_name}.joblib")

        dump(model, model_save_path)

        print(f"Model successfully saved to: {model_save_path}\n")

    except Exception as e:
        print(f"An unexpected error occurred while processing {crop_name}: {e}\n")

if __name__ == '__main__':
    DATA_FILE = 'Datasets/synthetic_crop_prices_2years.csv'


    print(f"Loading data from '{DATA_FILE}' to identify crops...")
    all_data = pd.read_csv(DATA_FILE)
    crop_columns = [col for col in all_data.columns if '(INR/Quintal)' in col]
    print(f"Found {len(crop_columns)} crops to train models for.\n")

    for crop in crop_columns:
        train_and_save_model(DATA_FILE, crop)

    print("--- All SARIMA models have been trained and saved successfully! ---")

