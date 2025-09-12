import pandas as pd
import os

WEATHER_DATA_FILE = "Datasets/NBeats Model/WEATHER.xlsx"
LOCATION_DATA_FILE = "Datasets/NBeats Model/Location information.xlsx"
OUTPUT_FILE = "Datasets/NBeats Model/Weather_data.csv"

def preprocess_and_save_data(weather_file, location_file, output_file):
    print("--- Starting Data Preprocessing ---")
    print(f"Attempting to load data from {weather_file} and {location_file}...")

    if not os.path.exists(weather_file) or not os.path.exists(location_file):
        print("\nError: One or both data files not found.")
        print("Please make sure the paths and filenames are correct.")
        return

    df_weather = pd.read_excel(weather_file)
    df_location = pd.read_excel(location_file)
    print("Successfully loaded raw data files.")

    df = pd.merge(df_weather, df_location, on='last_updated_epoch')
    print("Successfully merged weather and location data.")

    df = df.rename(columns={
        'last_updated_epoch': 'timestamp_epoch',
        'temperature_celsius': 'temp',
        'pressure_mb': 'pressure',
        'humidity': 'humidity',
        'wind_kph': 'wind_speed',
        'precip_mm': 'precipitation',
        'cloud': 'cloud_cover',
        'uv_index': 'uv',
        'location_name': 'location_id'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp_epoch'], unit='s')

    final_cols = [
        'timestamp',
        'location_id',
        'temp',
        'pressure',
        'humidity',
        'wind_speed',
        'precipitation',
        'cloud_cover',
        'uv'
    ]
    df_final = df[final_cols]
    print("Selected and renamed relevant columns.")
    
    df_final = df_final.fillna(method='ffill')

    df_final = df_final.sort_values(by=['location_id', 'timestamp'])
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    df_final.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved the clean, merged data to: {output_file}")
    print("--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    preprocess_and_save_data(WEATHER_DATA_FILE, LOCATION_DATA_FILE, OUTPUT_FILE)

