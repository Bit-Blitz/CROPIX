import pandas as pd
import os


WEATHER_DATA_FILE = "Weather data.xlsx - Sheet1.csv"
LOCATION_DATA_FILE = "Location information.xlsx - Sheet1.csv"


OUTPUT_FILE = "Weather_data.csv"


def preprocess_and_save_data(weather_file, location_file, output_file):
    print("--- Starting Data Preprocessing ---")
    print(f"Attempting to load data from {weather_file} and {location_file}...")

    if not os.path.exists(weather_file) or not os.path.exists(location_file):
        print("\nError: One or both data files not found.")
        print("Please make sure both CSV files are in the same directory.")
        return
    
    df_weather = pd.read_csv(weather_file)
    df_location = pd.read_csv(location_file)
    print("Successfully loaded raw data files.")

    df = pd.merge(df_weather, df_location, on='last_updated_epoch')
    print("Successfully merged weather and location data.")

    df = df.rename(columns={
        'last_updated_epoch': 'timestamp_epoch', # Keep epoch for now
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
    df_final = df_final.fillna(method='ffill')
    df_final = df_final.sort_values(by=['location_id', 'timestamp'])
    df_final.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved the clean, merged data to: {output_file}")
    print("--- Data Preprocessing Complete ---")


if __name__ == "__main__":
    preprocess_and_save_data(WEATHER_DATA_FILE, LOCATION_DATA_FILE, OUTPUT_FILE)
