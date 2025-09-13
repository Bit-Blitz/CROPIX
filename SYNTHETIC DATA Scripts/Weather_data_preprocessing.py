import pandas as pd
import os

WEATHER_DATA_FILE = "Datasets/NBeats Model/WEATHER.xlsx"
LOCATION_DATA_FILE = "Datasets/NBeats Model/Location information.xlsx"
OUTPUT_FILE = "Datasets/NBeats Model/Weather_data_daily.csv"

def preprocess_and_save_daily_data(weather_file, location_file, output_file):
    print("--- Starting Daily Data Preprocessing ---")
    print(f"Attempting to load data from {weather_file} and {location_file}...")

    if not os.path.exists(weather_file) or not os.path.exists(location_file):
        print("\nError: One or both data files not found.")
        return

    df_weather = pd.read_excel(weather_file)
    df_location = pd.read_excel(location_file)
    df = pd.merge(df_weather, df_location, on='last_updated_epoch')
    print("Successfully merged weather and location data.")

    df = df.rename(columns={
        'temperature_celsius': 'temp',
        'pressure_mb': 'pressure',
        'humidity': 'humidity',
        'wind_kph': 'wind_speed',
        'precip_mm': 'precipitation',
        'cloud': 'cloud_cover',
        'uv_index': 'uv',
        'location_name': 'location_id'
    })

    df['timestamp'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    print("Aggregating hourly data into daily summaries...")
    
    # Define how to aggregate each column
    aggregation_rules = {
        'temp': 'mean',
        'pressure': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'precipitation': 'sum', # Total daily rainfall
        'cloud_cover': 'mean',
        'uv': 'max' # Peak daily UV
    }

    # Group by location, then resample to daily frequency
    df_daily = df.groupby('location_id').resample('D').agg(aggregation_rules).reset_index()
    
    df_daily = df_daily.rename(columns={'timestamp': 'date'})
    df_daily = df_daily.dropna() # Remove days with no data
    df_daily = df_daily.sort_values(by=['location_id', 'date'])
    
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_daily.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved the clean, daily aggregated data to: {output_file}")
    print("--- Data Preprocessing Complete ---")

if __name__ == "__main__":
    preprocess_and_save_daily_data(WEATHER_DATA_FILE, LOCATION_DATA_FILE, OUTPUT_FILE)

