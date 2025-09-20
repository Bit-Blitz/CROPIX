import pandas as pd
import ast 

def clean_and_merge_weather_data(forecast_file, location_file, astro_file, output_file):
    try:

        df_forecast = pd.read_csv(forecast_file)
        df_location = pd.read_csv(location_file)
        df_astro = pd.read_csv(astro_file)
        
        print("Successfully loaded all three CSV files.")
        print("Processing forecast_data.csv...")
    
        df_forecast['time'] = pd.to_datetime(df_forecast['time'])
        df_forecast['date'] = df_forecast['time'].dt.date.astype(str)

        def extract_condition_text(condition_str):
            try:
                condition_dict = ast.literal_eval(condition_str)
                return condition_dict.get('text', 'Unknown')
            except (ValueError, SyntaxError):
                return 'Unknown'
        
        df_forecast['condition_text'] = df_forecast['condition'].apply(extract_condition_text)
        forecast_cols_to_keep = [
            'time', 'date', 'state', 'city', 'temp_c', 'is_day', 'wind_kph', 
            'wind_dir', 'pressure_mb', 'precip_mm', 'humidity', 'cloud', 
            'feelslike_c', 'vis_km', 'dewpoint_c', 'condition_text'
        ]
        df_forecast = df_forecast[forecast_cols_to_keep]
        print("Processing location_data.csv...")

        df_location['localtime'] = pd.to_datetime(df_location['localtime'])
        df_location['date'] = df_location['localtime'].dt.date.astype(str)
        

        location_cols_to_keep = ['date', 'region', 'name', 'lat', 'lon']
        df_location = df_location[location_cols_to_keep]
        df_location.rename(columns={'region': 'state', 'name': 'city'}, inplace=True)
        df_location.drop_duplicates(subset=['date', 'state', 'city'], inplace=True)
        print("Processing astro_data.csv...")
        df_astro.drop_duplicates(subset=['state', 'city'], inplace=True)

        print("Merging DataFrames...")
    
        df_merged = pd.merge(df_forecast, df_location, on=['date', 'state', 'city'], how='left')

        df_final = pd.merge(df_merged, df_astro, on=['state', 'city'], how='left')
        print("Performing final cleanup...")
        df_final.sort_values(by=['state', 'city', 'time'], inplace=True)
        df_final.fillna(method='ffill', inplace=True)
        df_final.dropna(inplace=True)
        df_final.drop(columns=['date'], inplace=True)

        df_final.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully created the consolidated file: {output_file}")
        print(f"Final dataset has {df_final.shape[0]} rows and {df_final.shape[1]} columns.")
        print("\nFirst 5 rows of the final dataset:")
        print(df_final.head())
        print("\nColumns in the final dataset:")
        print(df_final.columns.tolist())

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure all input CSV files are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    forecast_csv = 'Datasets/LSTM_data/forecast_data.csv'
    location_csv = 'Datasets/LSTM_data/location_data.csv'
    astro_csv = 'Datasets/LSTM_data/astro_data.csv'
    output_csv = 'Datasets/LSTM_data/weather_data.csv'
    clean_and_merge_weather_data(forecast_csv, location_csv, astro_csv, output_csv)

