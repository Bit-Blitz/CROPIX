import pandas as pd
import ast # To safely evaluate the string representation of the dictionary

def clean_and_merge_weather_data(forecast_file, location_file, astro_file, output_file):
    try:
        # 1. Load all datasets
        df_forecast = pd.read_csv(forecast_file)
        df_location = pd.read_csv(location_file)
        df_astro = pd.read_csv(astro_file)
        
        print("Successfully loaded all three CSV files.")

        # --- 2. Process Forecast Data ---
        print("Processing forecast_data.csv...")
        # Convert time column to datetime objects
        df_forecast['time'] = pd.to_datetime(df_forecast['time'])
        
        # Create a 'date' column for merging with daily data
        df_forecast['date'] = df_forecast['time'].dt.date.astype(str)

        # Safely parse the 'condition' column (it's a stringified dictionary)
        # We will extract just the 'text' part.
        def extract_condition_text(condition_str):
            try:
                # ast.literal_eval is safer than eval()
                condition_dict = ast.literal_eval(condition_str)
                return condition_dict.get('text', 'Unknown')
            except (ValueError, SyntaxError):
                return 'Unknown'
        
        df_forecast['condition_text'] = df_forecast['condition'].apply(extract_condition_text)

        # Select relevant columns and drop redundant ones
        forecast_cols_to_keep = [
            'time', 'date', 'state', 'city', 'temp_c', 'is_day', 'wind_kph', 
            'wind_dir', 'pressure_mb', 'precip_mm', 'humidity', 'cloud', 
            'feelslike_c', 'vis_km', 'dewpoint_c', 'condition_text'
        ]
        df_forecast = df_forecast[forecast_cols_to_keep]

        # --- 3. Process Location Data ---
        print("Processing location_data.csv...")
        # Convert localtime to datetime and extract date
        df_location['localtime'] = pd.to_datetime(df_location['localtime'])
        df_location['date'] = df_location['localtime'].dt.date.astype(str)
        
        # Keep relevant columns and remove duplicates for the same city on the same day
        location_cols_to_keep = ['date', 'region', 'name', 'lat', 'lon']
        df_location = df_location[location_cols_to_keep]
        df_location.rename(columns={'region': 'state', 'name': 'city'}, inplace=True)
        df_location.drop_duplicates(subset=['date', 'state', 'city'], inplace=True)

        # --- 4. Process Astro Data ---
        # As astro_data lacks a date, we prepare it for a merge based on city/state.
        # This assumes the astro data is constant for a city.
        print("Processing astro_data.csv...")
        df_astro.drop_duplicates(subset=['state', 'city'], inplace=True)

        # --- 5. Merge DataFrames ---
        print("Merging DataFrames...")
        # Merge forecast with location data
        df_merged = pd.merge(df_forecast, df_location, on=['date', 'state', 'city'], how='left')

        # Merge the result with astro data
        df_final = pd.merge(df_merged, df_astro, on=['state', 'city'], how='left')

        # --- 6. Final Cleanup ---
        print("Performing final cleanup...")
        # Sort the data to ensure correct time-series order
        df_final.sort_values(by=['state', 'city', 'time'], inplace=True)

        # Check for and handle any NaNs that may have resulted from merging
        # Using forward fill is a reasonable strategy for time-series data
        df_final.fillna(method='ffill', inplace=True)
        df_final.dropna(inplace=True) # Drop any remaining NaNs at the beginning
        
        # Drop the intermediate 'date' column
        df_final.drop(columns=['date'], inplace=True)

        # Save to the final CSV
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
    forecast_csv = 'forecast_data.csv'
    location_csv = 'location_data.csv'
    astro_csv = 'astro_data.csv'
    output_csv = 'consolidated_weather_data.csv'
    clean_and_merge_weather_data(forecast_csv, location_csv, astro_csv, output_csv)

