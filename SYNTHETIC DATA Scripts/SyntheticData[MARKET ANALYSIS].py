import pandas as pd
import sys
import random
import math
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CROP_CONFIG = {
    'Soybean': {'base_price': 4600, 'peak_month': 7, 'amplitude': 0.20}, # Harvest: Sep-Oct
    'Paddy (Rice)': {'base_price': 2300, 'peak_month': 8, 'amplitude': 0.15}, # Harvest: Oct-Nov
    'Wheat': {'base_price': 2450, 'peak_month': 1, 'amplitude': 0.18}, # Harvest: Mar-Apr
    'Gram (Chana)': {'base_price': 5500, 'peak_month': 9, 'amplitude': 0.22}, # Harvest: Feb-Mar
    'Maize': {'base_price': 2150, 'peak_month': 7, 'amplitude': 0.15}, # Harvest: Sep-Oct
    'Mustard': {'base_price': 5800, 'peak_month': 1, 'amplitude': 0.25}  # Harvest: Feb-Mar
}


START_DATE = datetime(2025, 9, 1) 
MONTHS_TO_GENERATE = 24 
ANNUAL_INFLATION_RATE = 0.05 
RANDOM_VOLATILITY = 0.05    

def generate_price(base_price, peak_month, amplitude, current_month_index, month_number):
    monthly_inflation = (1 + ANNUAL_INFLATION_RATE) ** (1/12) - 1
    inflated_base_price = base_price * ((1 + monthly_inflation) ** current_month_index)

    seasonal_factor = 1 + amplitude * math.sin((math.pi / 6) * (month_number - peak_month + 3))

    random_factor = random.uniform(1 - RANDOM_VOLATILITY, 1 + RANDOM_VOLATILITY)
    
    final_price = inflated_base_price * seasonal_factor * random_factor

    return round(final_price, 2)

def generate_dataset():
    dataset = []
    current_date = START_DATE

    for i in range(MONTHS_TO_GENERATE):
        month_data = {
            'Month': current_date.strftime('%Y-%m'),
            'Month Name': current_date.strftime('%B')
        }
        for crop, config in CROP_CONFIG.items():
            price = generate_price(
                config['base_price'],
                config['peak_month'],
                config['amplitude'],
                i,
                current_date.month
            )
            month_data[f'{crop} (INR/Quintal)'] = price

        dataset.append(month_data)
        current_date += relativedelta(months=1)

    return dataset

def save_to_csv(dataset, filename):
    
    if not dataset:
        print("No data to save.", file=sys.stderr)
        return
    
    df = pd.DataFrame(dataset)
    first_cols = ['Month', 'Month Name']
    other_cols = sorted([col for col in df.columns if col not in first_cols])
    df = df[first_cols + other_cols]
    df.to_csv(filename, index=False)
    print(f"\nData successfully saved to {filename}", file=sys.stderr)


if __name__ == '__main__':
    output_filename = "Datasets/synthetic_crop_prices_2years.csv"
    
    print("--- Synthetic Crop Price Data (Madhya Pradesh) ---", file=sys.stderr)
    print(f"--- Generating data for {MONTHS_TO_GENERATE} months starting from {START_DATE.strftime('%Y-%m')} ---", file=sys.stderr)
    synthetic_data = generate_dataset()
    save_to_csv(synthetic_data, output_filename)

