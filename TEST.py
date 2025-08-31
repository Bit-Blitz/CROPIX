import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def predict_yield():
    print("\n--- Crop Yield Prediction (XGBoost) ---")
    try:
        model = joblib.load('Trained_models/CROP_YIELD_MODEL.joblib')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("\nERROR: 'xgboost_yield_model.joblib' not found. Please train the model first.")
        return

    print("Please enter the following details:")
    crop = input("Enter Crop Name (e.g., Wheat, Soyabean): ")
    crop_year = int(input("Enter Year of Planting (e.g., 2025): "))
    season = input("Enter Season (e.g., Kharif, Rabi): ")
    area = float(input("Enter Area (in Hectares): "))
    rainfall = float(input("Enter Annual Rainfall (in mm): "))
    fertilizer = float(input("Enter Planned Fertilizer Usage (in kg): "))
    pesticide = float(input("Enter Planned Pesticide Usage (in kg): "))

    input_data = pd.DataFrame({
        'Crop': [crop],
        'Crop_Year': [crop_year],
        'Season': [season],
        'Area': [area],
        'Annual_Rainfall': [rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide]
    })
    
    prediction = model.predict(input_data)
    print("\n--- PREDICTION RESULT ---")
    print(f"Predicted Crop Yield: {prediction[0]:.2f} tonnes per hectare")
    print("-------------------------")


def recommend_fertilizer():
    print("\n--- Fertilizer Recommendation (Random Forest) ---")
    try:
        model = joblib.load('Trained_models/fertilizer_recommendation_model.joblib')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("\nERROR: 'fertilizer_recommendation_model.joblib' not found. Please train the model first.")
        return

    print("Please enter the following details:")
    crop = input("Enter Crop Name (e.g., Wheat, Soyabean): ")
    current_n = float(input("Enter current Nitrogen (N) content in soil (kg/ha): "))
    current_p = float(input("Enter current Phosphorus (P) content in soil (kg/ha): "))
    current_k = float(input("Enter current Potassium (K) content in soil (kg/ha): "))
    
    input_data = pd.DataFrame({
        'Crop': [crop],
        'Current_N': [current_n],
        'Current_P': [current_p],
        'Current_K': [current_k]
    })

    prediction = model.predict(input_data)
    print("\n--- RECOMMENDATION RESULT ---")
    print(f"Required Nitrogen (N): {prediction[0, 0]:.2f} kg/ha")
    print(f"Required Phosphorus (P): {prediction[0, 1]:.2f} kg/ha")
    print(f"Required Potassium (K): {prediction[0, 2]:.2f} kg/ha")
    print("-----------------------------")


def recommend_crop():
    
    print("\n--- Soil & Crop Recommendation (KNN) ---")
    try:
        model = joblib.load('Trained_models/Soil_crop_recom.joblib')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("\nERROR: 'knn_crop_recommendation_model.joblib' not found. Please train the model first.")
        return

   
    print("Please enter the following details for your field:")
    n = float(input("Enter Nitrogen (N) content in soil: "))
    p = float(input("Enter Phosphorus (P) content in soil: "))
    k = float(input("Enter Potassium (K) content in soil: "))
    temp = float(input("Enter average temperature (°C): "))
    humidity = float(input("Enter average humidity (%): "))
    ph = float(input("Enter soil pH value: "))
    rainfall = float(input("Enter average rainfall (mm): "))
    
    input_data = pd.DataFrame({
        'N': [n], 'P': [p], 'K': [k],
        'temperature': [temp], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall]
    })

    prediction = model.predict(input_data)
    print("\n--- RECOMMENDATION RESULT ---")
    print(f"The most suitable crop for your field is: {prediction[0]}")
    print("-----------------------------")


def forecast_market_price():
    print("\n--- Market Price Forecast (SARIMA) ---")
    
    crop_name = input("Enter Crop Name to Forecast (e.g., Soyabean, Wheat): ").lower()
    forecast_days = int(input("Enter number of days to forecast: "))
    
    model_path = f'Trained_models/SARIMA_MODELS/sarima_model_{crop_name}.joblib'
    
    try:
        model = joblib.load(model_path)
        print(f"Model for '{crop_name}' loaded successfully.")
    except FileNotFoundError:
        print(f"\nERROR: '{model_path}' not found.")
        print("Please ensure you have trained and saved a specific SARIMA model for this crop.")
        return
    
    forecast = model.predict(n_periods=forecast_days)
    
    print("\n--- FORECAST RESULT ---")
    print(f"Forecasted prices for the next {forecast_days} periods (days/weeks):")
    for i, price in enumerate(forecast):
        print(f"Period {i+1}: {price:.2f}")
    print("-----------------------")


def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\n\n===== CROPIX PREDICTION SYSTEM =====")
        print("Select a model to use:")
        print("  1. Crop Yield Prediction")
        print("  2. Fertilizer Recommendation")
        print("  3. Soil & Crop Recommendation")
        print("  4. Market Price Forecast")
        print("  5. Weather Forecast (Coming Soon)")
        print("  6. Pesticide Suggestion (Coming Soon)")
        print("  7. Exit")
        
        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            predict_yield()
        elif choice == '2':
            recommend_fertilizer()
        elif choice == '3':
            recommend_crop()
        elif choice == '4':
            forecast_market_price()
        elif choice == '5':
            print("\nWeather Forecasting model is under development.")
        elif choice == '6':
            print("\nPesticide Suggestion model is under development.")
        elif choice == '7':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 7.")


if __name__ == "__main__":
    main_menu()