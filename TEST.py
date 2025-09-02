import joblib
import pandas as pd
import numpy as np
import warnings
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.preprocessing import MinMaxScaler
import cv2

warnings.filterwarnings('ignore')

MODELS_DIR = 'Trained_models/'
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'CNN_MODEL', 'disease_detection_model.h5')
CNN_CLASSES_PATH = os.path.join(MODELS_DIR, 'CNN_MODEL', 'disease_classes.npy')
CNN_IMG_SIZE = (256, 256)
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'LSTM_model.keras')
LSTM_WEATHER_DATA_PATH = 'Datasets/Weather_data.csv'

cnn_model = None
cnn_class_names = None
lstm_model = None
lstm_scaler = None
lstm_sequence_length = 10
lstm_num_features = 9

def predict_yield():
    print("\n--- Crop Yield Prediction (XGBoost) ---")
    try:
        model = joblib.load('Trained_models/xgboost_yield_model.joblib')
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
        model = joblib.load('Trained_models/knn_crop_recommendation_model.joblib')
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
    model_path = f'Trained_models/sarima_model_{crop_name}.joblib'
    
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

def load_cnn_model():
    global cnn_model, cnn_class_names
    if cnn_model is None:
        try:
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            cnn_class_names = np.load(CNN_CLASSES_PATH, allow_pickle=True)
            print(f"[INFO] CNN model loaded from {CNN_MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Could not load CNN model or classes: {e}")
            return False
    return True

def load_lstm_model():
    global lstm_model, lstm_scaler
    if lstm_model is None:
        try:
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            print(f"[INFO] LSTM model loaded from {LSTM_MODEL_PATH}")
            
            df_full = pd.read_csv(LSTM_WEATHER_DATA_PATH)
            df_full.index = pd.to_datetime(df_full.datetime_utc)
            required_cols = [' _dewptm', ' _fog', ' _hail', ' _hum', ' _rain', ' _snow', ' _tempm', ' _thunder', ' _tornado']
            df_full = df_full[required_cols]
            df_full.fillna(method='ffill', inplace=True)
            df_full = df_full.resample('D').mean()
            df_full.fillna(method='ffill', inplace=True)
            
            lstm_scaler = MinMaxScaler()
            lstm_scaler.fit(df_full)
            print("[INFO] MinMaxScaler for LSTM model re-fitted to original weather data.")
        except Exception as e:
            print(f"[ERROR] Could not load LSTM model or fit scaler: {e}")
            return False
    return True

def preprocess_image_for_cnn(img_array):
    img_array = cv2.resize(img_array, CNN_IMG_SIZE)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_cnn_disease(img_array):
    if cnn_model is None or cnn_class_names is None:
        return "Prediction Error: CNN model not loaded."
    
    predictions = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_disease = cnn_class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return f"{predicted_disease} (Confidence: {confidence:.2f}%)"

def capture_image_from_camera():
    print("[INFO] Opening webcam... Press 'q' to capture the image.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None
    
    frame = None
    ret = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        cv2.imshow('Press \'q\' to Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Image captured.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return frame if ret else None

def upload_image_for_cnn():
    file_path = input("Enter the path to the image file: ").strip()
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at: {file_path}")
        return None
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Image loaded from {file_path}")
        return img_rgb
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the image: {e}")
        return None

def get_lstm_input_from_user():
    print("\n[INFO] Enter the last 10 days of weather data for prediction.")
    print("       Features: dewptm, fog, hail, hum, rain, snow, tempm, thunder, tornado")
    
    user_input_sequence = []
    for i in range(lstm_sequence_length):
        while True:
            try:
                day_data_str = input(f"Enter data for Day {i+1} (9 comma-separated values): ")
                day_data = [float(x.strip()) for x in day_data_str.split(',')]
                if len(day_data) == lstm_num_features:
                    user_input_sequence.append(day_data)
                    break
                else:
                    print(f"[ERROR] Expected {lstm_num_features} values, got {len(day_data)}. Please try again.")
            except ValueError:
                print("[ERROR] Invalid input. Please enter numbers separated by commas.")
    
    return np.array(user_input_sequence)

def predict_lstm_temperature(raw_input_sequence):
    if lstm_model is None or lstm_scaler is None:
        return "Prediction Error: LSTM model not loaded."

    scaled_input_sequence = lstm_scaler.transform(raw_input_sequence)
    scaled_input_sequence = np.expand_dims(scaled_input_sequence, axis=0)
    
    scaled_prediction = lstm_model.predict(scaled_input_sequence)
    
    dummy_inverse_array = np.zeros((1, lstm_num_features))
    dummy_inverse_array[0, 6] = scaled_prediction[0, 0]
    
    actual_prediction = lstm_scaler.inverse_transform(dummy_inverse_array)[0, 6]
    
    return f"Predicted Temperature: {actual_prediction:.2f}°C"

def main_menu():
    while True:
        print("\n\n===== CROPIX PREDICTION SYSTEM =====")
        print("Select a model to use:")
        print("  1. Crop Yield Prediction")
        print("  2. Fertilizer Recommendation")
        print("  3. Soil & Crop Recommendation")
        print("  4. Market Price Forecast")
        print("  5. Disease Detection (CNN)")
        print("  6. Weather Forecast (LSTM)")
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
            if load_cnn_model():
                cnn_sub_menu()
        elif choice == '6':
            if load_lstm_model():
                lstm_inference_flow()
        elif choice == '7':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 7.")

def cnn_sub_menu():
    while True:
        print("\n--- CNN Disease Detection ---")
        print("1. Capture image from camera")
        print("2. Upload image file")
        print("3. Back to main menu")
        cnn_choice = input("Enter your choice (1-3): ").strip()

        img_to_predict = None
        if cnn_choice == '1':
            img_to_predict = capture_image_from_camera()
        elif cnn_choice == '2':
            img_to_predict = upload_image_for_cnn()
        elif cnn_choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue
        
        if img_to_predict is not None:
            processed_img = preprocess_image_for_cnn(img_to_predict)
            result = predict_cnn_disease(processed_img)
            print(f"\n[CNN Result] {result}")

def lstm_inference_flow():
    print("\n--- LSTM Weather Prediction ---")
    raw_input_data = get_lstm_input_from_user()
    if raw_input_data is not None:
        result = predict_lstm_temperature(raw_input_data)
        print(f"\n[LSTM Result] {result}")

if __name__ == "__main__":
    main_menu()