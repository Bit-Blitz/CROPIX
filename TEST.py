import joblib
import pandas as pd
import numpy as np
import warnings
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.preprocessing import MinMaxScaler
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


MODELS_DIR = 'Trained_models/'
DATA_DIR = 'Datasets/'


CROP_YIELD_MODEL_PATH = os.path.join(MODELS_DIR, 'CROP_YIELD_MODEL.joblib')
FERTILIZER_MODEL_PATH = os.path.join(MODELS_DIR, 'fertilizer_recommendation_model.joblib')
SOIL_CROP_MODEL_PATH = os.path.join(MODELS_DIR, 'Soil_crop_recom.joblib')
SARIMA_MODELS_DIR = os.path.join(MODELS_DIR, 'SARIMA_MODELS')

CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'CNN_MODEL', 'disease_detection_model.h5')
CNN_CLASSES_PATH = os.path.join(MODELS_DIR, 'CNN_MODEL', 'disease_classes.npy')
CNN_IMG_SIZE = (256, 256)

# LSTM Model Paths and Config
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "LSTM_model" ,'LSTM_model.keras')
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "LSTM_model",'weather_scaler.pkl')
LSTM_WEATHER_DATA_PATH = os.path.join(DATA_DIR, 'Weather_data.csv')
SEQUENCE_LENGTH = 10
REQUIRED_COLS = [' _dewptm', ' _fog', ' _hail', ' _hum', ' _rain', ' _snow', ' _tempm', ' _thunder', ' _tornado']
TARGET_COL = ' _tempm'
TARGET_COL_INDEX = REQUIRED_COLS.index(TARGET_COL)
NUM_FEATURES = len(REQUIRED_COLS)

cnn_model = None
cnn_class_names = None
lstm_model = None
lstm_scaler = None

def load_cnn_model():
    global cnn_model, cnn_class_names
    if cnn_model is None:
        try:
            print(f"\n[INFO] Loading CNN model from {CNN_MODEL_PATH}...")
            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            cnn_class_names = np.load(CNN_CLASSES_PATH, allow_pickle=True)
            print("[INFO] CNN model and classes loaded successfully.")
            return True
        except (IOError, FileNotFoundError) as e:
            print(f"\n[ERROR] Could not load CNN model or classes file: {e}")
            return False
    return True

def load_lstm_model():
    global lstm_model, lstm_scaler
    if lstm_model is None:
        try:
            print(f"\n[INFO] Loading LSTM model from {LSTM_MODEL_PATH}...")
            if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(LSTM_SCALER_PATH):
                 raise FileNotFoundError("Model or scaler file not found.")
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            with open(LSTM_SCALER_PATH, 'rb') as f:
                lstm_scaler = pickle.load(f)
            print("[INFO] LSTM model and scaler loaded successfully.")
            return True
        except (IOError, FileNotFoundError) as e:
            print(f"\n[ERROR] Could not load LSTM model or scaler: {e}")
            print("[INFO] Please run the training script first to generate these files.")
            return False
    return True



def predict_yield():
    print("\n--- Crop Yield Prediction (XGBoost) ---")
    try:
        model = joblib.load(CROP_YIELD_MODEL_PATH)
    except FileNotFoundError:
        print(f"\n[ERROR] Model not found at '{CROP_YIELD_MODEL_PATH}'. Please train it first.")
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
        'Crop': [crop], 'Crop_Year': [crop_year], 'Season': [season],
        'Area': [area], 'Annual_Rainfall': [rainfall], 'Fertilizer': [fertilizer], 'Pesticide': [pesticide]
    })
    prediction = model.predict(input_data)
    print("\n--- PREDICTION RESULT ---")
    print(f"Predicted Crop Yield: {prediction[0]:.2f} tonnes per hectare")
    print("-------------------------")

def recommend_fertilizer():
    print("\n--- Fertilizer Recommendation (Random Forest) ---")
    try:
        model = joblib.load(FERTILIZER_MODEL_PATH)
    except FileNotFoundError:
        print(f"\n[ERROR] Model not found at '{FERTILIZER_MODEL_PATH}'. Please train it first.")
        return

    print("Please enter the following details:")
    crop = input("Enter Crop Name (e.g., Wheat, Soyabean): ")
    current_n = float(input("Enter current Nitrogen (N) content in soil (kg/ha): "))
    current_p = float(input("Enter current Phosphorus (P) content in soil (kg/ha): "))
    current_k = float(input("Enter current Potassium (K) content in soil (kg/ha): "))

    input_data = pd.DataFrame({
        'Crop': [crop], 'Current_N': [current_n], 'Current_P': [current_p], 'Current_K': [current_k]
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
        model = joblib.load(SOIL_CROP_MODEL_PATH)
    except FileNotFoundError:
        print(f"\n[ERROR] Model not found at '{SOIL_CROP_MODEL_PATH}'. Please train it first.")
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
        'N': [n], 'P': [p], 'K': [k], 'temperature': [temp],
        'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]
    })
    prediction = model.predict(input_data)
    print("\n--- RECOMMENDATION RESULT ---")
    print(f"The most suitable crop for your field is: {prediction[0]}")
    print("-----------------------------")

def forecast_market_price():
    print("\n--- Market Price Forecast (SARIMA) ---")
    crop_name = input("Enter Crop Name to Forecast (e.g., Soybean, Wheat): ").lower()
    forecast_days = int(input("Enter number of days to forecast: "))
    model_path = os.path.join(SARIMA_MODELS_DIR, f'sarima_model_{crop_name}.joblib')
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"\n[ERROR] Model not found at '{model_path}'.")
        return

    forecast = model.predict(n_periods=forecast_days)
    print("\n--- FORECAST RESULT ---")
    print(f"Forecasted prices for the next {forecast_days} periods:")
    for i, price in enumerate(forecast):
        print(f"Period {i+1}: {price:.2f}")
    print("-----------------------")


# --- Helper and Workflow Functions ---

def preprocess_image_for_cnn(img_array):
    img_array = cv2.resize(img_array, CNN_IMG_SIZE)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_cnn_disease(img_array):
    predictions = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_disease = cnn_class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return f"{predicted_disease} (Confidence: {confidence:.2f}%)"

def capture_image_from_camera():
    print("[INFO] Opening webcam... Press 'q' to capture.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None
    
    frame = None
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
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the image: {e}")
        return None
        
def load_and_preprocess_weather_data(data_path):
    print("[INFO] Loading historical weather data for context...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}.")
    df = pd.read_csv(data_path)
    df.index = pd.to_datetime(df['datetime_utc'])
    df = df[REQUIRED_COLS]
    df.fillna(method='ffill', inplace=True)
    df_final = df.resample('D').mean()
    df_final.fillna(method='ffill', inplace=True)
    return df_final

def predict_future_temperatures(start_date_str, num_days, df, model, scaler):
    try:
        start_date = pd.to_datetime(start_date_str)
    except ValueError:
        print("[ERROR] Invalid date format. Please use YYYY-MM-DD.")
        return None

    end_of_historical_data = start_date - pd.Timedelta(days=1)
    if end_of_historical_data not in df.index:
        print(f"[ERROR] The date {end_of_historical_data.strftime('%Y-%m-%d')} is not in the dataset.")
        return None
        
    last_sequence_df = df.loc[:end_of_historical_data].tail(SEQUENCE_LENGTH)
    if len(last_sequence_df) < SEQUENCE_LENGTH:
        print(f"[ERROR] Not enough historical data before {start_date_str}.")
        return None

    current_sequence = scaler.transform(last_sequence_df)
    predicted_temps = {}
    for i in range(num_days):
        input_data = np.reshape(current_sequence, (1, SEQUENCE_LENGTH, NUM_FEATURES))
        scaled_prediction = model.predict(input_data, verbose=0)[0][0]

        dummy_prediction_array = np.zeros((1, NUM_FEATURES))
        dummy_prediction_array[0, TARGET_COL_INDEX] = scaled_prediction
        actual_temp_prediction = scaler.inverse_transform(dummy_prediction_array)[0, TARGET_COL_INDEX]

        prediction_date = start_date + pd.Timedelta(days=i)
        predicted_temps[prediction_date] = actual_temp_prediction

        new_row = current_sequence[-1, :].copy()
        new_row[TARGET_COL_INDEX] = scaled_prediction
        current_sequence = np.vstack([current_sequence[1:], new_row])
    return predicted_temps

def plot_weather_predictions(historical_df, predictions, start_date_str):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    start_date = pd.to_datetime(start_date_str)
    history_start = start_date - pd.Timedelta(days=60)
    history_to_plot = historical_df.loc[history_start:start_date - pd.Timedelta(days=1)]

    ax.plot(history_to_plot.index, history_to_plot[TARGET_COL], label='Historical Temperature', color='royalblue', linewidth=2)
    pred_dates, pred_values = list(predictions.keys()), list(predictions.values())
    ax.plot(pred_dates, pred_values, label='Predicted Temperature', color='darkorange', marker='o', linestyle='--')

    ax.set_title('Weather Temperature Prediction', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



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

        if choice == '1': predict_yield()
        elif choice == '2': recommend_fertilizer()
        elif choice == '3': recommend_crop()
        elif choice == '4': forecast_market_price()
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
            print("\n[ERROR] Invalid choice. Please enter a number between 1 and 7.")

def cnn_sub_menu():
    while True:
        print("\n--- CNN Disease Detection ---")
        print("1. Capture image from camera")
        print("2. Upload image file")
        print("3. Back to main menu")
        cnn_choice = input("Enter your choice (1-3): ").strip()

        img_to_predict = None
        if cnn_choice == '1': img_to_predict = capture_image_from_camera()
        elif cnn_choice == '2': img_to_predict = upload_image_for_cnn()
        elif cnn_choice == '3': break
        else:
            print("[ERROR] Invalid choice. Please enter 1, 2, or 3.")
            continue
        
        if img_to_predict is not None:
            processed_img = preprocess_image_for_cnn(img_to_predict)
            result = predict_cnn_disease(processed_img)
            print(f"\n[CNN Result] {result}")

def lstm_inference_flow():
    print("\n--- LSTM Weather Prediction ---")
    try:
        full_dataframe = load_and_preprocess_weather_data(LSTM_WEATHER_DATA_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    while True:
        min_date = full_dataframe.index[SEQUENCE_LENGTH].strftime('%Y-%m-%d')
        max_date = full_dataframe.index[-1].strftime('%Y-%m-%d')
        print(f"\n(Note: Please enter a date between {min_date} and {max_date})")
        
        input_date = input("Enter start date for prediction (YYYY-MM-DD), or 'back' to return: ").strip()
        if input_date.lower() == 'back':
            break
            
        try:
            input_days = int(input("Enter the number of days to predict: "))
            if input_days <= 0:
                print("[ERROR] Please enter a positive number of days.")
                continue
        except ValueError:
            print("[ERROR] Invalid input. Please enter a whole number for days.")
            continue

        predictions = predict_future_temperatures(input_date, input_days, full_dataframe, lstm_model, lstm_scaler)

        if predictions:
            print("\n--- Predicted Temperatures ---")
            for date, temp in predictions.items():
                print(f"{date.strftime('%Y-%m-%d')}: {temp:.2f}°C")
            
            plot_weather_predictions(full_dataframe, predictions, input_date)

if __name__ == "__main__":
    main_menu()

