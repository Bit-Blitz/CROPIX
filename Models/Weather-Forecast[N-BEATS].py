import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError # type: ignore


DATA_PATH = 'Datasets/Weather_data.csv'
MODEL_DIR = 'Trained_models/LSTM_model'
METRICS_DIR = 'METRICS/Weather_forecast[LSTM]'
MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'weather_scaler.pkl')

SEQUENCE_LENGTH = 10
REQUIRED_COLS = [' _dewptm', ' _fog', ' _hail', ' _hum', ' _rain', ' _snow', ' _tempm', ' _thunder', ' _tornado']
TARGET_COL = ' _tempm'
TARGET_COL_INDEX = REQUIRED_COLS.index(TARGET_COL)

def load_and_preprocess_data(data_path):
    print("Loading and preprocessing data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please ensure the CSV file is in the correct directory.")
        
    df = pd.read_csv(data_path)
    df.index = pd.to_datetime(df['datetime_utc'])
    df = df[REQUIRED_COLS]

    df.fillna(method='ffill', inplace=True)
    df_final = df.resample('D').mean()
    df_final.fillna(method='ffill', inplace=True)
    print("Data preprocessing complete.")
    return df_final

def train_and_save_model(df, model_path, scaler_path, metrics_path):
    print("Starting model training...")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    sequences, labels = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        seq = scaled_data[i:i + SEQUENCE_LENGTH]
        label = scaled_data[i + SEQUENCE_LENGTH][TARGET_COL_INDEX]
        sequences.append(seq)
        labels.append(label)

    sequences, labels = np.array(sequences), np.array(labels)
    train_x, test_x, train_y, test_y = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(units=128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_x, train_y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model and scaler saved successfully to '{MODEL_DIR}' directory.")
    
    print(f"Saving metric plots to '{metrics_path}'...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_path, 'model_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model Training and Validation Accuracy (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_path, 'model_accuracy.png'))
    plt.close()

    predictions = model.predict(test_x)
    
    test_y_copies = np.repeat(test_y.reshape(-1, 1), scaled_data.shape[1], axis=-1)
    true_temp = scaler.inverse_transform(test_y_copies)[:, TARGET_COL_INDEX]
    
    prediction_copies = np.repeat(predictions, scaled_data.shape[1], axis=-1)
    predicted_temp = scaler.inverse_transform(prediction_copies)[:, TARGET_COL_INDEX]

    sample_size = 500
    smoothing_window = 15
    true_series = pd.Series(true_temp[:sample_size])
    predicted_series = pd.Series(predicted_temp.flatten()[:sample_size])
    
    true_smoothed = true_series.rolling(window=smoothing_window).mean()
    predicted_smoothed = predicted_series.rolling(window=smoothing_window).mean()
    
    plt.figure(figsize=(15, 7))
    plt.plot(true_smoothed, label=f'Actual Temperature ({smoothing_window}-Day MA)', color='royalblue', alpha=0.8)
    plt.plot(predicted_smoothed, label=f'Predicted Temperature ({smoothing_window}-Day MA)', color='darkorange', linestyle='--')
    plt.title(f'Smoothed Actual vs. Predicted Temperatures (First {sample_size} Test Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_path, 'actual_vs_predicted_smoothed.png'))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(true_temp, predicted_temp, alpha=0.3, s=10)
    plt.plot([min(true_temp), max(true_temp)], [min(true_temp), max(true_temp)], color='red', linestyle='--', label='Perfect Prediction Line')
    plt.title('Prediction Correlation on Test Data')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(metrics_path, 'prediction_correlation.png'))
    plt.close()

    print("Metrics plots saved.")

    return model, scaler

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    full_dataframe = load_and_preprocess_data(DATA_PATH)
    train_and_save_model(full_dataframe, MODEL_PATH, SCALER_PATH, METRICS_DIR)



