import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt
import os

PROCESSED_DATA_FILE = "Datasets/LSTM_data/Weather_data_daily.csv"
MODELS_DIR = "lstm_models"

LOOK_BACK = 30
FORECAST_HORIZON = 10
EARLY_STOPPING_PATIENCE = 15

def create_sliding_window(dataset, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[(i + look_back):(i + look_back + forecast_horizon), 0])
    return np.array(X), np.array(y)

def train_lstm_model_for_location(df, location):
    print(f"\n--- Processing location: {location} ---")
    
    location_df = df[df['location_id'] == location].copy()
    
    data = location_df['temp'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = create_sliding_window(scaled_data, LOOK_BACK, FORECAST_HORIZON)

    if len(X) == 0:
        print(f"Skipping {location}: Not enough data to create a single training sample.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        LSTM(50),
        Dense(FORECAST_HORIZON)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model_path = os.path.join(MODELS_DIR, f"lstm_model_{location}.keras")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                        callbacks=[early_stopping, model_checkpoint], verbose=2)

    print(f"Training complete. Best model saved to {model_path}")
    
def main():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Error: Processed data file '{PROCESSED_DATA_FILE}' not found.")
        print("Please run the 'preprocess_data.py' script first.")
        return
        
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    df = pd.read_csv(PROCESSED_DATA_FILE)
    locations = df['location_id'].unique()
    
    for location in locations:
        train_lstm_model_for_location(df, location)

if __name__ == "__main__":
    main()

