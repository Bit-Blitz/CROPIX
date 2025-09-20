import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback # type: ignore
import matplotlib.pyplot as plt
import os

class R2ScoreCallback(Callback):
    def __init__(self, validation_data, scaler, features, target_cols):
        super().__init__()
        self.validation_data = validation_data
        self.scaler = scaler
        self.features = features
        self.target_cols = target_cols

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_scaled = self.validation_data
        y_pred_scaled = self.model.predict(X_val, verbose=0)

        num_features = len(self.features)
        target_indices = [self.features.index(col) for col in self.target_cols]

        y_pred_dummy = np.zeros((len(y_pred_scaled), num_features))
        for i, index in enumerate(target_indices):
            y_pred_dummy[:, index] = y_pred_scaled[:, i]
        y_pred_unscaled = self.scaler.inverse_transform(y_pred_dummy)[:, target_indices]
        
        y_val_dummy = np.zeros((len(y_val_scaled), num_features))
        for i, index in enumerate(target_indices):
            y_val_dummy[:, index] = y_val_scaled[:, i]
        y_val_unscaled = self.scaler.inverse_transform(y_val_dummy)[:, target_indices]
        
        r2 = r2_score(y_val_unscaled[:, 0], y_pred_unscaled[:, 0])
        logs['val_r2_score'] = r2
        print(f" - val_r2_score (temp): {r2:.4f}")

def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

def plot_training_history(history):
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('Unified Model Training History', fontsize=16)

    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Model Loss (Mean Squared Error)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(history.history['mean_absolute_error'], label='Training MAE')
    axs[1].plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    axs[1].set_title('Model Mean Absolute Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].legend()
    axs[1].grid(True)

    if 'val_r2_score' in history.history:
        axs[2].plot(history.history['val_r2_score'], label='Validation R2 Score (Temp)')
        axs[2].set_title('R2 Score (Temperature) Over Epochs')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('R2 Score')
        axs[2].legend()
        axs[2].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_predictions_vs_actuals(model, X_test, y_test, scaler, features, target_cols):
    print("\nMaking predictions on test data...")
    y_pred_scaled = model.predict(X_test)

    num_features = len(features)
    y_pred_dummy = np.zeros((len(y_pred_scaled), num_features))
    target_indices = [features.index(col) for col in target_cols]
    
    for i, index in enumerate(target_indices):
        y_pred_dummy[:, index] = y_pred_scaled[:, i]
    y_pred_unscaled = scaler.inverse_transform(y_pred_dummy)[:, target_indices]
    
    y_test_dummy = np.zeros((len(y_test), num_features))
    for i, index in enumerate(target_indices):
        y_test_dummy[:, index] = y_test[:, i]
    y_test_unscaled = scaler.inverse_transform(y_test_dummy)[:, target_indices]

    print("\n--- Final R2 Scores on Test Data ---")
    for i, col in enumerate(target_cols):
        r2 = r2_score(y_test_unscaled[:, i], y_pred_unscaled[:, i])
        print(f"{col}: {r2:.4f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_unscaled[:, 0], y_pred_unscaled[:, 0], alpha=0.5)
    plt.plot([y_test_unscaled[:, 0].min(), y_test_unscaled[:, 0].max()], [y_test_unscaled[:, 0].min(), y_test_unscaled[:, 0].max()], 'r--', lw=2)
    plt.title(f'Unified Model: Actual vs. Predicted Temperature (R2 Score: {r2_score(y_test_unscaled[:, 0], y_pred_unscaled[:, 0]):.4f})')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.grid(True)
    plt.show()


def main():
    try:
        df = pd.read_csv('Datasets/LSTM_data/weather_data.csv')
    except FileNotFoundError:
        print("Error: 'weather_data.csv' not found. Please ensure the file is in the correct directory.")
        return

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['city', 'time'], inplace=True)
    df.set_index('time', inplace=True)
    
    print("Starting preprocessing for the unified model...")
    categorical_cols = ['city', 'wind_dir', 'condition_text', 'moon_phase']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    target_cols = ['temp_c', 'precip_mm', 'humidity']
    df = df.drop(columns=['state', 'sunrise', 'sunset', 'moonrise', 'moonset'])
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    features = df.columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features, index=df.index)
    
    print("Creating sequences for each city...")
    all_X, all_y = [], []
    # Find original city names from the one-hot encoded columns
    original_cities = [col for col in df.columns if col.startswith('city_')]

    for city_col in original_cities:
        city_name = city_col.replace('city_', '')
        
        # Filter the scaled data for the current city
        city_data_scaled = df_scaled[df_scaled[city_col] == 1]
        
        if len(city_data_scaled) < 100:
            print(f"Skipping {city_name} due to insufficient data.")
            continue

        scaled_features_data = city_data_scaled[features].values
        scaled_targets_data = city_data_scaled[target_cols].values

        SEQUENCE_LENGTH = 24
        X_city, y_city = create_sequences(scaled_features_data, scaled_targets_data, SEQUENCE_LENGTH)
        
        if len(X_city) > 0:
            all_X.append(X_city)
            all_y.append(y_city)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total sequences created from all cities: {len(X)}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    model.summary()

    print("\nStarting unified model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    r2_callback = R2ScoreCallback(validation_data=(X_val, y_val), scaler=scaler, features=features, target_cols=target_cols)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64, # Increased batch size for larger dataset
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, r2_callback]
    )
    print("Model training complete.")

    output_dir = 'Trained_models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filename = os.path.join(output_dir, 'LSTM_model.keras')
    print(f"\nSaving the trained model to '{model_filename}'...")
    model.save(model_filename)
    print("Model successfully saved.")

    print(f"\nEvaluating unified model on test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test Mean Absolute Error: {test_mae:.4f}")

    plot_training_history(history)
    plot_predictions_vs_actuals(model, X_test, y_test, scaler, features, target_cols)

if __name__ == '__main__':
    main()

