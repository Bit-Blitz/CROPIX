import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt

class R2ScoreCallback(Callback):
    """
    Custom callback to calculate R2 score on validation data at the end of each epoch.
    """
    def __init__(self, validation_data, scaler, features, target_cols):
        super().__init__()
        self.validation_data = validation_data
        self.scaler = scaler
        self.features = features
        self.target_cols = target_cols

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_scaled = self.validation_data
        y_pred_scaled = self.model.predict(X_val)

        # Inverse transform to get actual values
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
        
        # Calculate R2 for the first target (temperature)
        r2 = r2_score(y_val_unscaled[:, 0], y_pred_unscaled[:, 0])
        logs['val_r2_score'] = r2
        print(f" - val_r2_score (temp): {r2:.4f}")

def create_sequences(data, targets, sequence_length):
    """
    Creates sequences of data for time-series forecasting.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

def plot_training_history(history):
    """
    Plots the training/validation loss, MAE, and R2 score.
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))

    # Plotting Loss
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Model Loss (Mean Squared Error)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting Mean Absolute Error
    axs[1].plot(history.history['mean_absolute_error'], label='Training MAE')
    axs[1].plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    axs[1].set_title('Model Mean Absolute Error')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MAE')
    axs[1].legend()
    axs[1].grid(True)

    # Plotting R2 Score
    if 'val_r2_score' in history.history:
        axs[2].plot(history.history['val_r2_score'], label='Validation R2 Score (Temp)')
        axs[2].set_title('R2 Score (Temperature) Over Epochs')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('R2 Score')
        axs[2].legend()
        axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actuals(model, X_test, y_test, scaler, features, target_cols):
    """
    Makes predictions, calculates R2 score, and plots actual vs. predicted values.
    """
    print("\nMaking predictions on test data...")
    y_pred_scaled = model.predict(X_test)

    # We need to inverse transform the scaled data to get meaningful values
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

    # Plotting actual vs. predicted for the first target (Temperature)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_unscaled[:, 0], y_pred_unscaled[:, 0], alpha=0.5)
    plt.plot([y_test_unscaled[:, 0].min(), y_test_unscaled[:, 0].max()], [y_test_unscaled[:, 0].min(), y_test_unscaled[:, 0].max()], 'r--', lw=2)
    plt.title(f'Actual vs. Predicted Temperature (R2 Score: {r2_score(y_test_unscaled[:, 0], y_pred_unscaled[:, 0]):.4f})')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to load data, preprocess, build, train, and plot model performance.
    """
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv('weather_data.csv')
    except FileNotFoundError:
        print("Error: 'weather_data.csv' not found. Please ensure the file is in the correct directory.")
        return

    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    city_to_model = 'Mumbai'
    print(f"Filtering data for city: {city_to_model}")
    df_city = df[df['city'] == city_to_model].copy()

    if df_city.empty:
        print(f"Error: No data found for city '{city_to_model}'. Please choose another city.")
        print(f"Available cities: {df['city'].unique().tolist()}")
        return

    # --- 2. Feature Engineering and Preprocessing ---
    print("Starting preprocessing...")
    categorical_cols = ['wind_dir', 'condition_text', 'moon_phase']
    df_city = pd.get_dummies(df_city, columns=categorical_cols, drop_first=True)

    target_cols = ['temp_c', 'precip_mm', 'humidity']
    df_city = df_city.drop(columns=['state', 'city', 'sunrise', 'sunset', 'moonrise', 'moonset'])
    
    for col in df_city.columns:
        if df_city[col].dtype == 'object':
            df_city[col] = pd.to_numeric(df_city[col], errors='coerce')
    df_city.dropna(inplace=True)

    features = df_city.columns.tolist()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_city)
    df_scaled = pd.DataFrame(scaled_data, columns=features, index=df_city.index)
    
    scaled_features_data = df_scaled[features].values
    scaled_targets_data = df_scaled[target_cols].values

    # --- 3. Create Time-Series Sequences ---
    SEQUENCE_LENGTH = 24
    X, y = create_sequences(scaled_features_data, scaled_targets_data, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("Error: Not enough data to create sequences. Try a shorter SEQUENCE_LENGTH or use a city with more data.")
        return
    print(f"Created {len(X)} sequences.")

    # --- 4. Train-Validation-Test Split (Chronological) ---
    train_split_index = int(len(X) * 0.7)
    val_split_index = int(len(X) * 0.8)

    X_train, X_val, X_test = X[:train_split_index], X[train_split_index:val_split_index], X[val_split_index:]
    y_train, y_val, y_test = y[:train_split_index], y[train_split_index:val_split_index], y[val_split_index:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- 5. Build the LSTM Model ---
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

    # --- 6. Train the Model ---
    print("\nStarting model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    r2_callback = R2ScoreCallback(validation_data=(X_val, y_val), scaler=scaler, features=features, target_cols=target_cols)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[early_stopping, r2_callback]
    )
    print("Model training complete.")

    # --- 7. Evaluate and Plot ---
    print("\nEvaluating model on test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test Mean Absolute Error: {test_mae:.4f}")

    plot_training_history(history)
    plot_predictions_vs_actuals(model, X_test, y_test, scaler, features, target_cols)

if __name__ == '__main__':
    main()

