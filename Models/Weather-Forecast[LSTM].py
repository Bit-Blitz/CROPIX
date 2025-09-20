import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt

def create_sequences(data, targets, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(targets[i + sequence_length])
    return np.array(X), np.array(y)

def plot_training_history(history):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
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

    # Convert 'time' to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # --- Focus on a single city for this model ---
    city_to_model = 'Mumbai'
    print(f"Filtering data for city: {city_to_model}")
    df_city = df[df['city'] == city_to_model].copy()

    if df_city.empty:
        print(f"Error: No data found for city '{city_to_model}'. Please choose another city.")
        print(f"Available cities: {df['city'].unique().tolist()}")
        return

    # --- 2. Feature Engineering and Preprocessing ---
    print("Starting preprocessing...")
    # One-hot encode categorical features
    categorical_cols = ['wind_dir', 'condition_text', 'moon_phase']
    df_city = pd.get_dummies(df_city, columns=categorical_cols, drop_first=True)

    # Define target and feature columns
    target_cols = ['temp_c', 'precip_mm', 'humidity']
    
    # Drop non-numeric/identifier columns before scaling
    df_city = df_city.drop(columns=['state', 'city', 'sunrise', 'sunset', 'moonrise', 'moonset'])
    
    # Ensure all columns are numeric
    for col in df_city.columns:
        if df_city[col].dtype == 'object':
            df_city[col] = pd.to_numeric(df_city[col], errors='coerce')
    df_city.dropna(inplace=True)

    features = df_city.columns.tolist()

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_city)

    # Create a new DataFrame with scaled values
    df_scaled = pd.DataFrame(scaled_data, columns=features, index=df_city.index)
    
    # Prepare data for sequencing
    scaled_features_data = df_scaled[features].values
    scaled_targets_data = df_scaled[target_cols].values

    # --- 3. Create Time-Series Sequences ---
    SEQUENCE_LENGTH = 24  # Use 24 hours of data to predict the next hour
    X, y = create_sequences(scaled_features_data, scaled_targets_data, SEQUENCE_LENGTH)
    
    if len(X) == 0:
        print("Error: Not enough data to create sequences. Try a shorter SEQUENCE_LENGTH or use a city with more data.")
        return

    print(f"Created {len(X)} sequences.")

    # --- 4. Train-Test Split (Chronological) ---
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training data shape: {X_train.shape}")
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

    # --- 6. Train the Model ---
    print("\nStarting model training...")
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,  
        batch_size=32,
        validation_split=0.1, 
        verbose=1,
        callbacks=[early_stopping] 
    )
    print("Model training complete.")
    
    print("\nEvaluating model on test data...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test Mean Absolute Error: {test_mae:.4f}")

    plot_training_history(history)


if __name__ == '__main__':
    main()

