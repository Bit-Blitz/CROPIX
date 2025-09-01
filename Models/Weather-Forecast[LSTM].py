import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore


data_path = 'Datasets/Weather_data.csv'
df = pd.read_csv(data_path)
df.index = pd.to_datetime(df.datetime_utc)
required_cols = [' _dewptm', ' _fog', ' _hail', ' _hum', ' _rain', ' _snow', ' _tempm', ' _thunder', ' _tornado']
df = df[required_cols]


df.fillna(method='ffill', inplace=True)
df_final = df.resample('D').mean()
df_final.fillna(method='ffill', inplace=True)


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)
sequence_length = 10
num_features = len(df_final.columns)
sequences = []
labels = []


for i in range(len(scaled_data) - sequence_length):
    seq = scaled_data[i:i + sequence_length]
    label = scaled_data[i + sequence_length][6]
    sequences.append(seq)
    labels.append(label)


sequences = np.array(sequences)
labels = np.array(labels)
train_x, test_x, train_y, test_y = train_test_split(sequences, labels)


model = Sequential()
model.add(LSTM(units=128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('Trained_models/LSTM_model.keras', monitor='val_loss', save_best_only=True)
history = model.fit(
    train_x, train_y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)


test_loss = model.evaluate(test_x, test_y)
print("Test Loss:", test_loss)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
predictions = model.predict(test_x)
mae = mean_absolute_error(test_y, predictions)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=-1)
true_temp = scaler.inverse_transform(test_y_copies)[:, 6]
prediction_copies = np.repeat(predictions, 9, axis=-1)
predicted_temp = scaler.inverse_transform(prediction_copies)[:, 6]

true_temp_series = pd.Series(true_temp, index=df_final.index[-len(true_temp):])
predicted_temp_series = pd.Series(predicted_temp.flatten(), index=df_final.index[-len(predicted_temp):])


window_size = 30
true_temp_smoothed = true_temp_series.rolling(window=window_size).mean()
predicted_temp_smoothed = predicted_temp_series.rolling(window=window_size).mean()

plt.figure(figsize=(10, 6))
plt.plot(true_temp_smoothed, label='Actual (30-day MA)')
plt.plot(predicted_temp_smoothed, label='Predicted (30-day MA)')
plt.title('Temperature Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

