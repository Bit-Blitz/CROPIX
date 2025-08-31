import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib

model_save_path = 'Trained_models/fertilizer_recommendation_model.joblib'

training_data = pd.read_csv('Datasets/fertilizer_training_data.csv')

X = training_data[['Crop', 'Current_N', 'Current_P', 'Current_K']]
y = training_data[['Required_N', 'Required_P', 'Required_K']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

preprocessor = ColumnTransformer(transformers=[('crop_encoder', OneHotEncoder(handle_unknown='ignore'), ['Crop'])],remainder='passthrough')

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=52))
])

model_pipeline.fit(X_train, y_train)
print("Model training complete.")

print("\n--- Model Performance Metrics ---")
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")

joblib.dump(model_pipeline, model_save_path)
print(f"\nModel saved to: '{model_save_path}'")