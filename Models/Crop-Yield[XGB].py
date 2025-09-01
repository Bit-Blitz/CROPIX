import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

plot_folder = 'METRICS/Crop_Yield[XGB]'
os.makedirs(plot_folder, exist_ok=True)

df = pd.read_csv('Datasets/Crop_yield.csv')
df = df[df['Area'] > 0]
df['Yield'] = df['Production'] / df['Area']

X = df.drop(columns=['State', 'Production', 'Yield'])
y = df['Yield']

categorical_features = ['Crop', 'Season']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=250,
        learning_rate=0.1,
        max_depth=20,
        random_state=35,
    ))
])

print("\nTraining Model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

y_pred = model_pipeline.predict(X_test)

print("\n--- Model Performance Metrics ---")
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²) Score: {r2:.4f}")
print("---------------------------------")
print("MAE shows the average prediction error in tonnes/hectare.")
print("R² shows the percentage of variance in yield that the model can explain (closer to 1.0 is better).")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs. Actual Crop Yield', fontsize=16)
plt.xlabel('Actual Yield (tonnes/hectare)', fontsize=12)
plt.ylabel('Predicted Yield (tonnes/hectare)', fontsize=12)
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid(True)
plt.savefig(os.path.join(plot_folder, 'predicted_vs_actual.png'))
plt.show()

model_save_path = 'Trained_models/CROP_YIELD_MODEL.joblib'
joblib.dump(model_pipeline, model_save_path)
print(f"\nModel successfully saved to '{model_save_path}'")