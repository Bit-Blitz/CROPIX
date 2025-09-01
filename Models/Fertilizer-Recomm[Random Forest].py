import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model_save_path = 'Trained_models/fertilizer_recommendation_model.joblib'
plot_folder = 'METRICS/Fertilizer_Recom[Random Forest]'
os.makedirs(plot_folder, exist_ok=True)

training_data = pd.read_csv('Datasets/fertilizer_training_data.csv')

X = training_data[['Crop', 'Current_N', 'Current_P', 'Current_K']]
y = training_data[['Required_N', 'Required_P', 'Required_K']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

preprocessor = ColumnTransformer(transformers=[('crop_encoder', OneHotEncoder(handle_unknown='ignore'), ['Crop'])], remainder='passthrough')

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

ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['crop_encoder'].get_feature_names_out(['Crop'])
all_feature_names = list(ohe_feature_names) + ['Current_N', 'Current_P', 'Current_K']

regressor = model_pipeline.named_steps['regressor']
importances = regressor.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances (Random Forest)', fontsize=16)
plt.xlabel('Importance (Gini)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'feature_importance.png'))
plt.show()


targets = ['Required_N', 'Required_P', 'Required_K']
for i, target in enumerate(targets):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test[target], y=y_pred[:, i], alpha=0.6)
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()],
             'r--', lw=2)
    plt.title(f'Predicted vs. Actual Values for {target}', fontsize=16)
    plt.xlabel(f'Actual {target} (kg/hectare)', fontsize=12)
    plt.ylabel(f'Predicted {target} (kg/hectare)', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f'predicted_vs_actual_{target}.png'))
    plt.show()

joblib.dump(model_pipeline, model_save_path)
print(f"\nModel saved to: '{model_save_path}'")