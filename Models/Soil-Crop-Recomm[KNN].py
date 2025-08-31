import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os 

model_path = 'Trained_models/Soil_crop_recom.joblib'

df = pd.read_csv("Datasets/Soil_Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=35)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.4f}%")

print(f"Saving the model to: {model_path}")
joblib.dump(knn, model_path)
print("Model saved successfully!")