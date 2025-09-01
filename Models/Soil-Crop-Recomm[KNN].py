import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os 
import matplotlib.pyplot as plt
import numpy as np

model_path = 'Trained_models/Soil_crop_recom.joblib'
plot_folder = 'METRICS/Soil_Crop_Recom[KNN]'
os.makedirs(plot_folder, exist_ok=True)

df = pd.read_csv("Datasets/Soil_Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=35)

neighbors = np.arange(1, 10)
accuracies = []

for k in neighbors:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

plt.figure(figsize=(10, 6))
plt.plot(neighbors, accuracies, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(neighbors)
plt.grid(True)
plt.savefig(os.path.join(plot_folder, 'elbow_method.png'))
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.4f}%")

print(f"\nSaving the model to: {model_path}")
joblib.dump(knn, model_path)
print("Model saved successfully!")