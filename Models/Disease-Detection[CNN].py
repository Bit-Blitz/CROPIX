import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

image_dir = 'Datasets/CNN_Images_dataset/'
labels_csv_path = 'Datasets/CNN_labels.csv'
plot_save_dir = 'METRICS/Disease_Detection[CNN]/'
model_save_path = 'Trained_models/CNN_MODEL/Disease_Detection_model[CNN].h5'
classes_save_path = 'Trained_models/CNN_MODEL/disease_classes.npy'


IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 32
EPOCHS = 25 


try:
    df = pd.read_csv(labels_csv_path)
    df['filepath'] = df['filepath'].apply(lambda f: os.path.join(os.path.dirname(labels_csv_path), 'CNN_Images_dataset', os.path.basename(os.path.dirname(f)), os.path.basename(f)))
    print(f"Successfully loaded {labels_csv_path} with {len(df)} entries.")
except FileNotFoundError:
    print(f"ERROR: '{labels_csv_path}' not found. Please ensure the file exists.")
    exit()

encoder = LabelEncoder()
df['encoded_label'] = encoder.fit_transform(df['label'])
NUM_CLASSES = len(df['label'].unique())
print(f"Found {NUM_CLASSES} unique disease classes.")

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df, x_col='filepath', y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)
validation_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df, x_col='filepath', y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df, x_col='filepath', y_col='label',
    target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',     
    patience=2,             
    restore_best_weights=True
)

print("\n--- Starting Model Training ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping] 
)
print("--- Model Training Complete ---")

print("\n--- Evaluating Model on Test Data ---")
results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")


model.save(model_save_path)
np.save(classes_save_path, encoder.classes_)
print(f"\nModel saved successfully to '{model_save_path}'")
print(f"Label classes saved to '{classes_save_path}'")


def save_plots(history, save_dir):

    print(f"\n--- Saving performance plots to '{save_dir}' ---")
    

    os.makedirs(save_dir, exist_ok=True)
    

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
   
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'accuracy_loss_plots.png')
    plt.savefig(plot_path)
    print(f"Plots saved successfully to '{plot_path}'")
    plt.close()

save_plots(history, plot_save_dir)