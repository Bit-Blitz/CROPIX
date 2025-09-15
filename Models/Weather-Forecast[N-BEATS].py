import pandas as pd
import numpy as np
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import NBEATS, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

PROCESSED_DATA_FILE = "Datasets/NBeats Model/Weather_data_daily.csv"

def train_weather_model(data_file):
    print("\n--- Starting Long-Range Daily Model Training ---")
    if not os.path.exists(data_file):
        print(f"Error: Processed data file '{data_file}' not found.")
        print("Please run the 'preprocess_data.py' script first.")
        return

    data_df = pd.read_csv(data_file, parse_dates=['date'])
    print(f"Successfully loaded preprocessed daily data from {data_file}.")

    data_df['time_idx'] = (data_df['date'] - data_df['date'].min()).dt.days
    data_df = data_df.drop_duplicates(subset=['location_id', 'time_idx'])

    max_prediction_length = 60  # Predict 60 days ahead
    max_encoder_length = 90     # Use last 90 days of data as input
    training_cutoff = data_df["time_idx"].max() - max_prediction_length

    training_data = TimeSeriesDataSet(
        data_df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="temp",
        group_ids=["location_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=['pressure', 'humidity', 'wind_speed', 'precipitation', 'cloud_cover', 'uv'],
        target_normalizer=GroupNormalizer(groups=["location_id"], transformation="softplus"),
    )

    validation_data = TimeSeriesDataSet.from_dataset(training_data, data_df, predict=True, stop_randomization=True)
    batch_size = 128
    train_dataloader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_data.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    print(f"Created dataloaders for a {max_prediction_length}-day forecast using {max_encoder_length} days of history.")


    checkpoint_callback = ModelCheckpoint(dirpath=".", filename="best_nbeats_model_daily_60day", monitor="val_loss", mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=50, accelerator="auto", enable_model_summary=True,
        gradient_clip_val=0.1, callbacks=[early_stop_callback, checkpoint_callback],
    )

    nbeats = NBEATS.from_dataset(
        training_data, learning_rate=3e-2, weight_decay=1e-2,
        widths=[32, 512], backcast_loss_ratio=0.1,
    )
    print(f"Number of parameters in network: {nbeats.size()/1e3:.1f}k")

    # --- Step 5: Train the Model ---
    print("\nStarting model training...")
    trainer.fit(nbeats, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("Model training finished.")
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")
    print("\n--- Model Training Complete! You can now run the prediction servers. ---")


if __name__ == "__main__":
    train_weather_model(PROCESSED_DATA_FILE)

