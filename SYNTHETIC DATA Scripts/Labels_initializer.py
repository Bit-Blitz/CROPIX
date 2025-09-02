import os
import pandas as pd

dataset_path = 'Datasets/CNN_Images_dataset' 
output_csv_path = 'Datasets/CNN_labels.csv'

def generate_labels_csv():
    print(f"Scanning directory: '{dataset_path}'...")

    if not os.path.exists(dataset_path):
        print(f"ERROR: Directory not found at '{dataset_path}'")
        return

    image_data = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file_name)
                file_path = file_path.replace('\\', '/')
                

                label = os.path.basename(root)
                image_data.append({'filepath': file_path, 'label': label})

    if not image_data:
        print("No image files were found in the directory.")
        return

    labels_df = pd.DataFrame(image_data)
    labels_df.to_csv(output_csv_path, index=False)
    
    print(f"\n--- Success! ---")
    print(f"Successfully created '{output_csv_path}' with {len(labels_df)} entries.")
    print("Here's a preview of the corrected file paths:")
    print(labels_df.head())

if __name__ == "__main__":
    generate_labels_csv()