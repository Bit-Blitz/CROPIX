import os
import pathlib
import shutil
import kagglehub
from PIL import Image

DATASETS_TO_DOWNLOAD = {
    "plantdisease": "emmarex/plantdisease",
    "rice-leaf-diseases": "vbookshelf/rice-leaf-diseases",
    "wheat-disease": "sinadunk23/behzad-safari-jalal"
}


output_dir = 'Datasets/CNN_Images_dataset'

IMG_SIZE = (256, 256)
IMG_QUALITY = 80

def download_datasets():
    print("--- STEP 1: DOWNLOADING DATASETS ---")
    print("This may take several minutes for the first run...")
    
    downloaded_paths = {}
    for key, dataset_path in DATASETS_TO_DOWNLOAD.items():
        try:
            print(f"\nDownloading: {dataset_path}...")
            path = kagglehub.dataset_download(dataset_path)
            downloaded_paths[key] = path
            print(f"Successfully located '{key}' at: {path}")
        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Failed to download '{dataset_path}'.")
            print("Please check your Kaggle authentication and the dataset path.")
            print(f"Error details: {e}")
            return None
            
    print("\n--- All datasets downloaded successfully! ---")
    return downloaded_paths

def process_and_merge(source_paths):
    print("\n--- STEP 2: PROCESSING AND MERGING IMAGES ---")
    print(f"Output will be saved to '{output_dir}/'")

    if os.path.exists(output_dir):
        print(f"Removing existing '{output_dir}' directory...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    total_images_processed = 0

    for dataset_key, dataset_path in source_paths.items():
        print(f"\n--- Processing dataset: {dataset_key} ---")
        source_path = pathlib.Path(dataset_path)
        category_folders = sorted(list(set(p.parent for p in source_path.glob('**/*.[jp][pn]g'))))

        if not category_folders:
            print(f"No image category folders found in {dataset_path}. Skipping.")
            continue

        for category_path in category_folders:
            category_name = category_path.name
            output_category_dir = os.path.join(output_dir, category_name)
            os.makedirs(output_category_dir, exist_ok=True)
            
            images = list(category_path.glob('*.[jp][pn]g'))
            if not images: continue
            
            print(f"  -> Processing category: {category_name} ({len(images)} images)")

            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img_rgb = img.convert('RGB')
                        img_resized = img_rgb.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                        new_filename = f"{dataset_key}_{img_path.name}"
                        save_path = os.path.join(output_category_dir, new_filename)
                        img_resized.save(save_path, 'jpeg', quality=IMG_QUALITY)
                        total_images_processed += 1
                except Exception as e:
                    print(f"    - Could not process {img_path}. Error: {e}")

    print(f"\n--- PROCESSING COMPLETE ---")
    print(f"Successfully processed and merged {total_images_processed} images.")
    print(f"Your final dataset is ready in the '{output_dir}' folder.")


if __name__ == "__main__":
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: 'Pillow' is required. Please run: pip install Pillow")
        exit()
        
    downloaded_paths = download_datasets()
    
    if downloaded_paths:
        process_and_merge(downloaded_paths)
    else:
        print("\nAborting processing due to download failure.")