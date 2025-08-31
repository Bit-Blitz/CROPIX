import pandas as pd
import numpy as np

crop_nutrient_table = {
    'Wheat': {'N': 120, 'P': 60, 'K': 40},
    'Soyabean': {'N': 20, 'P': 80, 'K': 40},
    'Maize': {'N': 150, 'P': 75, 'K': 50},
    'Cotton': {'N': 120, 'P': 60, 'K': 60},
    'Gram': {'N': 20, 'P': 60, 'K': 20},
    'Paddy': {'N': 120, 'P': 60, 'K': 60},
    'Mustard': {'N': 80, 'P': 40, 'K': 40},
    'Lentil': {'N': 20, 'P': 60, 'K': 20},
    'Groundnut': {'N': 25, 'P': 60, 'K': 40},
    'Sugarcane': {'N': 275, 'P': 85, 'K': 60},
    'Potato': {'N': 150, 'P': 80, 'K': 120},
    'Arhar/Tur': {'N': 20, 'P': 80, 'K': 40}
}

def generate_synthetic_data(num_samples=5000):
    data = []
    crops = list(crop_nutrient_table.keys())
    
    for _ in range(num_samples):
        crop = np.random.choice(crops)
        ideal_n, ideal_p, ideal_k = crop_nutrient_table[crop].values()
        
        current_n = np.random.randint(0, int(ideal_n * 1.5))
        current_p = np.random.randint(0, int(ideal_p * 1.5))
        current_k = np.random.randint(0, int(ideal_k * 1.5))
        
        required_n = max(0, ideal_n - current_n)
        required_p = max(0, ideal_p - current_p)
        required_k = max(0, ideal_k - current_k)
        
        data.append([crop, current_n, current_p, current_k, required_n, required_p, required_k])
        
    columns = ['Crop', 'Current_N', 'Current_P', 'Current_K', 'Required_N', 'Required_P', 'Required_K']
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    synthetic_dataset = generate_synthetic_data(num_samples=10000) 
    
    
    file_path = '~/Projects/CROPIX/Datasets/fertilizer_training_data.csv'
    synthetic_dataset.to_csv(file_path, index=False)
    
    print(f"Successfully created and saved the expanded dataset to '{file_path}'")
    print("Dataset preview:")
    print(synthetic_dataset.sample(10)) 