import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import os
import pandas as pd
from PIL import Image

def create_image_dataset_csv(root_dir, output_csv='qr_dataset.csv'):
    """
    Crea un CSV con metadata de imágenes organizadas por carpetas
    
    Estructura esperada:
    root_dir/
        ├── benign/benign/
        │   ├── img1.png
        │   └── img2.png
        ├── malicious/malicious/
        │   ├── img1.png
        │   └── img2.png
    """
    
    data = []
    
    # Mapeo de etiquetas
    label_map = {
        'normal': 0,
        'benign': 0,
        'legitimate': 0,
        'phishing': 1,
        'quishing': 1,
        'malware': 2,
        'malicious': 2
    }
    
    # Recorrer primer nivel (ej: benign/, malicious/)
    for class_dir in os.listdir(root_dir):
        class_dir_path = os.path.join(root_dir, class_dir)
        
        if not os.path.isdir(class_dir_path):
            continue
        
        # Recorrer segundo nivel (ej: benign/benign/)
        for sub_dir in os.listdir(class_dir_path):
            sub_dir_path = os.path.join(class_dir_path, sub_dir)
            
            if not os.path.isdir(sub_dir_path):
                continue
            
            # Determinar etiqueta usando la subcarpeta
            label_name = sub_dir.lower()
            label = label_map.get(label_name, -1)
            
            if label == -1:
                print(f"⚠️ Carpeta '{sub_dir}' no reconocida, saltando...")
                continue
            
            # Procesar imágenes
            for img_file in os.listdir(sub_dir_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(sub_dir_path, img_file)
                    
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            mode = img.mode
                        
                        data.append({
                            'filepath': img_path,
                            'filename': img_file,
                            'label': label,
                            'label_name': ['normal', 'phishing', 'malware'][label],
                            'width': width,
                            'height': height,
                            'mode': mode,
                            'size_bytes': os.path.getsize(img_path)
                        })
                    
                    except Exception as e:
                        print(f"❌ Error procesando {img_path}: {e}")
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Estadísticas
    print("\nRESUMEN DEL DATASET")
    print(f"Total de imágenes: {len(df)}")
    
    if not df.empty:
        print("\nDistribución por clase:")
        print(df['label_name'].value_counts())
        
        print("\nTamaños de imagen:")
        print(df.groupby('label_name')[['width', 'height']].describe())
    
    # Guardar CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV guardado en: {output_csv}")
    
    return df

def prepare_qr_dataset(data_dir='data/raw', output_dir='data/processed'):
    """
    Pipeline completo de preparación de dataset
    """
    
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Crear CSV con metadata
    print("📊 Paso 1: Creando CSV de metadata...")
    df = create_image_dataset_csv(data_dir, f'{output_dir}/qr_full_dataset.csv')
    
    # 2. Análisis exploratorio básico
    print("\n📈 Paso 2: Análisis exploratorio...")
    print(df.groupby('label_name').agg({
        'filepath': 'count',
        'width': ['mean', 'std'],
        'height': ['mean', 'std'],
        'size_bytes': ['mean', 'min', 'max']
    }))
    
    # 3. Split del dataset
    print("\n✂️ Paso 3: Dividiendo dataset...")
    from sklearn.model_selection import train_test_split
    
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # 4. Guardar metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'n_classes': df['label'].nunique(),
        'class_distribution': df['label_name'].value_counts().to_dict(),
        'image_sizes': {
            'mean_width': df['width'].mean(),
            'mean_height': df['height'].mean()
        }
    }
    
    import json
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset preparado en: {output_dir}")
    return df

if __name__ == "__main__":
    df = prepare_qr_dataset('data/kaggle-qr-codes/QR codes', 'data_procesada')
