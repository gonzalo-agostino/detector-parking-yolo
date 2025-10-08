# =====================================================================================
# SCRIPT DE CONVERSIÓN DE DATASET DE PARKING A FORMATO YOLO (VERSIÓN CORREGIDA)
# =====================================================================================
#
# INSTRUCCIONES:
# 1. Asegúrate de que en tu carpeta de proyecto (`Proyecto_Parking`) tienes:
#    - Este script (`convert.py`).
#    - Tu archivo de anotaciones (`annotations.xml`).
#    - La carpeta `images` con todas las imágenes originales.
#
# 2. Abre una terminal, activa tu entorno (`.\venv\Scripts\activate`) y ejecuta:
#    python convert.py
#
# =====================================================================================

import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURACIÓN ---
CLASS_MAPPING = {
    'free_parking_space': 0,
    'partially_free_parking_space': 0, # Mapeado a 'libre'
    'not_free_parking_space': 1,       # Mapeado a 'ocupado'
}
XML_FILE = 'annotations.xml'
SOURCE_IMAGES_DIR = 'images'
OUTPUT_DATASET_DIR = 'datasets'

# --- LÓGICA DEL SCRIPT ---

def convert_polygon_to_yolo_bbox(polygon_points, img_width, img_height):
    points_x = [p[0] for p in polygon_points]
    points_y = [p[1] for p in polygon_points]
    min_x, max_x = min(points_x), max(points_x)
    min_y, max_y = min(points_y), max(points_y)
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    center_x = min_x + bbox_width / 2
    center_y = min_y + bbox_height / 2
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = bbox_width / img_width
    norm_height = bbox_height / img_height
    return norm_center_x, norm_center_y, norm_width, norm_height

def main():
    print("Iniciando la conversión del dataset al formato YOLO...")

    print(f"Leyendo anotaciones desde '{XML_FILE}'...")
    try:
        tree = ET.parse(XML_FILE)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo '{XML_FILE}'.")
        return
    except ET.ParseError:
        print(f"ERROR: El archivo '{XML_FILE}' está corrupto.")
        return

    all_images_data = []
    for image_node in tqdm(root.findall('image'), desc="Procesando imágenes del XML"):
        img_name_from_xml = image_node.get('name')
        img_width = int(image_node.get('width'))
        img_height = int(image_node.get('height'))
        
        yolo_annotations = []
        for polygon_node in image_node.findall('polygon'):
            label = polygon_node.get('label')
            if label in CLASS_MAPPING:
                class_id = CLASS_MAPPING[label]
                points_str = polygon_node.get('points').split(';')
                points = [tuple(map(float, p.split(','))) for p in points_str]
                yolo_bbox = convert_polygon_to_yolo_bbox(points, img_width, img_height)
                yolo_annotations.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
        
        if yolo_annotations:
            all_images_data.append({
                'name': img_name_from_xml, # Guardamos el nombre original por ahora
                'annotations': "\n".join(yolo_annotations)
            })

    print(f"Se procesaron {len(all_images_data)} imágenes con anotaciones válidas.")

    print("Dividiendo el dataset...")
    train_val_data, test_data = train_test_split(all_images_data, test_size=0.10, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=(10/90), random_state=42)
    print(f"  - Entrenamiento: {len(train_data)}, Validación: {len(val_data)}, Prueba: {len(test_data)}")

    datasets_to_create = {'train': train_data, 'valid': val_data, 'test': test_data}

    if os.path.exists(OUTPUT_DATASET_DIR):
        print(f"Eliminando la carpeta '{OUTPUT_DATASET_DIR}' existente...")
        shutil.rmtree(OUTPUT_DATASET_DIR)

    print("Creando la nueva estructura de carpetas y archivos YOLO...")
    for set_name, data in datasets_to_create.items():
        images_dir = os.path.join(OUTPUT_DATASET_DIR, set_name, 'images')
        labels_dir = os.path.join(OUTPUT_DATASET_DIR, set_name, 'labels')
        os.makedirs(images_dir)
        os.makedirs(labels_dir)

        for item in tqdm(data, desc=f"Creando conjunto '{set_name}'"):
            # ===== INICIO DE LA CORRECCIÓN =====
            img_filename_with_path = item['name']
            # Obtenemos solo el nombre del archivo, sin la ruta 'images/'
            img_filename = os.path.basename(img_filename_with_path)
            
            base_filename = os.path.splitext(img_filename)[0]
            
            source_img_path = os.path.join(SOURCE_IMAGES_DIR, img_filename)
            dest_img_path = os.path.join(images_dir, img_filename)

            if not os.path.exists(source_img_path):
                print(f"ADVERTENCIA: No se encontró la imagen '{source_img_path}'. Se omitirá.")
                continue # Si la imagen no existe, saltamos a la siguiente

            shutil.copy(source_img_path, dest_img_path)
            
            label_filepath = os.path.join(labels_dir, f"{base_filename}.txt")
            with open(label_filepath, 'w') as f:
                f.write(item['annotations'])
            # ===== FIN DE LA CORRECIÓN =====
                
    print("\n¡Proceso completado con éxito!")
    print(f"Tu dataset en formato YOLO está listo en la carpeta '{OUTPUT_DATASET_DIR}'.")

if __name__ == '__main__':
    main()