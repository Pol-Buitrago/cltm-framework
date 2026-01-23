import pandas as pd
import os
import json

# --- Configuración ---
# ❗❗ Reemplaza esta ruta con la ruta correcta en tu sistema ❗❗
BASE_DIR = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/data/processed/cv-22.0/cv-corpus-22.0-2025-06-20__gender_id/05_balanced_120_cv_gender/tsv"
OUTPUT_FILE = "/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/scripts/labels/language_abbreviation_map.txt"
# Nombre de la columna que contiene el nombre completo del idioma
LANG_COLUMN = "lang" 

# Diccionario para almacenar el mapeo: Abreviatura -> Nombre Completo
language_map = {}

print(f"Buscando archivos .train.tsv en: {BASE_DIR}")

# 1. Recorrer la carpeta y encontrar archivos .train.tsv
for filename in os.listdir(BASE_DIR):
    if filename.endswith(".train.tsv"):
        
        # Extraer la abreviatura del idioma (ej: 'ca' de 'ca.train.tsv')
        abbreviation = filename.split(".")[0]
        full_path = os.path.join(BASE_DIR, filename)
        
        # Verificar si ya tenemos el mapeo para evitar reprocesar si hay archivos duplicados
        if abbreviation in language_map:
            continue
            
        print(f"Procesando: {filename} (Abreviatura: {abbreviation})")
        
        try:
            # 2. Leer el archivo TSV
            # Especificamos el separador '\t'
            df = pd.read_csv(full_path, sep='\t')
            
            # 3. Extraer el nombre completo del idioma
            if LANG_COLUMN in df.columns:
                # Tomamos el primer valor no nulo de esa columna.
                full_lang_name = df[LANG_COLUMN].dropna().iloc[0]
                
                # Almacenar en el mapeo
                language_map[abbreviation] = full_lang_name
                print(f" -> Mapeado: {abbreviation} -> {full_lang_name}")
                
            else:
                print(f" -> ¡ADVERTENCIA! Columna '{LANG_COLUMN}' no encontrada en {filename}. Saltando.")
                
        except pd.errors.EmptyDataError:
            print(f" -> ADVERTENCIA: El archivo {filename} está vacío. Saltando.")
        except Exception as e:
            print(f" -> ERROR al leer {filename}: {e}. Saltando.")

# 4. Generar el fichero de salida en formato JSON
if language_map:
    output_path = os.path.join(os.getcwd(), OUTPUT_FILE)
    
    # Escribir el diccionario en un archivo JSON con formato legible
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(language_map, f, indent=4, ensure_ascii=False)
            
    print(f"\n🎉 ¡Éxito! Mapeo generado y guardado en formato JSON en: {output_path}")
    print(f"Este archivo es directamente cargable en Python como un diccionario.")
else:
    print("\nAVISO: No se encontraron archivos '.train.tsv' o no se pudo extraer ninguna información.")