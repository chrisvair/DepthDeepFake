import os
import subprocess
import shutil


def run_faceswap_pipeline_single_image(input_image, output_image, model_folder):
    temp_input_folder = "/tmp/faceswap_input"
    output_folder = os.path.dirname(output_image)
    
    os.makedirs(temp_input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    temp_input_image = os.path.join(temp_input_folder, os.path.basename(input_image))
    shutil.copy(input_image, temp_input_image)

    try:
        # extraction
        extract_command = [
            'python', 'faceswap.py', 'extract',
            '-i', temp_input_folder,
            '-o', temp_input_folder
        ]
        print("Extraction des visages en cours...")
        subprocess.run(extract_command, check=True)
        
        # conversion
        convert_command = [
            'python', 'faceswap.py', 'convert',
            '-i', temp_input_folder,
            '-o', output_folder,
            '-m', model_folder
        ]
        print("Conversion du visage en cours...")
        subprocess.run(convert_command, check=True)

        # Vérifier les fichiers générés
        generated_files = os.listdir(output_folder)
        if not generated_files:
            raise RuntimeError("Aucun fichier généré par la conversion.")

        # on utilise le premier fichier généré
        generated_file = os.path.join(output_folder, generated_files[0])
        shutil.move(generated_file, output_image)
        print(f"Pipeline terminé avec succès ! Image générée dans: {output_image}")
    finally:
        shutil.rmtree(temp_input_folder)

input_image = '/srv/src/christina/zoomed_person_5.png'  # Remplacez par le chemin de votre image d'entrée
output_image = '/srv/output_christina_lou'  # Remplacez par le chemin de votre image de sortie
model_folder = '/srv/models/model_christina_lou'

run_faceswap_pipeline_single_image(input_image, output_image, model_folder)
