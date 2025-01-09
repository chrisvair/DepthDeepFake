import os
import subprocess



def run_faceswap_pipeline(input_folder, output_folder, model_folder):
    # Étape 1: Extraction 
    extract_command = [
        'python', 'faceswap.py', 'extract',
        '-i', input_folder, 
        '-o', os.path.join(output_folder, 'faces')
    ]
    
    print("Extraction des visages en cours...")
    subprocess.run(extract_command, check=True)
    
    # Étape 2: Conversion 
    convert_command = [
        'python', 'faceswap.py', 'convert',
        '-i', input_folder,
        '-o', output_folder,
        '-m', model_folder
    ]
    
    print("Conversion des visages en cours...")
    subprocess.run(convert_command, check=True)
    
    print("Pipeline terminé avec succès!")

input_folder = '/srv/src/christina'  # Remplacer par le chemin de votre dossier d'entrée
output_folder = '/srv/output_christina_lou'  # Remplacer par le chemin de votre dossier de sortie
model_folder = '/srv/model_christina_lou'

run_faceswap_pipeline(input_folder, output_folder, model_folder)