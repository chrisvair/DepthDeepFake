# DepthDeepFake


## Partie faceswap:

#Extraction: 
To extract the faces from photos in a folder:
python faceswap.py extract -i ~/faceswap/src/christina -o ~/faceswap/faces/christina
To extract faces from a video file:
python faceswap.py extract -i ~/faceswap/src/lou.mp4 -o ~/faceswap/faces/lou

#Conversion: 
python faceswap.py convert -i ~/faceswap/src/christina/ -o ~/faceswap/output_christina_lou/ -m ~/faceswap/model_christina_lou/ 


