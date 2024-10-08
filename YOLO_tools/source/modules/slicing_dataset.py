import os
import shutil
import json

""" 
função para dividir imagens e labels em val e train
"""

def slicing_dataset_for_traning(imageFolder, annotationsFolder, yoloClasses, TestPercentualDivisor):

    list_archives = [imageFolder, annotationsFolder]
    yolo_dataset_dir = "dataset/dataset_YOLO"
    os.makedirs(yolo_dataset_dir, exist_ok=True)

    for folder in list_archives:
        if folder != None:
            path_folder, name_folder = folder
            counter = 0
            for file in os.listdir(path_folder):
                path_file = os.path.join(path_folder, file)

                if os.path.isfile(path_file):
                    os.makedirs(f"dataset/dataset_YOLO/{name_folder}/train", exist_ok=True)
                    os.makedirs(f"dataset/dataset_YOLO/{name_folder}/val", exist_ok=True)

                    if counter >= (len(os.listdir(path_folder))*(100-int(TestPercentualDivisor))/100):
                        destination = f"dataset/dataset_YOLO/{name_folder}/val"
                    else:
                        destination = f"dataset/dataset_YOLO/{name_folder}/train"
                        counter += 1
                    
                    path_destination = os.path.join(destination, file)
                    shutil.copy(path_file, path_destination)

    with open("dataset/dataset_YOLO/dataset.yaml", "w") as file:
        file.write(
f"""path: dataset_YOLO
train: images/train
val: images/val
names: ["{yoloClasses}"]
"""
        )

    # Construa o dicionário de dados.
    data = {
        "categories": [
            {
                "id": 0,
                "name": yoloClasses
            }
        ],
        "info": {
            "year": 2024,
            "version": "1.0",
            "contributor": "senrajudson"
        }
    }

    with open("dataset/dataset_YOLO/notes.json", "w") as file:
        json.dump(data, file, indent=2)

    with open("dataset/dataset_YOLO/classes.txt", "w") as file:
        file.write(
f"""{yoloClasses}
"""
        )