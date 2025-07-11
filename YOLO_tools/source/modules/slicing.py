import shutil
import json
import os

""" 
função para dividir imagens e labels em train, val e test
a depender da task, detect ou classify
"""

def slicing_dataset_for_traning(task, imageFolder, annotationsFolder, yoloClasses, TestPercentualDivisor, dataset_path):
    if task == 'detect':
        detect_YOLO_dataset(imageFolder, annotationsFolder, yoloClasses, TestPercentualDivisor, dataset_path)
    if task == 'classify':
        classify_YOLO_dataset(imageFolder, yoloClasses, TestPercentualDivisor, dataset_path)

def detect_YOLO_dataset(imageFolder, annotationsFolder, yoloClasses, TestPercentualDivisor, dataset_path):
    annotationsFolder = (annotationsFolder, "labels")
    list_archives = [imageFolder, annotationsFolder]

    os.makedirs(f"datasets/{dataset_path}", exist_ok=True)

    for folder in list_archives:
        if folder is not None:
            path_folder, name_folder = folder
            files = [f for f in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, f))]
            files.sort()
            total_imgs = len(files)
            n_test = int(total_imgs * TestPercentualDivisor)
            n_val = int(total_imgs * TestPercentualDivisor)
            n_train = total_imgs - n_test - n_val

            for idx, file in enumerate(files):
                path_file = os.path.join(path_folder, file)

                os.makedirs(f"datasets/{dataset_path}/{name_folder}/train", exist_ok=True)
                os.makedirs(f"datasets/{dataset_path}/{name_folder}/val", exist_ok=True)
                os.makedirs(f"datasets/{dataset_path}/{name_folder}/test", exist_ok=True)

                if idx < n_test:
                    destination = f"datasets/{dataset_path}/{name_folder}/test"
                elif idx < n_test + n_val:
                    destination = f"datasets/{dataset_path}/{name_folder}/val"
                else:
                    destination = f"datasets/{dataset_path}/{name_folder}/train"

                path_destination = os.path.join(destination, file)
                shutil.copy(path_file, path_destination)

    # Salva dataset.yaml
    with open(f"datasets/{dataset_path}/dataset.yaml", "w") as file:
        if isinstance(yoloClasses, list):
            nc = len(yoloClasses)
            names = "\n".join([f"  {i}: '{v}'" for i, v in enumerate(yoloClasses)])
        else:
            nc = 1
            names = f"  0: '{yoloClasses}'"
        file.write(
f"""train: images\\train
val: images\\val
test: images\\test
nc: {nc}
names:
{names}
""")

    # Salva notes.json
    if isinstance(yoloClasses, list):
        categories = [{"id": i, "name": name} for i, name in enumerate(yoloClasses)]
    else:
        categories = [{"id": 0, "name": yoloClasses}]
    data = {
        "categories": categories,
        "info": {
            "year": 2024,
            "version": "1.0",
            "contributor": "senrajudson"
        }
    }
    with open(f"datasets/{dataset_path}/notes.json", "w") as file:
        json.dump(data, file, indent=2)

    # Salva classes.txt
    if isinstance(yoloClasses, list):
        classes = "\n".join(yoloClasses)
    else:
        classes = yoloClasses
    with open(f"datasets/{dataset_path}/classes.txt", "w") as file:
        file.write(f"{classes}\n")

def classify_YOLO_dataset(imageFolder, yoloClasses, TestPercentualDivisor, dataset_path):
    os.makedirs(f"datasets/{dataset_path}", exist_ok=True)
    caminho_pasta, nome_pasta = imageFolder

    for folder in os.listdir(caminho_pasta):
        path_folder = os.path.join(caminho_pasta, folder)
        print(folder)
        print(path_folder)
        if folder is not None:
            files = [f for f in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, f))]
            files.sort()
            total_imgs = len(files)
            n_test = int(total_imgs * TestPercentualDivisor)
            n_val = int(total_imgs * TestPercentualDivisor)
            n_train = total_imgs - n_test - n_val

            for idx, file in enumerate(files):
                path_file = os.path.join(path_folder, file)

                os.makedirs(f"datasets/{dataset_path}/train/{folder}", exist_ok=True)
                os.makedirs(f"datasets/{dataset_path}/val/{folder}", exist_ok=True)
                os.makedirs(f"datasets/{dataset_path}/test/{folder}", exist_ok=True)

                if idx < n_test:
                    destination = f"datasets/{dataset_path}/test/{folder}"
                elif idx < n_test + n_val:
                    destination = f"datasets/{dataset_path}/val/{folder}"
                else:
                    destination = f"datasets/{dataset_path}/train/{folder}"

                path_destination = os.path.join(destination, file)
                shutil.copy(path_file, path_destination)

    # Salva classes.txt
    if isinstance(yoloClasses, list):
        classes = "\n".join(yoloClasses)
    else:
        classes = yoloClasses
    with open(f"datasets/{dataset_path}/classes.txt", "w") as file:
        file.write(f"{classes}\n")

    # Salva dataset.yaml
    with open(f"datasets/{dataset_path}/dataset.yaml", "w") as file:
        if isinstance(yoloClasses, list):
            nc = len(yoloClasses)
            names = f"{yoloClasses}"
        else:
            nc = 1
            names = f"['{yoloClasses}']"
        file.write(
f"""train: train
val: val
test: test
nc: {nc}
names: {names}
""")

    # Salva notes.json
    if isinstance(yoloClasses, list):
        categories = [{"id": i, "name": name} for i, name in enumerate(yoloClasses)]
    else:
        categories = [{"id": 0, "name": yoloClasses}]
    data = {
        "categories": categories,
        "info": {
            "year": 2024,
            "version": "1.0",
            "contributor": "senrajudson"
        }
    }
    with open(f"datasets/{dataset_path}/notes.json", "w") as file:
        json.dump(data, file, indent=2)
