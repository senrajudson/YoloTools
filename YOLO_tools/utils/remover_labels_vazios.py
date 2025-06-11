def processar_dataset(dataset_dir):
    import shutil
    import os
    import numpy as np

    base_labels_dir = os.path.join(dataset_dir, "labels")
    base_images_dir = os.path.join(dataset_dir, "images")
    sem_nada_labels_dir = os.path.join(base_labels_dir, "sem_nada_lb")
    sem_nada_images_dir = os.path.join(base_images_dir, "sem_nada_im")
    os.makedirs(sem_nada_labels_dir, exist_ok=True)
    os.makedirs(sem_nada_images_dir, exist_ok=True)

    for root, dirs, files in os.walk(base_labels_dir):
        if "sem_nada_lb" in dirs:
            dirs.remove("sem_nada_lb")
        for filename in files:
            if not filename.lower().endswith(".txt"):
                continue
            label_path = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, base_labels_dir)
            if rel_dir == ".":
                rel_dir = ""
            image_dir = os.path.join(base_images_dir, rel_dir)
            base_name = os.path.splitext(filename)[0]
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = os.path.join(image_dir, base_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            try:
                if os.path.getsize(label_path) == 0:
                    shutil.move(label_path, os.path.join(sem_nada_labels_dir, filename))
                    if image_path:
                        shutil.move(
                            image_path,
                            os.path.join(
                                sem_nada_images_dir, os.path.basename(image_path)
                            ),
                        )
                    continue
            except Exception:
                shutil.move(label_path, os.path.join(sem_nada_labels_dir, filename))
                if image_path:
                    shutil.move(
                        image_path,
                        os.path.join(sem_nada_images_dir, os.path.basename(image_path)),
                    )
                continue

            invalid = False
            corrected_lines = []
            try:
                with open(label_path, "r") as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
            except Exception:
                invalid = True

            if not invalid and len(lines) == 0:
                invalid = True

            if not invalid:
                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        invalid = True
                        break
                    try:
                        class_id = int(parts[0])
                        bbox = np.array([float(x) for x in parts[1:5]])
                        bbox = np.clip(bbox, 0.0, 1.0)
                        corrected_line = (
                            f"{class_id} {' '.join(f'{x:.6f}' for x in bbox)}"
                        )
                        corrected_lines.append(corrected_line)
                    except:
                        invalid = True
                        break

            if invalid:
                shutil.move(label_path, os.path.join(sem_nada_labels_dir, filename))
                if image_path:
                    shutil.move(
                        image_path,
                        os.path.join(sem_nada_images_dir, os.path.basename(image_path)),
                    )
            else:
                with open(label_path, "w") as f:
                    for cline in corrected_lines:
                        f.write(cline + "\n")


# Exemplo de uso:
# dataset_dir = 'YOLO_tools/emissoes_dataset_YOLO'
# processar_dataset(dataset_dir)
