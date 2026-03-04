from pathlib import Path
from PIL import Image

img_dir = Path(r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_completo_yolo_1607\images\val")

for img_path in img_dir.rglob("*.[jp][pn]g"):  # pega .jpg e .png
    try:
        with Image.open(img_path) as im:
            im.verify()  # testa se a imagem está OK
    except Exception as e:
        print(f"Imagem corrompida: {img_path} -> {e}")
