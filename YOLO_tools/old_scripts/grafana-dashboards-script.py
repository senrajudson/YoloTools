import os
import shutil
import time

# Diretório de origem (onde o Ray Tune salva os dashboards)
src_dir = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions\session_latest\metrics\grafana\dashboards"

# Diretório de destino (pasta persistente)
dest_dir = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\grafana_dashboards\session_latest\metrics\grafana\dashboards"

# Garante que o diretório de destino exista
os.makedirs(dest_dir, exist_ok=True)

while True:
    try:
        # Lista todos os arquivos do diretório de origem
        for filename in os.listdir(src_dir):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            # Se for um arquivo, copia para o destino
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dest_file)
                print(f"Arquivo '{filename}' copiado para {dest_dir}")
    except Exception as e:
        print("Erro durante a cópia dos arquivos:", e)

    # Aguarda 10 segundos antes de repetir
    time.sleep(10)
