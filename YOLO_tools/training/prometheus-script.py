import shutil
import time
import os

def copy_file():

    # Caminho do arquivo que o Ray Tune atualiza
    src_file = r"C:/Users/AUTOU4/AppData/Local/Temp/ray/prom_metrics_service_discovery.json"

    # Pasta onde o Prometheus consome o arquivo (defina um diretório, não o arquivo)
    dest_folder = r"C:/Users/AUTOU4/AppData/Local/Temp/ray"

    # Garante que a pasta de destino existe
    os.makedirs(dest_folder, exist_ok=True)

    # Caminho completo para o arquivo de destino
    dest_file = os.path.join(dest_folder, "copy_prom_metrics_service_discovery.json")

    try:
        # Tenta copiar o arquivo
        shutil.copy2(src_file, dest_file)
        print(f"Arquivo copiado para {dest_file}")
    except Exception as e:
        print("Erro ao copiar o arquivo:", e)

while True:

    copy_file()
    
    # Aguarda 10 segundos antes de tentar novamente
    time.sleep(10)
