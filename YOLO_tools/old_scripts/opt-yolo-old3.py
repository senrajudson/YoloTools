from ultralytics import YOLO
from ray import tune
import ray

# pip install -U ultralytics "ray[tune]"

best_recall = (
    0.0  # Variáveis globais para rastrear o melhor recall e a época correspondente
)
best_precision = 0.0
best_f1score = 0.0
best_mAP50 = 0.0
best_mAP5095 = 0.0
best_epoch = 0
patience = 100
limit = patience


def on_train_epoch_end(trainer):
    global best_recall, best_precision, best_f1score, best_mAP5095, best_mAP50, best_epoch, limit

    logging.basicConfig(  # Configuração do logger
        level=logging.INFO,  # Nível mínimo de mensagens para registrar
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="training.log",  # Arquivo onde as mensagens serão salvas
        filemode="w",  # Sobrescreve o arquivo a cada execução
    )

    # current_recall = trainer.metrics.get('metrics/recall(B)', 0.0)       # Obtenha o recall atual dos resultados de validação
    # current_precision = trainer.metrics.get('metrics/precision(B)', 0.0)    # Obtenha o precision atual dos resultados de validação
    # current_f1score = 2 * (current_precision * current_recall) / (current_precision + current_recall) if current_recall > 0 else 0.0      # f1score

    current_mAP50 = trainer.metrics.get(
        "metrics/mAP50(B)", 0.0
    )  # mAP50 ajuda no melhor 'recall'
    current_mAP5095 = trainer.metrics.get(
        "metrics/mAP50-95(B)", 0.0
    )  # mAP50-95 ajuda no melhor 'precision'

    if (
        current_mAP50 > best_mAP50
    ):  # Verifique se o recall atual é melhor que o melhor recall registrado
        best_mAP50 = current_mAP50
        best_epoch = trainer.epoch

        logging.info(
            f"\nBest actual metric : {round(best_mAP50, 4)} on epoch {best_epoch}"
        )
        limit = patience

        model.save(
            f"best_metric.pt"
        )  # Salve os pesos do modelo para a melhor época com base no recall

    print(trainer.metrics)
    print(f"\nActual mAP50 : {round(current_mAP50, 4)}")
    print(f"\nBest actual metric : {round(best_mAP50, 4)} on epoch {best_epoch}")

    tune.report(mAP50=current_mAP50, mAP5095=current_mAP5095, epoch=trainer.epoch)

    limit -= 1

    if limit == 0:
        logging.warning(f"Patience has reached limit at epoch {trainer.epoch}")
        # logging.error("Erro inesperado no treinamento")

        raise KeyboardInterrupt

    return current_mAP50


### this are a mix of all YOLO built-in augments, if ur implementing manual augments, it's ideal to disable YOLO augments to avoid overlay
from ultralytics.data.augment import (
    Albumentations,
    CenterCrop,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)

albumentations_yolo = Albumentations(p=0.0)
centercrop_yolo = CenterCrop(0)
randomflip_yolo = RandomFlip(p=0.0)
randomhsv_yolo = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.0)
randomperspective_yolo = RandomPerspective(translate=0.0, scale=0.0)

space = {  # Configurar o espaço de busca
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(1e-5, 1e-2),
    "weight_decay": tune.uniform(1e-3, 1e-2),
    "momentum": tune.uniform(0.8, 0.95),
    "warmup_epochs": tune.randint(1, 5),
    "warmup_momentum": tune.uniform(0.4, 0.8),
    "warmup_bias_lr": tune.uniform(1e-5, 1e-1),
    # "epochs": tune.randint(50, 200),
    "epochs": 70,
    "optimizer": tune.choice(["AdamW", "SGD"]),
    "imgsz": tune.choice([360, 480, 640]),
    "batch": tune.randint(8, 48),
}

import logging

logging.basicConfig(  # Configuração do log
    filename="app_errors.log",  # Nome do arquivo de log
    level=logging.ERROR,  # Nível de log configurado para capturar erros
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato da mensagem de log
)

model = YOLO(r"yolo11n.pt")
data_yaml = (
    r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml"
)
ray.init(
    include_dashboard=True,
    _temp_dir=r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions",
)
model.add_callback(
    "on_train_epoch_end", on_train_epoch_end
)  # Adicione o callback personalizado ao modelo

while True:

    try:

        results = model.tune(
            data=data_yaml,
            use_ray=True,
            iterations=100,
            space=space,
            gpu_per_trial=1,
            project_name="YOLO11n-emissoes-no-yolo-aug",
        )

        print(results)

    except Exception as e:

        logging.error(f"Erro capturado: {e}")

        import time

        time.sleep(20)

# 'ray metrics launch-prometheus'
# '.\prometheus-3.2.1.windows-amd64\prometheus.exe --config.file='./prometheus.yaml''
# 'netstat -aon | findstr :9090'
# 'taskkill /PID <PID> /F'
