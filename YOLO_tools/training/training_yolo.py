from ultralytics import YOLO
import logging
import json


### this are a mix of all YOLO built-in augments, if ur implementing manual augments, it's ideal to disable YOLO augments to avoid overlay
from ultralytics.data.augment import (
    Albumentations,
    CenterCrop,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)


### YOLO é gambiarra e eu posso provar:
""" Para treinamentos de classificação com YOLO, você deve indicar o dir com o dataset
que deve estar especificado dentro de uma pasta chamada 'datasets'. No entando, para detecção
o YOLO é diferente. Você precisa indicar o caminho do arquivo 'dataset.yaml' para que ele 
possa encontrar o dataset e realizar o treinamento. É a mesma função, de uma mesma lib,
mas os caras fizeram de forma que o mesmo argumento recebe duas entradas completamente 
diferentes a depender do treinamento que você vai fazer.
"""


with open(
    r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\params.json", "r"
) as file:  # Carregar configurações de um arquivo
    config = json.load(file)

model = YOLO(r"yolo11n.pt")

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",
    filemode="w",
)

# # # variáveis globais para configuração
task = "detect"
range_of_search = 100

# Variáveis globais para rastrear a melhor métrica
best_mAP50 = 0.0
best_mAP5095 = 0.0
best_acc = 0.0
best_loss = 100.0
best_epoch = 0
best_metric = 0.0
limit = patience = range_of_search
current_metric_value = 0
current_metric_support = 0
last_epoch = 0


def on_train_epoch_end(trainer):
    global best_mAP50, best_epoch, best_acc, best_mAP5095, best_metric, best_loss, limit, patience, current_metric_value, current_metric_support, last_epoch

    if task == "detect":

        # Obter a métrica atual
        current_metric_value = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        current_metric_support = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)

        # Atualiza a melhor métrica se a atual for melhor
        if current_metric_value > best_mAP50 or current_metric_support > best_mAP5095:
            best_mAP50 = current_metric_value
            best_epoch = trainer.epoch
            logging.info(
                f"Melhor mAP50 atual: {round(best_mAP50, 4)} na época {best_epoch}"
            )
            limit = patience  # Reinicia a paciência
            model.save("best_metric.pt")

        print(trainer.metrics)
        print(
            f"mAP50 atual: {round(current_metric_value, 4)} | mAP5095 atual: {round(current_metric_support, 4)} | Época atual: {trainer.epoch}"
        )
        print(f"Melhor até agora na época: {best_epoch}")

        best_metric = best_mAP50 if best_mAP50 > best_metric else best_metric

    if task == "classify":

        # Pegue a métrica de accuracy da classificação
        current_metric_value = trainer.metrics.get("metrics/accuracy_top1", 0.0)
        current_metric_support = trainer.metrics.get("val/loss", 0.0)
        # ou experimente "metrics/acc(B)", depende do YOLO

        if current_metric_value > best_acc or current_metric_support < best_loss:
            best_acc = current_metric_value
            best_loss = current_metric_support if current_metric_support != 0 else 100.0
            best_epoch = trainer.epoch
            logging.info(
                f"Melhor accuracy atual: {round(best_acc, 4)} na época {best_epoch}"
            )
            limit = patience
            model.save("best_metric.pt")

        print(trainer.metrics)
        print(
            f"Accuracy atual: {round(current_metric_value, 4)} | Loss atual: {round(current_metric_support, 4)} | Época atual: {trainer.epoch}"
        )
        print(f"Melhor até agora na época: {best_epoch}")

        best_metric = best_acc if best_acc > best_metric else best_metric
        last_epoch = trainer.epoch

    limit -= 1
    if limit == 0:
        logging.warning(f"Patience atingido na época {trainer.epoch}")
        raise KeyboardInterrupt

    return current_metric_value


# model.add_callback(
#     "on_train_epoch_end", on_train_epoch_end
# )  # Adicione o callback personalizado ao modelo
###

def training():

    # Iniciar o treinamento com os parâmetros do JSON
    model.train(
        data=r"D:/Judson_projetos/Yolo_trainer/YOLO_tools/datasets/emissoes_completo_yolo_1607/dataset.yaml",
        device="cuda",
        patience=50,
        # workers=0,
        **config
    )

if __name__ == "__main__":
    training()
