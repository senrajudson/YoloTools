import optuna
import logging
from ultralytics import YOLO

"""_summary_
o YOLO tem modelos específicos para cada tarefa. Nunca se esqueça de trocar os modelos, pois a 
sua task precisa do modelo correto.

Não se esqueça de modificar também a função de callback para retornar as métricas do seu modelo.
"""

# Remova ou comente as importações relacionadas ao ray/tune se não forem necessárias.

# Inicialize o modelo
model = YOLO(r"yolo11s.pt")

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",
    filemode="w",
)

# # # variáveis globais para configuração
task = "detect"
current_metric_value = 0


def on_train_epoch_end(trainer):
    global current_metric_value

    if task == "detect":

        # Obter a métrica atual
        current_metric_value = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)

    if task == "classify":

        # Pegue a métrica de accuracy da classificação
        current_metric_value = trainer.metrics.get("metrics/accuracy_top1", 0.0)

    return current_metric_value


def training(config):

    # Remove callbacks anteriores e adiciona o callback customizado
    model.reset_callbacks()
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # Executa o treinamento
    model.train(
        data=r"D:/Judson_projetos/Yolo_trainer/YOLO_tools/datasets/emissoes_completo_yolo_1607/dataset.yaml",
        device="cuda",
        patience=10,
        **config
    )

    print("Treinamento finalizado para esta configuração.")


def objective(trial):
    global current_metric_value

    # Define o espaço de busca usando o objeto trial
    config = {
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
        "lrf": trial.suggest_float("lrf", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "momentum": trial.suggest_float("momentum", 0.6, 0.98),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
        "warmup_momentum": trial.suggest_float("warmup_momentum", 0.4, 0.8),
        "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]),
        "imgsz": trial.suggest_categorical("imgsz", [360, 480, 640]),
        "batch": trial.suggest_int("batch", 8, 32),
        "epochs": trial.suggest_int("epochs", 30, 200),
        "box": trial.suggest_float("box", 0.02, 0.2),
        "cls": trial.suggest_float("cls", 0.2, 4.0),
        "kobj": trial.suggest_float("kobj", 0.2, 4.0),
        # Data augmentation params
        # "hsv_h": trial.suggest_float("hsv_h", 0.0, 0.1),
        # "degrees": trial.suggest_float("degrees", 0.0, 45.0),
        # "translate": trial.suggest_float("translate", 0.0, 0.9),
        # "scale": trial.suggest_float("scale", 0.0, 0.9),
        # "shear": trial.suggest_float("shear", 0.0, 10.0),
        # "perspective": trial.suggest_float("perspective", 0.0, 0.001),
        # "flipud": trial.suggest_float("flipud", 0.0, 1.0),
        # "fliplr": trial.suggest_float("fliplr", 0.0, 1.0),
        # "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
        # "mixup": trial.suggest_float("mixup", 0.0, 1.0),
        # "copy_paste": trial.suggest_float("copy_paste", 0.0, 1.0),
    }

    training(config)

    # O Optuna espera que a função objetivo retorne a métrica a ser otimizada.
    # Aqui, assumimos que queremos maximizar o mAP5095.
    return current_metric_value


if __name__ == "__main__":
    # Cria o estudo especificando que a métrica deve ser maximizada
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///yolo11-opt.db",
        study_name="yolo11-emissoes-1807250005",
        load_if_exists=True,
    )

    # Número de trials pode ser ajustado conforme sua necessidade
    study.optimize(objective, n_trials=200)
    print("Melhor valor da métrica: ", study.best_value)
    print("Melhores hiperparâmetros: ", study.best_params)

# # # optuna-dashboard sqlite:///yolo11-opt.db --server=wsgiref --port=8070
