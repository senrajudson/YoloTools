import optuna
import os
import json
import logging
from ultralytics import YOLO
# Remova ou comente as importações relacionadas ao ray/tune se não forem necessárias.

# Inicialize o modelo
model = YOLO(r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\yolo11n.pt")

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",
    filemode="w",
)

# Variáveis globais para rastrear a melhor métrica
best_mAP50 = 0.0
best_epoch = 0
patience = 100
limit = patience

def on_train_epoch_end(trainer):
    global best_mAP50, best_epoch, limit, patience

    # Obter a métrica atual
    current_mAP50 = trainer.metrics.get('metrics/mAP50(B)', 0.0)
    current_mAP5095 = trainer.metrics.get('metrics/mAP50-95(B)', 0.0)

    # Atualiza a melhor métrica se a atual for melhor
    if current_mAP50 > best_mAP50:
        best_mAP50 = current_mAP50
        best_epoch = trainer.epoch
        logging.info(f"Melhor mAP50 atual: {round(best_mAP50, 4)} na época {best_epoch}")
        limit = patience  # Reinicia a paciência
        model.save('best_metric.pt')

    print(trainer.metrics)
    print(f"mAP50 atual: {round(current_mAP50, 4)}")
    print(f"Melhor mAP50 até agora: {round(best_mAP50, 4)} na época {best_epoch}")

    limit -= 1
    if limit == 0:
        logging.warning(f"Patience atingido na época {trainer.epoch}")
        raise KeyboardInterrupt

    return current_mAP50

def training(config):

    # Remove callbacks anteriores e adiciona o callback customizado
    model.reset_callbacks()
    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    # Executa o treinamento
    model.train(
        data=r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml",
        device="cuda",
        batch=config['batch'],
        epochs=300,  # ou ajuste conforme necessário
        imgsz=config['imgsz'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        optimizer=config['optimizer'],
        warmup_bias_lr=config['warmup_bias_lr'],
        warmup_epochs=config['warmup_epochs'],
        warmup_momentum=config['warmup_momentum'],
        weight_decay=config['weight_decay'],
    )
    print('Treinamento finalizado para esta configuração.')

def objective(trial):
    global best_mAP50, best_epoch, limit, patience

    # Reinicia os indicadores a cada novo trial
    best_mAP50 = 0.0
    best_epoch = 0
    limit = patience

    # Define o espaço de busca usando o objeto trial
    config = {
        "lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
        "lrf": trial.suggest_float("lrf", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-2),
        "momentum": trial.suggest_float("momentum", 0.8, 0.95),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 1, 5),
        "warmup_momentum": trial.suggest_float("warmup_momentum", 0.4, 0.8),
        "warmup_bias_lr": trial.suggest_float("warmup_bias_lr", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ['AdamW', "SGD"]),
        "imgsz": trial.suggest_categorical("imgsz", [360, 480, 640]),
        "batch": trial.suggest_int("batch", 8, 48),
    }

    try:
        training(config)
    except KeyboardInterrupt:
        logging.info("Treinamento interrompido por early stopping.")

    # O Optuna espera que a função objetivo retorne a métrica a ser otimizada.
    # Aqui, assumimos que queremos maximizar o mAP50.
    return best_mAP50

if __name__ == "__main__":
    # Cria o estudo especificando que a métrica deve ser maximizada
    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///yolo11-opt.db',
        study_name='yolo11-opt',
        load_if_exists=True
    )
    # Número de trials pode ser ajustado conforme sua necessidade
    study.optimize(objective, n_trials=100)
    print("Melhor valor de mAP50:", study.best_value)
    print("Melhores hiperparâmetros:", study.best_params)
