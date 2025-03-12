from ray.tune.schedulers import ASHAScheduler
from training_yolo import training
from ray.air import RunConfig
from ray import tune
import logging
import hashlib
import json
import ray
import os

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
gpu_per_trial = 1

def train_yolo(space):
    # Escreve o dicionário 'space' no arquivo 'params.json'
    with open(r'D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\params.json', 'w') as file:
        json.dump(space, file, indent=4)

    training()

# Definindo o espaço de busca dos hiperparâmetros
space={     # Configurar o espaço de busca
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(1e-5, 1e-2),
    "weight_decay": tune.uniform(1e-3, 1e-2),
    "momentum": tune.uniform(0.8, 0.95),
    "warmup_epochs": tune.randint(1, 5),
    "warmup_momentum": tune.uniform(0.4, 0.8),
    "warmup_bias_lr": tune.uniform(1e-5, 1e-1),
    # "epochs": tune.randint(50, 200),
    # "epochs": 70,
    "optimizer": tune.choice(['AdamW', "SGD"]),
    "imgsz": tune.choice([360, 480, 640]),
    "batch": tune.choice([8, 16, 32, 48, 64]),
    }

folder_project = f"D:/Judson_projetos/Yolo_trainer/YOLO_tools/training"

ray.init(runtime_env={"working_dir": folder_project}, include_dashboard=True, _temp_dir=f"{folder_project}/ray_sessions")
# ray.init(runtime_env={"working_dir": folder_project}, _temp_dir=f"{folder_project}/ray_sessions",)

def shorten_trial_dirname(trial):  # Ajustando para receber o objeto trial
    return hashlib.md5(trial.trial_id.encode()).hexdigest()[:8]    # Gerar um hash curto do trial_id para garantir que o nome do diretório seja único e curto

# Define the trainable function with allocated resources
trainable_with_resources = tune.with_resources(train_yolo, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})
# trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": 1})

# Define the ASHA scheduler for hyperparameter search
asha_scheduler = ASHAScheduler(
    time_attr="epoch",
    metric="mAP50",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=3,
)

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=space,
    tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=10, trial_dirname_creator=shorten_trial_dirname),
    # run_config=RunConfig(storage_path=folder_project,),
)

# Run the hyperparameter search
results = tuner.fit()

# Executando o tuning com Ray Tune
# results = tune.run(
#     train_yolo,
#     config=space,
#     metric="mAP50",
#     mode="max",
#     storage_path=folder_project,  # salva os resultados na pasta
#     trial_dirname_creator=shorten_trial_dirname,
# )

logging.basicConfig(      # Configuração do logger
    level=logging.INFO,  # Nível mínimo de mensagens para registrar
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="tuning.log",  # Arquivo onde as mensagens serão salvas
    filemode="w",  # Sobrescreve o arquivo a cada execução
)

best_trial = results.get_best_trial(metric="loss", mode="min")
print("Melhor trial:", best_trial)
print("Diretório de logs do trial:", best_trial.logdir)

logging.info(f"\nBest actual trial : {best_trial} on dir {best_trial.logdir}")