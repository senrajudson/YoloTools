from ultralytics import YOLO
from ray import tune
import ray

# Inicialize o Ray com o dashboard habilitado
ray.init(
    include_dashboard=True,
    _temp_dir=r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions",
)

# Inicializa o modelo e o dataset
model = YOLO(r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\yolo11n.pt")
data_yaml = (
    r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml"
)

# Defina o espaço de busca (hyperparâmetros) para o Tune
space = {
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(1e-5, 1e-2),
    "weight_decay": tune.uniform(1e-3, 1e-2),
    "momentum": tune.uniform(0.8, 0.95),
    "warmup_epochs": tune.randint(1, 5),
    "warmup_momentum": tune.uniform(0.4, 0.8),
    "warmup_bias_lr": tune.uniform(1e-5, 1e-1),
    "epochs": 100,
    "optimizer": tune.choice(["AdamW", "SGD"]),
    # "imgsz": tune.choice([360, 480, 640]),
    "imgsz": tune.choice([360, 480]),
    "batch": tune.randint(8, 48),
}

# Execute o treinamento (tune integrado ao YOLO)
results = model.tune(
    data=data_yaml,
    use_ray=True,
    iterations=100,
    space=space,
    cache="ram",
    workers=4,
    # train_args={
    #   'cache' : 'ram',
    #   'workers' : 2,
    #   }
    # gpu_per_trial=1,
    # num_threads = 2,
    # max_samples = 2,
)

"""
.\prometheus-3.2.1.windows-amd64\prometheus.exe --config.file=D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions\session_2025-03-19_16-47-39_139098_18500\metrics\prometheus\prometheus.yml

"""
