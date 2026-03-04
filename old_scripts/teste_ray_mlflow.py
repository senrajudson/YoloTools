from ultralytics import YOLO
from ray import tune
from ray.air import RunConfig
from ray.tune.tuner import Tuner
from ray.air.integrations.mlflow import MLflowLoggerCallback

def train_yolo(config):
    model = YOLO(config["weights"])

    model.train(
        data=config["data"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        epochs=config["epochs"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        warmup_epochs=config["warmup_epochs"],
        augment=config["augment"],
        # importante: não force mlflow interno aqui; deixe o Ray logar
        verbose=False,
    )
    # métricas por época já são reportadas ao Ray quando em sessão Ray
    # via ultralytics.utils.callbacks.raytune.on_fit_epoch_end :contentReference[oaicite:5]{index=5}

search_space = {
    "weights": "yolov8n.pt",
    "data": "coco8.yaml",
    "epochs": 30,

    "imgsz": tune.choice([640, 800]),
    "batch": tune.choice([16, 32]),
    "lr0": tune.loguniform(1e-4, 5e-2),
    "lrf": tune.uniform(0.01, 0.2),
    "momentum": tune.uniform(0.85, 0.98),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "warmup_epochs": tune.uniform(0.0, 5.0),
    "augment": tune.choice([True, False]),
}

experiment_name = "yolo_detect_coco8_20260303"

tuner = Tuner(
    tune.with_resources(train_yolo, resources={"cpu": 4, "gpu": 1}),
    param_space=search_space,
    run_config=RunConfig(
        name=experiment_name,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="http://SEU_MLFLOW:5000",
                experiment_name=experiment_name,
            )
        ],
    ),
    tune_config=tune.TuneConfig(
        metric="metrics/mAP50-95(B)",  # ajuste pro nome exato que seu YOLO reportar
        mode="max",
        num_samples=20,
    ),
)

results = tuner.fit()
print("Melhor trial:", results.get_best_result().config)