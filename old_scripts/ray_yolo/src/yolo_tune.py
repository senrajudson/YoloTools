import os
import ray
import json
from ray import tune
from ultralytics import YOLO
from ray.tune.schedulers import ASHAScheduler

model_ref = ray.put(YOLO("yolo11s.pt"))

name_exp = 'yolo_tunning_253007012'

# === Espaço de busca para hyperparametros ===
search_space = {
    # Parâmetros de execução (não vão pro hyp.yaml)
    "data": "/workspace/datasets/dataset.yaml",
    "device": "cpu",
    "cache": False,
    # "workers": 0,

    # Parâmetros que vão para hyp.yaml
    "imgsz": tune.choice[360, 480, 640],
    "batch": tune.choice[8, 12, 16, 32],
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(0.1, 1.0),
    "momentum": tune.uniform(0.6, 0.98),
    "weight_decay": tune.uniform(0.0001, 0.01),
    "warmup_epochs": tune.randint(0, 5),
    "warmup_momentum": tune.uniform(0.0, 0.95),
    "warmup_bias_lr": tune.uniform(0.0, 0.2),
    "box": tune.uniform(0.02, 0.2),
    "cls": tune.uniform(0.2, 4.0),
    "kobj": tune.uniform(0.2, 4.0),
}

def train_yolo(config):
    model = ray.get(model_ref)
    # configuração do hyp.yaml e training...
    results = model.train(**config)
    mAP = results.metrics.get("metrics/mAP_0.5:0.95", 0.0) or 0.0
    tune.report(mAP5095=mAP)

def run_tuning():
    scheduler = ASHAScheduler(metric="mAP5095", mode="max", max_t=30,
                              grace_period=3, reduction_factor=2)

    analysis = tune.run(
        train_yolo,
        config=search_space,
        num_samples=200,
        scheduler=scheduler,
        name="exp_yolo",
        max_concurrent_trials=1,
        reuse_actors=False,
        storage_path=f"file://{os.getcwd()}/ray_logs"
    )

    best = analysis.get_best_config(metric="mAP5095", mode="max")
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("✅ Parâmetros salvos.")

if __name__ == "__main__":
    run_tuning()