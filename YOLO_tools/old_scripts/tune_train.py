import os
import yaml
import glob
from ultralytics import YOLO
from datetime import datetime
from ray import tune

"""
lembre-se sempre = precisa trocar o modelo a depender da task
troque o dataset para o treinamento desejado
"""

def tune_yolo11(
    model_path: str,
    data_yaml: str,
    epochs: int = 20,
    max_samples: int = 10,
    gpu_per_trial: int = 1,
    custom_space: dict = None,
    grace_period: int = 5,
    output_yaml: str = "melhores_parametros.yaml",
    resume: bool = False,
    name: str = "yolo_tunn_exp"
):
    """
    Executa otimização com Ray Tune na API do ultralytics para YOLOv11.
    """
    model = YOLO(model_path)

    tune_args = {
        "data": data_yaml,
        "epochs": epochs,
        "gpu_per_trial": gpu_per_trial,
        "iterations": max_samples,
        "grace_period": grace_period,
        "resume": resume,
        "name": name
    }

    if custom_space:
        tune_args["space"] = custom_space

    # Ativa Ray Tune internamente
    result_grid = model.tune(use_ray=True, **tune_args)
    best = result_grid.get_best_result(metric="metrics/mAP50(B)", mode="max")
    hyperparams = best.config
    metrics_last = best.metrics
    metrics_history = best.metrics_dataframe

    saida_completa = {**hyperparams, **metrics_last}
    print("✨ Melhores hiperparâmetros e métricas finais:")
    print(saida_completa)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    metrics_history.to_csv(f"melhores_metricas_{timestamp}.csv", index=False)

    # Salva em YAML
    output_yaml = f"melhores_parametros_{timestamp}.yaml"
    with open(output_yaml, "w") as f:
        yaml.dump(saida_completa, f)

    return best

def train_yolo11(
    model_path: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device=None,         # GPUs ou CPU
    lr0: float = 0.01,
    lrf: float = 0.2,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    box: float = 0.05,
    cls: float = 0.5,
    kobj: float = 1.0,
    hsv_h: float = 0.01,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.0,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
):
    """Treina YOLOv11 usando todos os parâmetros possíveis."""

    model = YOLO(model_path)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        box=box,
        cls=cls,
        kobj=kobj,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    print("✅ Treino finalizado, métricas:")
    print("mAP50:", results.box.map50)
    print("mAP50-95:", results.box.map)
    print("Precision (mp):", results.box.mp)
    print("Recall (mr):", results.box.mr)         # se disponível

    # Ou imprima tudo em forma de dicionário
    print("Todas as métricas:", results.results_dict)

    return results

def escolher_melhor_parametro(pasta=".", padrao="melhores_parametros_*.yaml"):
    melhores_arquivos = glob.glob(os.path.join(pasta, padrao))

    melhor_arquivo = None
    melhor_score = -1
    melhor_cfg = None

    for arquivo in melhores_arquivos:
        try:
            with open(arquivo, "r") as f:
                cfg = yaml.safe_load(f)

            score = cfg.get("metrics/mAP50-95(B)", -1)
            if score > melhor_score:
                melhor_score = score
                melhor_arquivo = arquivo
                melhor_cfg = cfg
        except Exception as e:
            print(f"❌ Erro ao ler {arquivo}: {e}")

    if melhor_cfg:
        print(f"✅ Melhor arquivo: {melhor_arquivo} com score {melhor_score}")
    else:
        raise ValueError("Nenhum arquivo válido encontrado.")

    return melhor_cfg

if __name__ == "__main__":

    space = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.98),
        "weight_decay": tune.uniform(0.0, 1e-3),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "box": tune.uniform(0.02, 0.2),
        "cls": tune.uniform(0.2, 4.0),
        "kobj": tune.uniform(0.2, 4.0),
        "hsv_h": tune.uniform(0.0, 0.1),
        "degrees": tune.uniform(0.0, 45.0),
        "translate": tune.uniform(0.0, 0.9),
        "scale": tune.uniform(0.0, 0.9),
        "shear": tune.uniform(0.0, 10.0),
        "perspective": tune.uniform(0.0, 0.001),
        "flipud": tune.uniform(0.0, 1.0),
        "fliplr": tune.uniform(0.0, 1.0),
        "mosaic": tune.uniform(0.0, 1.0),
        "mixup": tune.uniform(0.0, 1.0),
        "copy_paste": tune.uniform(0.0, 1.0),
    }

    dataset = "D:/Judson_projetos/Yolo_trainer/YOLO_tools/datasets/emissoes_completo_yolo_1607/dataset.yaml"

    tune_yolo11(
        model_path="yolo11n.pt",
        data_yaml=dataset,
        epochs=50,
        max_samples=100,
        gpu_per_trial=1,
        custom_space=space,
        grace_period=5,
        resume=True,
        name='exp_000001'
    )

    # Escolher melhor config
    cfg = escolher_melhor_parametro()

    train_yolo11(
        model_path="yolo11n.pt",
        data_yaml=dataset,
        epochs=cfg.get("epochs", 50),
        imgsz=cfg.get("img_size", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", None),

        # parâmetros de otimização
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.2),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        warmup_momentum=cfg.get("warmup_momentum", 0.8),

        # taxas de perda
        box=cfg.get("box", 0.05),
        cls=cfg.get("cls", 0.5),
        kobj=cfg.get("kobj", 1.0),

        # parâmetros de augmentação
        hsv_h=cfg.get("hsv_h", 0.01),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 0.0),
        translate=cfg.get("translate", 0.0),
        scale=cfg.get("scale", 0.5),
        shear=cfg.get("shear", 0.0),
        perspective=cfg.get("perspective", 0.0),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.0),
        copy_paste=cfg.get("copy_paste", 0.0),
    )