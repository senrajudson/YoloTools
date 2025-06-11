if __name__ == "__main__":
    from YOLO_tools.old_scripts.yolo_tools import YoloTrainer, YoloTuner
    from ray import tune

    # Define os caminhos do projeto
    config_path = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\params.json"
    model_path = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\yolo11n.pt"
    dataset_yaml = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml"
    storage_path = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions"

    trainer = YoloTrainer(
        config_path=config_path,
        model_path=model_path,
        dataset_yaml=dataset_yaml,
    )

    # trainer.train()

    # Espaço de busca dos hiperparâmetros
    hyper_space = {
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(1e-5, 1e-2),
        "weight_decay": tune.uniform(1e-3, 1e-2),
        "momentum": tune.uniform(0.8, 0.95),
        "warmup_epochs": tune.randint(1, 5),
        "warmup_momentum": tune.uniform(0.4, 0.8),
        "warmup_bias_lr": tune.uniform(1e-5, 1e-1),
        "optimizer": tune.choice(["AdamW", "SGD"]),
        "imgsz": tune.choice([360, 480, 640]),
        "batch": tune.choice([8, 16, 32, 48, 64]),
        # 'epochs': 100,
    }

    # Instancia e executa o tuner
    tuner_instance = YoloTuner(
        config_path=config_path,
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        storage_path=storage_path,
        hyper_space=hyper_space,
        include_dashboard=True,
    )

    tuner_instance.run()
