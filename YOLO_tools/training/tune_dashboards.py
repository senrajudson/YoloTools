from ultralytics.utils.tuner import run_ray_tune  # ou o nome correto da função de treino interna
from ray import tune

experiment_path = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions\session_2025-01-23_11-32-44_550677_25496"
print(f"Carregando resultados de {experiment_path}...")

restored_tuner = tune.Tuner.restore(experiment_path, trainable=run_ray_tune)
result_grid = restored_tuner.get_results()
