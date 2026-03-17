import os
from ultralytics import YOLO
import mlflow

#------------------------------------------------------------------------
# INICIE O SERVIDOR MLFLOW   mlflow server --host 127.0.0.1 --port 5312 --backend-store-uri ./mlruns |
# mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
#------------------------------------------------------------------------

BASE_DIR = "/app"
DATA_YAML = os.path.join(BASE_DIR, "datasets", "emissoes_YOLO", "dataset.yaml")

def tune(model_path, args):
    model = YOLO(os.path.join(BASE_DIR, model_path))

    import ultralytics.utils

    # 2. Force o valor que você quer (ex: 6 CPUs por trial)
    ultralytics.utils.NUM_THREADS = 6

    model.tune(
        data=DATA_YAML,
        device="cuda",
        cache="ram",
        use_ray=True,
        # gpu_per_trial=0.5, # trava o tuning # deixa o cuda administrar, aí funciona, o ray trava 
        **args
    )

if __name__ == "__main__":

    # # Conecta ao servidor que você abriu no terminal
    # os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

    # # Garante que o Ultralytics use o MLflow (opcional, ele costuma auto-detectar)
    # os.environ["REPORT_TO"] = "mlflow"

    args = {
        "iterations": 100,
        "epochs": 70,
        "patience": 10
    }

    # Nomeia o grupo de testes
    mlflow.set_tracking_uri("file:///app/mlruns")

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Otimizacao_YOLO_v12"
    tune("yolo26s.pt", args)

    # Nomeia o grupo de testes
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Otimizacao_YOLO_v13"
    tune("yolo26n.pt", args)

#--------------------------------------------------------------------------------------
# CONFIGURAR POETRY PARA LOCAL VITUALVENV | poetry config virtualenvs.in-project true |
#--------------------------------------------------------------------------------------