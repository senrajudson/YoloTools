import os
from ultralytics import YOLO

#------------------------------------------------------------------------
# INICIE O SERVIDOR MLFLOW "mlflow server --host 127.0.0.1 --port 5000" |
#------------------------------------------------------------------------

def tune(model, args):
    model = YOLO(model)

    # Ao rodar isso, o Ray Tune enviará cada métrica para o link acima
    model.tune(
        data="D:\Judson_projetos\YoloTools\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml",
        device = "cuda",
        use_ray=True,
        **args
    )

    return

if __name__ == "__main__":


    # Conecta ao servidor que você abriu no terminal
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

    # Garante que o Ultralytics use o MLflow (opcional, ele costuma auto-detectar)
    os.environ["REPORT_TO"] = "mlflow"

    args = {
        "iterations": 100,
        "epochs": 70,
        "patience": 10
    }

    # Nomeia o grupo de testes
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Otimizacao_YOLO_v6"
    tune("yolo26n.pt", args)

    # Nomeia o grupo de testes
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Otimizacao_YOLO_v7"
    tune("yolo26s.pt", args)

#--------------------------------------------------------------------------------------
# CONFIGURAR POETRY PARA LOCAL VITUALVENV | poetry config virtualenvs.in-project true |
#--------------------------------------------------------------------------------------