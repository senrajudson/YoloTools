import os
from ultralytics import YOLO

#------------------------------------------------------------------------
# INICIE O SERVIDOR MLFLOW "mlflow server --host 127.0.0.1 --port 5000" |
#------------------------------------------------------------------------

# 1. Conecta ao servidor que você abriu no terminal
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

# 2. Nomeia o grupo de testes
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Otimizacao_YOLO_v3"

# 3. Garante que o Ultralytics use o MLflow (opcional, ele costuma auto-detectar)
os.environ["REPORT_TO"] = "mlflow"

model = YOLO("yolo26s.pt")

# Ao rodar isso, o Ray Tune enviará cada métrica para o link acima
model.tune(
    data="D:\Judson_projetos\YoloTools\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml",
    device = "cuda",
    iterations=100,
    use_ray=True,
    epochs=70,
    patience=10
)

#--------------------------------------------------------------------------------------
# CONFIGURAR POETRY PARA LOCAL VITUALVENV | poetry config virtualenvs.in-project true |
#--------------------------------------------------------------------------------------