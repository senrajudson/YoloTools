from ultralytics import YOLO
import logging
import json

### YOLO é gambiarra e eu posso provar:
""" Para treinamentos de classificação com YOLO, você deve indicar o dir com o dataset
que deve estar especificado dentro de uma pasta chamada 'datasets'. No entando, para detecção
o YOLO é diferente. Você precisa indicar o caminho do arquivo 'dataset.yaml' para que ele 
possa encontrar o dataset e realizar o treinamento. É a mesma função, de uma mesma lib,
mas os caras fizeram de forma que o mesmo argumento recebe duas entradas completamente 
diferentes a depender do treinamento que você vai fazer.
"""

# Configuração do logger (A chamada correta para instanciar as configurações)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",
    filemode="w",
)

# Variáveis globais para configuração
task = "detect"
range_of_search = 100

# Variáveis globais para rastrear a melhor métrica
best_mAP50 = 0.0
best_mAP5095 = 0.0
best_acc = 0.0
best_loss = 100.0
best_epoch = 0
best_metric = 0.0
limit = patience = range_of_search
current_metric_value = 0
current_metric_support = 0
last_epoch = 0

def on_train_epoch_end(trainer):
    global best_mAP50, best_epoch, best_acc, best_mAP5095, best_metric, best_loss, limit, patience, current_metric_value, current_metric_support, last_epoch

    if task == "detect":
        # Obter a métrica atual
        current_metric_value = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        current_metric_support = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)

        # Atualiza a melhor métrica se a atual for melhor
        if current_metric_value > best_mAP50 or current_metric_support > best_mAP5095:
            best_mAP50 = current_metric_value
            best_epoch = trainer.epoch
            # Correção: Adicionado 'f' antes da string para as variáveis funcionarem
            logging.info(f"Melhor mAP50 atual: {round(best_mAP50, 4)} na época {best_epoch}")
            limit = patience  # Reinicia a paciência
            
            # Nota: O trainer.model salva os pesos automaticamente no YOLO26,
            # mas se você realmente quiser forçar o salvamento manual do objeto de treinamento:
            # trainer.model.save("best_metric.pt") 

        print(trainer.metrics)
        # Correção: Adicionado 'f' antes da string
        print(f"mAP50 atual: {round(current_metric_value, 4)} | mAP5095 atual: {round(current_metric_support, 4)} | Época atual: {trainer.epoch}")
        print(f"Melhor até agora na época: {best_epoch}")

        best_metric = best_mAP50 if best_mAP50 > best_metric else best_metric

    if task == "classify":
        # Pegue a métrica de accuracy da classificação
        current_metric_value = trainer.metrics.get("metrics/accuracy_top1", 0.0)
        current_metric_support = trainer.metrics.get("val/loss", 0.0)

        if current_metric_value > best_acc or current_metric_support < best_loss:
            best_acc = current_metric_value
            best_loss = current_metric_support if current_metric_support != 0 else 100.0
            best_epoch = trainer.epoch
            # Correção: Adicionado 'f' antes da string
            logging.info(f"Melhor accuracy atual: {round(best_acc, 4)} na época {best_epoch}")
            limit = patience
            # trainer.model.save("best_metric.pt")

        print(trainer.metrics)
        # Correção: Adicionado 'f' antes da string
        print(f"Accuracy atual: {round(current_metric_value, 4)} | Loss atual: {round(current_metric_support, 4)} | Época atual: {trainer.epoch}")
        print(f"Melhor até agora na época: {best_epoch}")

        best_metric = best_acc if best_acc > best_metric else best_metric
        last_epoch = trainer.epoch

    limit -= 1
    if limit == 0:
        # Correção: Adicionado 'f' antes da string
        logging.warning(f"Patience atingido na época {trainer.epoch}")
        raise KeyboardInterrupt

    return current_metric_value


def training(model_name, params_file, dataset_path):

    import ultralytics.utils

    # 2. Force o valor que você quer (ex: 6 CPUs por trial)
    ultralytics.utils.NUM_THREADS = 12    

    # Instancia o modelo dentro da função
    model = YOLO(model_name if model_name else "yolo26n.pt") # Nota: yolo26n.pt não é padrão, usei 26n como fallback de exemplo.

    # CORREÇÃO PRINCIPAL: O erro do `open()` estava aqui. 
    # Estava faltando passar o modo de leitura "r" (read) corretamente.
    with open(params_file, "r") as file:
        config = json.load(file)

    # Iniciar o treinamento com os parâmetros do JSON
    model.train(
        data=dataset_path,
        device="cuda",
        cache="ram",
        patience=20,
        epochs=200,
        **config
    )

if __name__ == "__main__":
    # Caminhos relativos ao WORKDIR do Docker (/app)

    from YOLO_tools.scripts.overwrite_file import patch_metrics

    # Treino 1
    metrics = [0.0, 0.0, 1.0, 0.0] # [P, R, MAP50, MAP5095]
    patch_metrics(metrics)
    model_path = "yolo26s.pt"
    dataset = "datasets/emissoes_YOLO/dataset.yaml"
    params = "YOLO_tools/train/params4.json"
    training(model_path, params, dataset)

    # Treino 2
    # params = "YOLO_tools/train/params2.json"
    # training(model_path, params, dataset)

    # Treino 3
    metrics = [0.0, 0.0, 0.0, 1.0] # [P, R, MAP50, MAP5095]
    patch_metrics(metrics)
    model_path = "yolo26s.pt"
    dataset = "datasets/emissoes_YOLO/dataset.yaml"
    params = "YOLO_tools/train/params3.json"
    training(model_path, params, dataset)

    # # Treino 4
    # model_path = "yolo26n.pt"
    # params = "YOLO_tools/train/params.json"