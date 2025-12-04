"""
    exemplos de funções callback
"""
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",
    filemode="w",
)

# Variáveis globais para rastrear a melhor métrica
best_metric = 0.0
best_epoch = 0
limit = patience = 10
current_metric_value = 0
last_epoch = 0

# # callback para detecção usando métrica "mAP50(B)"
def on_train_epoch_end(trainer):
    global best_metric, best_epoch, limit, patience, current_metric_value, last_epoch

    current_metric_value = trainer.metrics.get("metrics/mAP50(B)", 0.0)

    # existem outras métricas
    # current_metric_value = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)

    # Atualiza a melhor métrica se a atual for melhor
    if current_metric_value > best_metric:
        best_metric = current_metric_value
        best_epoch = trainer.epoch
        limit = patience  # Reinicia a paciência

    last_epoch = trainer.epoch

    print(trainer.metrics)
    print(
        f"mAP50 atual: {round(current_metric_value, 4)} | Época atual: {trainer.epoch}"
    )

    print(f"Melhor até agora na época: {best_epoch}")

    limit -= 1
    if limit == 0:
        logging.warning(f"Patience atingido na época {trainer.epoch}")
        raise KeyboardInterrupt

    return current_metric_value


# # callback para classificação usando métrica "accuracy_top1"
# def on_train_epoch_end(trainer):
#     global best_metric, best_epoch, limit, patience, current_metric_value, last_epoch

#     # Pegue a métrica de accuracy da classificação
#     current_metric_value = trainer.metrics.get("metrics/accuracy_top1", 0.0)

#     # existem outras métricas
#     # current_metric_value = trainer.metrics.get("val/loss", 0.0)
#     # ou experimente "metrics/acc(B)"

#     if current_metric_value > best_metric:
#         best_metric = current_metric_value
#         best_epoch = trainer.epoch
#         limit = patience

#     last_epoch = trainer.epoch

#     print(trainer.metrics)
#     print(
#         f"Accuracy atual: {round(current_metric_value, 4)} | Época atual: {trainer.epoch}"
#     )
#     print(f"Melhor até agora na época: {best_epoch}")

#     limit -= 1
#     if limit == 0:
#         logging.warning(f"Patience atingido na época {trainer.epoch}")
#         raise KeyboardInterrupt

#     return current_metric_value