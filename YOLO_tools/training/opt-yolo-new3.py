from ray.tune.schedulers import ASHAScheduler
# from ray.air import RunConfig
from ultralytics import YOLO
from ray import tune
import logging
import hashlib
import json
import ray
import os

# from ray.air import session
# from ray.train._internal import session ## can get ID for the trial and report metrics to ray
from ray.tune.context import get_context

model = YOLO(r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\yolo11n.pt")

ray.init(include_dashboard=True, _temp_dir=r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\ray_sessions")

model_in_store = ray.put(model)
model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning

### this are a mix of all YOLO built-in augments, if ur implementing manual augments, it's ideal to disable YOLO augments to avoid overlay
from ultralytics.data.augment import Albumentations, CenterCrop, RandomFlip, RandomHSV, RandomPerspective

"""
eu modifiquei as transformações dentro da classe 'Albumentations' no '.../ultralytics/data/augment, 
foi necessário zerar a probabilidade usando um float 0.0 no transform 'T' e
mudando o valor de 'p' em __init__ para 'p=0' 
"""

albumentations_yolo = Albumentations(p=0.0)
centercrop_yolo = CenterCrop(0)
randomflip_yolo = RandomFlip(p=0.0)
randomhsv_yolo = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.0)
randomperspective_yolo = RandomPerspective(translate=0.0, scale=0.0)

"""
também modifiquei o tune.Config em '.../ultralytics/utils/tuner'
adicionei um 'trial_dirname_creator' com uma função de 'shorten_name_creator' usando 
a lib hashlib, pois os nomes que o raytune gera automaticamente são muito grandes para 
salvar no dir do windows.
"""

"""
também modifiquei a função on_fit_epoch_end em '.../ultralytics/utils/callbacks/raytune'
eu fiz o import da função 'from ray.train._internal.session import get_session' e 
substitui na condicional 'ray.train._internal.session.get_session()'
"""

### YOLO é gambiarra e eu posso provar:
""" Para treinamentos de classificação com YOLO, você deve indicar o dir com o dataset
que deve estar especificado dentro de uma pasta chamada 'datasets'. No entando, para detecção
o YOLO é diferente. Você precisa indicar o caminho do arquivo 'dataset.yaml' para que ele 
possa encontrar o dataset e realizar o treinamento. É a mesma função, de uma mesma lib,
mas os caras fizeram de forma que o mesmo argumento recebe duas entradas completamente 
diferentes a depender do treinamento que você vai fazer.
"""

# # Carregar configurações de um arquivo 
# with open('../hyper_yolo.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# train_config = config['train']
# aug_config = config['train']['augmentation']

# with open(r'D:\Judson_projetos\Yolo_trainer\YOLO_tools\training\params.json', 'r') as file:   # Carregar configurações de um arquivo
#     config = json.load(file)

PARAMS_FILENAME = "params_trials.json"

def update_params_file(trial_id, config):
    global PARAMS_FILENAME

    # Carrega os dados existentes, se houver
    if os.path.exists(PARAMS_FILENAME):
        with open(PARAMS_FILENAME, "r") as file:
            all_params = json.load(file)
    else:
        all_params = {}
    
    # Atualiza ou adiciona os parâmetros do trial
    all_params[trial_id] = config
    
    # Escreve o JSON atualizado
    with open(PARAMS_FILENAME, "w") as file:
        json.dump(all_params, file, indent=4)

best_recall = 0.0   # Variáveis globais para rastrear o melhor recall e a época correspondente
best_precision = 0.0
best_f1score = 0.0
best_mAP50 = 0.0
best_mAP5095 = 0.0
best_epoch = 0
patience = 100
limit = patience

logging.basicConfig(      # Configuração do logger
    level=logging.INFO,  # Nível mínimo de mensagens para registrar
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="opt.log",  # Arquivo onde as mensagens serão salvas
    filemode="w",  # Sobrescreve o arquivo a cada execução
)

def on_train_epoch_end(trainer):
    global best_recall, best_precision, best_f1score, best_mAP5095, best_mAP50, best_epoch, limit

    # current_recall = trainer.metrics.get('metrics/recall(B)', 0.0)       # Obtenha o recall atual dos resultados de validação
    # current_precision = trainer.metrics.get('metrics/precision(B)', 0.0)    # Obtenha o precision atual dos resultados de validação
    # current_f1score = 2 * (current_precision * current_recall) / (current_precision + current_recall) if current_recall > 0 else 0.0      # f1score

    current_mAP50 = trainer.metrics.get('metrics/mAP50(B)', 0.0)       # mAP50 ajuda no melhor 'recall'
    current_mAP5095 = trainer.metrics.get('metrics/mAP50-95(B)', 0.0)       # mAP50-95 ajuda no melhor 'precision'

    if current_mAP50 > best_mAP50:        # Verifique se o recall atual é melhor que o melhor recall registrado
        best_mAP50 = current_mAP50
        best_epoch = trainer.epoch

        logging.info(f"\nBest actual metric : {round(best_mAP50, 4)} on epoch {best_epoch}")
        limit = patience

        model_to_train.save(f'best_metric.pt')            # Salve os pesos do modelo para a melhor época com base no recall

    print(trainer.metrics)
    print(f"\nActual mAP50 : {round(current_mAP50, 4)}")
    print(f"\nBest actual metric : {round(best_mAP50, 4)} on epoch {best_epoch}")

    tune.report({"mAP50":current_mAP50, "mAP5095":current_mAP5095, "epoch":trainer.epoch})

    limit -= 1

    if limit == 0 :
        logging.warning(f"Patience has reached limit at epoch {trainer.epoch}")
        # logging.error("Erro inesperado no treinamento")
        
        raise KeyboardInterrupt
    
    return current_mAP50

def training(config):

    context = get_context()
    trial_id = context.get_trial_id()
    update_params_file(trial_id, config)

    # model_to_train.reset_callbacks()
    model_to_train.add_callback('on_train_epoch_end', on_train_epoch_end)    # Adicione o callback personalizado ao modelo
    model_to_train.train(
        data = r"D:\Judson_projetos\Yolo_trainer\YOLO_tools\datasets\emissoes_YOLO\dataset.yaml",
        device = "cuda",

        batch = config['batch'],    ### training configs
        # epochs = config['epochs'],
        epochs = 300,
        imgsz = config['imgsz'],

        lr0 = config['lr0'],
        lrf = config['lrf'],
        momentum = config['momentum'],
        optimizer = config['optimizer'],

        warmup_bias_lr = config['warmup_bias_lr'],
        warmup_epochs = config['warmup_epochs'],
        warmup_momentum = config['warmup_momentum'],
        weight_decay = config['weight_decay'],
    )

    print('aqui, \nestou printando alguma coisa aqui, \nsó para ter certeza de que chegou até aqui')

# Define the trainable function with allocated resources
trainable_with_resources = tune.with_resources(training, {"cpu": 8, "gpu": 1})
# trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": 1})

# Defina o espaço de busca (hyperparâmetros) para o Tune
space = {
    "lr0": tune.uniform(1e-5, 1e-1),
    "lrf": tune.uniform(1e-5, 1e-2),
    "weight_decay": tune.uniform(1e-3, 1e-2),
    "momentum": tune.uniform(0.8, 0.95),
    "warmup_epochs": tune.randint(1, 5),
    "warmup_momentum": tune.uniform(0.4, 0.8),
    "warmup_bias_lr": tune.uniform(1e-5, 1e-1),
    "epochs": 70,
    "optimizer": tune.choice(['AdamW', "SGD"]),
    "imgsz": tune.choice([360, 480, 640]),
    "batch": tune.randint(8, 48),
}

# Define the ASHA scheduler for hyperparameter search
asha_scheduler = ASHAScheduler(
    time_attr="epoch",
    metric="mAP50",
    mode="max",
    max_t= 100,
    grace_period=10,
    reduction_factor=3,
)

def shorten_trial_dirname(trial):  # Ajustando para receber o objeto trial
    return hashlib.md5(trial.trial_id.encode()).hexdigest()[:8]    # Gerar um hash curto do trial_id para garantir que o nome do diretório seja único e curto

tuner = tune.Tuner(
    trainable_with_resources,
    param_space=space,
    tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=10, trial_dirname_creator=shorten_trial_dirname),
    # run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),
)

# Run the hyperparameter search
tuner.fit()

# Get the results of the hyperparameter search
results = tuner.get_results()

print(results)