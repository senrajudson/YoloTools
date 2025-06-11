from ultralytics.utils import DEFAULT_CFG_DICT, NUM_THREADS
from ray.tune.schedulers import ASHAScheduler
from ultralytics.cfg import TASK2METRIC
from ray.air import RunConfig
from ultralytics import YOLO
from ray import tune
import logging
import hashlib
import json
import ray

### this are a mix of all YOLO built-in augments, if ur implementing manual augments, it's ideal to disable YOLO augments to avoid overlay
from ultralytics.data.augment import (
    Albumentations,
    CenterCrop,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)

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

### YOLO é gambiarra e eu posso provar:
""" Para treinamentos de classificação com YOLO, você deve indicar o dir com o dataset
que deve estar especificado dentro de uma pasta chamada 'datasets'. No entando, para detecção
o YOLO é diferente. Você precisa indicar o caminho do arquivo 'dataset.yaml' para que ele 
possa encontrar o dataset e realizar o treinamento. É a mesma função, de uma mesma lib,
mas os caras fizeram de forma que o mesmo argumento recebe duas entradas completamente 
diferentes a depender do treinamento que você vai fazer.
"""


class YoloTrainer:
    def __init__(
        self,
        config_path,
        model_path,
        dataset_yaml,
        log_file="training.log",
        patience=50,
    ):
        """
        Inicializa o YoloTrainer carregando as configurações, instanciando o modelo e configurando o callback.

        Args:
            config_path (str): Caminho para o arquivo JSON de parâmetros.
            model_path (str): Caminho para o arquivo de pesos do modelo.
            dataset_yaml (str): Caminho para o arquivo dataset.yaml.
            log_file (str, opcional): Nome do arquivo de log. Padrão 'training.log'.
            patience (int, opcional): Número de épocas sem melhoria antes de interromper o treinamento.
        """

        self.config_path = config_path
        self.model_path = model_path
        self.dataset_yaml = dataset_yaml

        # Carrega a configuração a partir do arquivo JSON
        with open(self.config_path, "r") as file:
            self.config = json.load(file)

        # Instancia o modelo YOLO
        self.model = YOLO(model_path)

        # Inicializa as métricas e variáveis de controle
        self.best_recall = 0.0
        self.best_precision = 0.0
        self.best_f1score = 0.0
        self.best_mAP50 = 0.0
        self.best_mAP5095 = 0.0
        self.best_epoch = 0
        self.patience = patience
        self.limit = patience

        # Configuração do logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode="w",
        )

        # Registra o callback de fim de época
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)

    def on_train_epoch_end(self, trainer):
        """
        Callback executado ao final de cada época de treinamento.
        Atualiza as métricas, salva o modelo se necessário e reporta métricas para o Ray Tune.

        Args:
            trainer: Objeto que contém as métricas e o número da época atual.
        """
        # Obter as métricas atuais
        current_mAP50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
        current_mAP5095 = trainer.metrics.get("metrics/mAP50-95(B)", 0.0)

        # Se houver melhora no mAP50, atualiza a melhor métrica e salva o modelo
        if current_mAP50 > self.best_mAP50:
            self.best_mAP50 = current_mAP50
            self.best_epoch = trainer.epoch
            logging.info(
                f"Best actual metric: {round(self.best_mAP50, 4)} on epoch {self.best_epoch}"
            )
            self.limit = self.patience  # Reseta a paciência
            self.model.save("best_metric.pt")

        print(trainer.metrics)
        print(f"Actual mAP50: {round(current_mAP50, 4)}")
        print(
            f"Best actual metric: {round(self.best_mAP50, 4)} on epoch {self.best_epoch}"
        )

        # Reporta as métricas para o Ray Tune
        tune.report(mAP50=current_mAP50, mAP5095=current_mAP5095, epoch=trainer.epoch)

        # Atualiza o contador de paciência
        self.limit -= 1
        if self.limit == 0:
            logging.warning(f"Patience has reached limit at epoch {trainer.epoch}")
            raise KeyboardInterrupt

    def train(self):
        """
        Inicia o treinamento do modelo YOLO utilizando os parâmetros configurados.
        """
        self.model.train(
            data=self.dataset_yaml,
            device="cuda",
            batch=self.config["batch"],
            epochs=300,  # Você pode alterar este valor se desejar utilizar um parâmetro do arquivo de configuração
            imgsz=self.config["imgsz"],
            lr0=self.config["lr0"],
            lrf=self.config["lrf"],
            momentum=self.config["momentum"],
            optimizer=self.config["optimizer"],
            warmup_bias_lr=self.config["warmup_bias_lr"],
            warmup_epochs=self.config["warmup_epochs"],
            warmup_momentum=self.config["warmup_momentum"],
            weight_decay=self.config["weight_decay"],
        )


class YoloTuner(YoloTrainer):
    def __init__(
        self,
        config_path,
        model_path,
        dataset_yaml,
        storage_path,
        hyper_space,
        include_dashboard=True,
    ):
        """
        Inicializa o YoloTuner com os caminhos, espaço de busca dos hiperparâmetros e configura o Ray.

        Args:
            config_path (str): Caminho para o arquivo JSON de configuração (params.json).
            model_path (str): Caminho para o arquivo de pesos do modelo.
            dataset_yaml (str): Caminho para o arquivo dataset.yaml.
            storage_path (str): Pasta onde os resultados do Ray Tune serão armazenados.
            hyper_space (dict): Espaço de busca dos hiperparâmetros.
            include_dashboard (bool, opcional): Se True, inicia o dashboard do Ray. Padrão True.
        """

        super().__init__(config_path, model_path, dataset_yaml)

        self.storage_path = storage_path
        self.hyper_space = hyper_space
        self.gpu_per_trial = 1
        self.model_in_store = None

        # Inicializa o Ray com o dashboard e define o diretório temporário
        # ray.init(include_dashboard=include_dashboard, _temp_dir=storage_path)
        ray.init(
            runtime_env={"py_modules": ["yolo_tools"]},
            include_dashboard=include_dashboard,
            _temp_dir=storage_path,
        )

        # Configuração básica do logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="tuning.log",
            filemode="w",
        )

    @staticmethod
    def shorten_trial_dirname(trial):
        """
        Gera um nome curto e único para o diretório do trial usando um hash do trial_id.
        """
        return hashlib.md5(trial.trial_id.encode()).hexdigest()[:8]

    def _train_yolo(self, space):
        """
        Função objetivo para o Ray Tune.
        Atualiza o arquivo de configuração com os hiperparâmetros do trial e inicia o treinamento.
        """
        # Atualiza o arquivo de parâmetros com os hiperparâmetros atuais do trial
        with open(self.config_path, "w") as file:
            json.dump(space, file, indent=4)

        # Importa e instancia o YoloTrainer (certifique-se de que a classe YoloTrainer esteja disponível)

        trainer = YoloTrainer(
            config_path=self.config_path,
            model_path=self.model_path,
            dataset_yaml=self.dataset_yaml,
        )
        trainer.train()

    def run(self):
        """
        Executa o tuning com o Ray Tune e retorna o objeto de análise.
        Também imprime e registra o melhor trial encontrado.
        """

        task = self.model.task
        self.model_in_store = ray.put(self.model)

        asha_scheduler = ASHAScheduler(
            time_attr="epoch",
            metric=TASK2METRIC[task],
            mode="max",
            max_t=self.hyper_space.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
            grace_period=10,
            reduction_factor=3,
        )

        trainable_with_resources = tune.with_resources(
            self._train_yolo, {"cpu": NUM_THREADS, "gpu": self.gpu_per_trial or 0}
        )

        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=self.hyper_space,
            tune_config=tune.TuneConfig(
                scheduler=asha_scheduler,
                num_samples=10,
                trial_dirname_creator=YoloTuner.shorten_trial_dirname,
            ),
            run_config=RunConfig(storage_path=self.storage_path),
        )

        results = tuner.fit()

        # tuner = tune.run(
        #     self._train_yolo,
        #     config=self.hyper_space,
        #     metric="mAP50",
        #     mode="max",
        #     # storage_path=f"file://{self.storage_path}",
        #     storage_path=self.storage_path,
        #     trial_dirname_creator=YoloTuner.shorten_trial_dirname,
        # )

        best_trial = results.get_best_trial(metric="loss", mode="min")
        logging.info(f"\nBest actual trial: {best_trial} on dir {best_trial.logdir}")
        print("Melhor trial:", best_trial)
        print("Diretório de logs do trial:", best_trial.logdir)
        return results
