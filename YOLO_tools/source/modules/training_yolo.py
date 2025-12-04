from ultralytics import YOLO
import json

class ModelObjetct():
    def __init__(self, model, datapath, config="", device="cuda", 
                 patience=10, callback=None):
        
        self.model: str = model
        self.datapath: str = datapath
        self.config: json = config
        self.device: str = device
        self.patience: int = patience
        self.callback: function = callback

def training(modelobjetct: ModelObjetct):
    """_summary_
    A mesma função funciona qualquer modelo, só precisa indicar corretamente o 
    dataset e o tipo do modelo
    """
    
    model = YOLO(modelobjetct.model)

    try:
        if modelobjetct.callback != None:
            model.add_callback('on_train_epoch_end', modelobjetct.callback)
    except Exception as e:
        print(f"Nenhuma função de callback detectada. {e}")

    # Iniciar o treinamento com os parâmetros do JSON
    model.train(
        data=modelobjetct.datapath,
        device=modelobjetct.device,
        patience=modelobjetct.device,
        # workers=0,
        **modelobjetct.config
    )

if __name__ == "__main__":

    obj = ModelObjetct()
    obj.model = (rf"yolo11n.pt")
    obj.datapath = (rf"//datapath.yaml") ## << isso aqui muda se for classificação

    training(obj)

    pass

""" Para treinamentos de classificação com YOLO, você deve indicar o dir com o dataset
que deve estar especificado dentro de uma pasta chamada 'datasets'. No entando, para detecção
o YOLO é diferente. Você precisa indicar o caminho do arquivo 'dataset.yaml' para que ele 
possa encontrar o dataset e realizar o treinamento. É a mesma função, de uma mesma lib,
mas os caras fizeram de forma que o mesmo argumento recebe duas entradas completamente 
diferentes a depender do treinamento que você vai fazer.
"""
""" Além disso, nunca se esqueça de modificar o modelo dependendo da task. Os modelos 
possuem nomes diferentes para cada tarefa, detecção, classificação ou segmentação.
Para mais informações, conferir a documentação oficial do Ultralytics.
"""