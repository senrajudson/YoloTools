from source.modules.slicing_dataset import slicing_dataset_for_traning
from source.modules.training_YOLO_model import training_YOLO_model
from source.modules.model_predict import predict_YOLO_model

class YOLOTrainer:

    def __init__(self):
        self.image_folder = None
        self.annotations_folder = None
        self.test_percentual_divisor = None
        self.predict_YOLO = None
        self.yolo_Classes = None
        self.object_to_predict = None
        self.predict_confidence = None
        self.train_model = None
        self.img_sz = None
        self.training_epochs = None
        self.task = None

    def slicing(self):
        slicing_dataset_for_traning(self.image_folder, self.annotations_folder, self.yolo_Classes, self.test_percentual_divisor)

    def training(self):
        training_YOLO_model(self.img_sz, self.training_epochs, self.task)

    def predict(self):
        predict_YOLO_model(self.train_model, self.object_to_predict, self.predict_confidence)

    @property
    def task(self):
        return self._task
    
    @task.setter
    def task(self, option):
        self._task = option

    @property
    def img_sz(self):
        return self._img_sz
    
    @img_sz.setter
    def img_sz(self, img):
        self._img_sz = img

    @property
    def training_epochs(self):
        return self._training_epochs
    
    @training_epochs.setter
    def training_epochs(self, epochs):
        self._training_epochs = epochs

    @property
    def object_to_predict(self):
        return self._object_to_predict
    
    @object_to_predict.setter
    def object_to_predict(self, object):
        self._object_to_predict = object

    @property
    def train_model(self):
        return self._train_model
    
    @train_model.setter
    def train_model(self, model):
        self._train_model = model

    @property
    def predict_confidence(self):
        return self._predict_confidence
    
    @predict_confidence.setter
    def predict_confidence(self, confidence):
        self._predict_confidence = confidence

    @property
    def yolo_Classes(self):
        return self._yolo_Classes
    
    @yolo_Classes.setter
    def yolo_Classes(self, cls):
        self._yolo_Classes = cls

    @property
    def predict_YOLO(self):
        return self._predict_YOLO
    
    @predict_YOLO.setter
    def predict_YOLO(self, predict_object):
        self._predict_YOLO = predict_object

    @property
    def annotations_folder(self):
        return self._annotations_folder
    
    @annotations_folder.setter
    def annotations_folder(self, path):
        self._annotations_folder = (path, "labels")

    @property
    def image_folder(self):
        return self._image_folder
    
    @image_folder.setter
    def image_folder(self, path):
        self._image_folder = (path, "images")

    @property
    def test_percentual_divisor(self):
        return self._test_percentual_divisor
    
    @test_percentual_divisor.setter
    def test_percentual_divisor(self, value):
        self._test_percentual_divisor = value