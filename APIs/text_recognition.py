from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class VietOCR:
    def __init__(self, model_path="weights/transformerocr.pth", gpu_enable=False):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = model_path
        config['cnn']['pretrained'] = False
        # config['device'] = 'cuda:0'
        config['device'] = 'cpu' if gpu_enable==False else 'cuda:0'
        config['predictor']['beamsearch'] = False
        self.predictor = Predictor(config)

    def predict(self, img):
        result, prob = self.predictor.predict(img, return_prob=True)

        return result, prob
    def predict_craft(self, img):
        img = Image.fromarray(img)
        result, prob = self.predictor.predict(img, return_prob=True)

        return result, prob

