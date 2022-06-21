
from sklearn import feature_selection
import torch
import os
import cv2
from PIL import Image
import glob
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
)
from craft_text_detector.file_utils import rectify_poly
from APIs.text_recognition import VietOCR
#from APIs.text_detect import Paddle_detection
from APIs.craft import predict_craft
from preprocess_background import preprocess_background
# except:
#     from text_recognition import VietOCR
#     from text_detect import Paddle_detection
#     from craft import predict_craft
    #from preprocess_background import preprocess_background
from paddleocr import PaddleOCR
import numpy as np 
import scipy.spatial.distance as distance
import json
# from code.preprocess_background import preprocess_background
import pandas as pd

def processing_text(text_list):
    if len(text_list) == 0:
        return ""
    else:
        #text_list=[x.lower() for x in text_list if x!=""]
        return " ".join(text_list)
def order_points(pts):
    if isinstance(pts, list):
        pts = np.asarray(pts, dtype='float32')
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # tl=[tl[0]*0.9,int(tl[1]*1.2)]
    # tr=[int(tr[0]*1.1),int(tr[1]*1.2)]
    # bl=[bl[0]*0.9,int(bl[1])]
    # br=[int(br[0]*1.1),int(br[1])]

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (int(maxWidth), int(maxHeight*1.2)))
    return warped

class Predictor():
    def __init__(self):
        self.model_reg_path = './weights/transformerocr.pth'
        self.model_detect_path = './weights/best.pt'

    def load_detect_model(self):
        self.model_detect = torch.hub.load(
            "yolov5", "custom", path=self.model_detect_path, force_reload=True, source='local')
        self.model_detect.eval()
        self.model_detect.conf=0.4
    def load_craft_model(self):
        self.refine_net = load_refinenet_model(cuda=True)
        self.craft_net = load_craftnet_model(cuda=True)
    def load_reg_model(self):
        self.model_reg = VietOCR(model_path=self.model_reg_path)
    def load_paddle_model(self):
        self.paddle_ocr = PaddleOCR()
    def predict(self, img_path,detect_model="craft"):
        img_folder = os.path.dirname(img_path)
        preprocess_background(img_path,"static/src/uploads")
        #results_paddle=Paddle_detection(self.paddle_ocr,img_path)      
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model_detect = self.model_detect
        model_reg = self.model_reg
        #model_text_detect = self.model_text_detect
        # predict region of information extract
        predict = model_detect(img, size=800)
        locate = predict.pandas().xyxy[0]
        print("Yolo predict sucess")
        # lấy danh sách kết quả
        ten_sach = []
        ten_tac_gia = []
        nha_xuat_ban = []
        tap = []
        nguoi_dich = []
        tai_ban = []
        # cắt từng vùng ảnh và thêm vảo mảng đã định danh và dự đoán dựa trên việt OCR
        for index, row in locate.iterrows():
            text=""
            print(row)
            x1, x2, y1, y2 = int(row['xmin']), int(
                row['xmax']), int(row['ymin']), int(row['ymax'])
            crop = img[y1:y2, x1:x2]
            # print(crop)
            # print(x1,x2,y1,y2)
            img_crop = "./crop/"+str(row["class"])+"/"+str(index)+".jpg"
            print(img_crop)
            cv2.imwrite(img_crop, crop)
            # print(img_crop)
            if row["class"] >2:
                crop=Image.open(img_crop)
                text,prob=model_reg.predict(crop)
            else:
                text=[]
                craft_result,craft_result_expand=predict_craft(crop, row['class'], self.craft_net, self.refine_net)
                for i in range(len(craft_result)):
                    text_predict,prob=model_reg.predict_craft(craft_result[i])
                    text_predict_expand,prob_expand=model_reg.predict_craft(craft_result_expand[i])
                    if prob_expand==max(prob,prob_expand):
                        text_predict=text_predict_expand
                    if  max(prob,prob_expand)>0.7:
                        text.append(text_predict)
                text=processing_text(text)
            if row['class'] == 0:
                ten_sach.append(text)
            elif row['class'] == 1:
                ten_tac_gia.append(text)
            elif row['class'] == 2:
                nha_xuat_ban.append(text)
            elif row['class'] == 3:
                tap.append(text)
            elif row['class'] == 4:
                nguoi_dich.append(text)
            else:
                tai_ban.append(text)
        ten_sach = processing_text(ten_sach)
        #ten_sach=ten_sach.upper()
        ten_tac_gia = processing_text(ten_tac_gia)
        ten_tac_gia=ten_tac_gia.upper()
        nha_xuat_ban = processing_text(nha_xuat_ban)
        tap = processing_text(tap)
        nguoi_dich = processing_text(nguoi_dich)
        nguoi_dich=nguoi_dich.upper()
        tai_ban = processing_text(tai_ban)

        #post processs
        if ten_tac_gia in ten_sach:
            ten_sach.replace(ten_tac_gia,"")
        if nguoi_dich in tai_ban:
            tai_ban.replace(nguoi_dich,"")
        # boxes = [line[0] for line in results_paddle]
        # txts = [line[1][0] for line in results_paddle]
        # scores = [line[1][1] for line in results_paddle]
        features = {
            0: ten_sach,
            1: ten_tac_gia,
            2: nha_xuat_ban,
            3: tap,
            4: nguoi_dich,
            5: tai_ban
        }
        json_file="results/"+os.path.basename(img_path).split(".")[0]+".json"
        with open(json_file, 'w') as f:
            json.dump(features, f)
        return features
