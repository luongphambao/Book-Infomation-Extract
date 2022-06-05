
from sklearn import feature_selection
import torch 
import os 
import cv2 
from PIL import Image
from text_recognition import VietOCR

def processing_text(text_list):
    if len(text_list) == 0:
        return ""
    else:
        return " ".join(text_list)
class Predictor():
    def __init__(self):
        self.model_reg_path='./weights/transformerocr.pth'
        self.model_detect_path='./weights/best.pt'
    def load_detect_model(self):
        self.model_detect = torch.hub.load("ultralytics/yolov5", "custom", path = self.model_detect_path, force_reload=True)
        self.model_detect.eval()
        self.model_detect.conf=0.7
        
    def load_reg_model(self):
        self.model_reg = VietOCR(model_path=self.model_reg_path)
    def predict(self,img_path):
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        model_detect=self.model_detect
        model_reg=self.model_reg
        # predict region of information extract
        predict = model_detect(img, size=720)    
        locate = predict.pandas().xyxy[0]
        print("Yolo predict sucess")
        #lấy danh sách kết quả
        ten_sach = []
        ten_tac_gia = []
        nha_xuat_ban = []
        tap = []
        nguoi_dich = []
        tai_ban = []
        #cắt từng vùng ảnh và thêm vảo mảng đã định danh và dự đoán dựa trên việt OCR
        for index, row in locate.iterrows():
            
            x1,x2,y1,y2 = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
            crop=img[y1:y2,x1:x2]
            img_crop="./crop/"+str(row["class"])+"/"+str(index)+".jpg"
            cv2.imwrite(img_crop,crop)
            image=Image.open(img_crop)
            text,prob=model_reg.predict(image)
            if prob <0.6:
                continue
            #crop.write("crop/"+str(row['class'])+".jpg")
            #exit()
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
        ten_tac_gia = processing_text(ten_tac_gia)
        nha_xuat_ban = processing_text(nha_xuat_ban)
        tap=processing_text(tap)
        nguoi_dich=processing_text(nguoi_dich)
        tai_ban=processing_text(tai_ban)
        features = {
            0: ten_sach,
            1: ten_tac_gia,
            2: nha_xuat_ban,
            3: tap,
            4: nguoi_dich,
            5: tai_ban
        }
        return features
    
# predictor=Predictor()
# predictor.load_detect_model()
# predictor.load_reg_model()
# print(predictor.predict("uploads/0.jpg"))