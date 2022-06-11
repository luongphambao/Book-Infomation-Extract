
from sklearn import feature_selection
import torch 
import os 
import cv2 
from PIL import Image
from text_recognition import VietOCR
from text_detect import Paddle_detection
from paddleocr import PaddleOCR
from preprocess_background import preprocess_background
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
        self.model_detect = torch.hub.load("yolov5", "custom", path = self.model_detect_path, force_reload=True,source='local')
        self.model_detect.eval()
        
    def load_text_detect_model(self):
        self.model_text_detect =PaddleOCR(use_gpu=True)
    def load_reg_model(self):
        self.model_reg = VietOCR(model_path=self.model_reg_path)
    def predict(self,img_path):
        img_folder=os.path.dirname(img_path)
        print(img_folder)
        #preprocess_background(img_path,"uploads")
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        model_detect=self.model_detect
        model_reg=self.model_reg
        model_text_detect=self.model_text_detect

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
            print(row)
            x1,x2,y1,y2 = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
            crop=img[y1:y2,x1:x2]
            #print(crop)
            #print(x1,x2,y1,y2)
            img_crop="./crop/"+str(row["class"])+"/"+str(index)+".jpg"
            cv2.imwrite(img_crop,crop)
            #print(img_crop)
            image=Image.open(img_crop)
            
            text,prob=model_reg.predict(image)
            print(prob)
            #print(prob)
            if float(prob) <0.7:
                
                result=Paddle_detection(model_text_detect,img_crop)
                boxes= [line[0] for line in result]
                result=""
                img_text_detect=cv2.imread(img_crop)
                count=0
                print("số lượng line",boxes)
                for box in boxes:
                    top_left     = (int(box[0][0]), int(box[0][1]))
                    bottom_right = (int(box[2][0]), int(box[2][1]))
                    xmin,ymin=min(top_left[0],bottom_right[0]),min(top_left[1],bottom_right[1])
                    xmax,ymax=max(top_left[0],bottom_right[0]),max(top_left[1],bottom_right[1])
                    ymax=int(ymax+10)
                    #ymin=int(ymin*1.2)
                    
                    crop=img_text_detect[ymin:ymax,xmin:xmax]
                    name_img="crop/"+str(row["class"])+"/"+str(index)+"_"+str(count)+".jpg"
                    cv2.imwrite(name_img,crop)

                    image=Image.open(name_img)
                    text_line,prob=model_reg.predict(image)
                    result=result+text_line+" "
                    #print(text_line)
                    text=result
                    count+=1
            #print(text)
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

# print(torch.cuda.is_available())
# predictor=Predictor()
# predictor.load_detect_model()
# predictor.load_reg_model()
# predictor.load_text_detect_model()
# print(predictor.predict("covers/Camera/20220331_101002.jpg"))