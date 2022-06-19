
from sklearn import feature_selection
import torch 
import os 
import cv2 
from PIL import Image
from text_recognition import VietOCR
from text_detect import Paddle_detection
from paddleocr import PaddleOCR
from preprocess_background import preprocess_background
import numpy as np
def processing_text(text_list):
    if len(text_list) == 0:
        return ""
    else:
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
    tl=[tl[0]*0.9,int(tl[1]*1.2)]
    tr=[int(tr[0]*1.1),int(tr[1]*1.2)]
    bl=[bl[0]*0.9,int(bl[1])]
    br=[int(br[0]*1.1),int(br[1])]

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
    warped = cv2.warpPerspective(img, M, (maxWidth, int(maxHeight)))
    return warped
class Predictor():
    def __init__(self):
        self.model_reg_path='./weights/seq2seq.pth'
        self.model_detect_path='./weights/best.pt'
    def load_detect_model(self):
        self.model_detect = torch.hub.load("yolov5", "custom", path = self.model_detect_path, force_reload=True,source='local')
        self.model_detect.eval()
        self.model_detect.conf=0.3
        
    def load_text_detect_model(self):
        self.model_text_detect =PaddleOCR(use_gpu=True)
    def load_reg_model(self):
        self.model_reg = VietOCR(model_path=self.model_reg_path)
    def predict(self,img_path):
        img_folder=os.path.dirname(img_path)
        print(img_folder)
        preprocess_background(img_path,"uploads")
        img=cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        model_detect=self.model_detect
        model_reg=self.model_reg
        model_text_detect=self.model_text_detect

        # predict region of information extract
        predict = model_detect(img, size=1000)    
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
            
            # text,prob=model_reg.predict(image)
            #print(prob)
            #print(prob)
            result=Paddle_detection(model_text_detect,img_crop)
            boxes= [line[0] for line in result]
            result=""
            img_text_detect=cv2.imread(img_crop)
            count=0
            print("số lượng line",boxes)
            for box in boxes:
                # top_left     = (int(box[0][0]), int(box[0][1]))
                # bottom_right = (int(box[2][0]), int(box[2][1]))
                # xmin,ymin=min(top_left[0],bottom_right[0]),min(top_left[1],bottom_right[1])
                # xmax,ymax=max(top_left[0],bottom_right[0]),max(top_left[1],bottom_right[1])
                
                # ymin=int(ymin*0.97)
                # ymax=int(ymax*1.05)
                crop = perspective_transform(img_text_detect, box)
                #crop=img_text_detect[ymin:ymax,xmin:xmax]
                name_img="crop/"+str(row["class"])+"/"+str(index)+"_"+str(count)+".jpg"
                cv2.imwrite(name_img,crop)

                image=Image.open(name_img)
                text_line,prob=model_reg.predict(image)
                print(text_line)
                print((prob))
                if float(prob)>0.7:
                    result+=text_line+" "
                #result=result+text_line+" "
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