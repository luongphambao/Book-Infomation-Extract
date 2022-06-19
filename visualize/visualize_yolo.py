from sklearn import feature_selection
import torch 
import os 
import cv2 
from PIL import Image
import matplotlib.pyplot as plt

yolo_detect_model=torch.hub.load("yolov5", "custom", path = "./weights/best.pt", force_reload=True,source='local')
yolo_detect_model.eval()

for file in os.listdir("Test"):
    img_path="Test/"+file
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict = yolo_detect_model(img, size=720)
    locate = predict.pandas().xyxy[0]
    print("Yolo predict sucess")
    for index, row in locate.iterrows():
            print(row)
            x1,x2,y1,y2 = int(row['xmin']),int(row['xmax']),int(row['ymin']),int(row['ymax'])
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            plt.imshow(img)
    plt.savefig("visualize_yolodetect2/"+file)