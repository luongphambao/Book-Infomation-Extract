import matplotlib.pyplot as plt
import cv2 
import pandas as pd 
import numpy as np
import os 
import glob 
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# with open("val_.txt") as f:
#     lines = f.readlines()

for file in os.listdir("val_labels"):
    ax = plt.gca()
    file_path="val_labels/"+file
    with open(file_path) as f:
        lines = f.readlines()
    img_name=file.replace(".txt",".jpg")
    lines=[line.strip() for line in lines]
    if img_name not in os.listdir("val"):
        continue
    for line in lines:
        
        line=line.split(" ")

        
        # bbox=" ".join(line[1:])
        # print(bbox)
        # with open("val_labels/"+txt_name,"a+") as w:
        #     w.writelines(bbox+"\n")

        print("img name",img_name)
        bbox=[int(i) for i in line]
        print(bbox)
        
        img=cv2.imread("val/"+img_name)
        poly = [(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])]
        rect = patches.Polygon(poly ,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        #plt.imshow(img)
        plt.imshow(img)
    plt.savefig("visualize_ABC_detect/"+img_name)
    plt.close()
    #plt.show()
    #exit()