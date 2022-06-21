# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions
)
from craft_text_detector.file_utils import rectify_poly
from PIL import Image
import cv2
import os
import numpy as np
#from text_recognition import VietOCR
#sort chữ khi crop line để theo thứ tự từ trái sang phải
def sort_img(regions):
  #print("regions:",regions)
  for i in range(len(regions) - 1):
    min = i
    for j in range(i, len(regions)):
      if abs(regions[min][0, 1] - regions[j][0, 1]) > 10:
        if regions[min][0, 1] > regions[j][0, 1]:
          min = j
      else:
        if regions[min][0, 0] > regions[j][0, 0]:
          min = j
    regions[min], regions[i] = regions[i], regions[min]
  return regions

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
def predict_craft(img, key, craft_net, refine_net):
  image = read_image(img)
                    
  #predict craft
  if key == 0:
      prediction_result = get_prediction(
          image=image,
          craft_net=craft_net,
          refine_net=refine_net,
          text_threshold=0.7,
          link_threshold=0.3,
          low_text=0.3,
          cuda=True,
          long_size=1280
      )
  elif key == 3:
      prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.1,
            low_text=0.05,
            cuda=True,
            long_size=1280
        )
  else:
      prediction_result = get_prediction(
          image=image,
          craft_net=craft_net,
          refine_net=refine_net,
          text_threshold=0.7,
          link_threshold=0.1,
          low_text=0.2,
          cuda=True,
          long_size=1280
      )
  regions=prediction_result["polys"]
  regions_expand=[]
  for box in prediction_result['boxes']:
      box=order_points(box)
      bl,br,tr,tl=box[0],box[1],box[2],box[3]
      tl=[tl[0],int(tl[1]*1)]
      tr=[int(tr[0]),int(tr[1]*1)]
      bl=[bl[0],int(bl[1]*0.85)]
      br=[int(br[0]),int(br[1]*0.85)]
      regions_expand.append([bl,br,tr,tl])
  regions_expand=np.array(regions_expand)
  #sắp xếp lại ảnh sau khi craft
  #regions=order_points(regions)
  #print("regions:",regions)
  regions=sort_img(regions)
  regions_expand=sort_img(regions_expand)
  #print("region after sort:",regions)
  a=[]
  a_expand=[]
  for i in regions:
    a.append(rectify_poly(image, i))
  for i in regions_expand:
    a_expand.append(rectify_poly(image, i))
  return a,a_expand
  
