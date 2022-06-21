from paddleocr import PaddleOCR
import cv2
# Also switch the language by modifying the lang parameter


def Paddle_detection(ocr,img_path):
    
    results = ocr.ocr(img_path)
    return results 
# import craft functions
# from craft_text_detector import (
#     read_image,
#     load_craftnet_model,
#     load_refinenet_model,
#     get_prediction,
# )
# from craft_text_detector.file_utils import rectify_poly
# from PIL import Image
# import cv2
# import os
# from text_recognition import VietOCR
# #sort chữ khi crop line để theo thứ tự từ trái sang phải
# def sort_img(regions):
#   print("regions:",regions)
#   for i in range(len(regions) - 1):
#     min = i
#     for j in range(i, len(regions)):
#       if abs(regions[min][0, 1] - regions[j][0, 1]) > 10:
#         if regions[min][0, 1] > regions[j][0, 1]:
#           min = j
#       else:
#         if regions[min][0, 0] > regions[j][0, 0]:
#           min = j
#     regions[min], regions[i] = regions[i], regions[min]
#   return regions

    
# def predict_craft(img, key, craft_net, refine_net):
#   image = read_image(img)
                    
#   #predict craft
#   if key == 0:
#       prediction_result = get_prediction(
#           image=image,
#           craft_net=craft_net,
#           refine_net=refine_net,
#           text_threshold=0.7,
#           link_threshold=0.3,
#           low_text=0.3,
#           cuda=True,
#           long_size=1280
#       )
#   elif key == 3:
#       prediction_result = get_prediction(
#             image=image,
#             craft_net=craft_net,
#             refine_net=refine_net,
#             text_threshold=0.7,
#             link_threshold=0.1,
#             low_text=0.05,
#             cuda=True,
#             long_size=1280
#         )
#   else:
#       prediction_result = get_prediction(
#           image=image,
#           craft_net=craft_net,
#           refine_net=refine_net,
#           text_threshold=0.7,
#           link_threshold=0.1,
#           low_text=0.2,
#           cuda=True,
#           long_size=1280
#       )
#   regions=prediction_result["polys"]
  
#   #sắp xếp lại ảnh sau khi craft
#   print(regions)
#   regions=sort_img(regions)
#   print("region after sort:",regions)
#   a=[]
#   for i in regions:
#     a.append(rectify_poly(image, i))
#   return a
  

# def craft_detect():

#     #load model
#     refine_net = load_refinenet_model(cuda=True)
#     craft_net = load_craftnet_model(cuda=True)
    
#     img_path="val/33.jpg" 
#     img = Image.open(img_path)
#     text_reg_model=VietOCR(model_path="weights/seq2seq.pth")
#     a = read(img_path, 0, craft_net, refine_net)
#     for i in a:
#       text,prob=text_reg_model.predict_craft(i)
#       if prob>0.7:
#         print(text,prob)
# craft_detect()
# if __name__ == '__main__':
#     ocr = PaddleOCR() # The model file will be downloaded automatically when executed for the first time
#     img_path ='crop/0/0.jpg'
#     result = ocr.ocr(img_path)

#     for line in result:
#         print(line)
    
#     # Visualization
#     mat = cv2.imread(img_path)
    

#     from PIL import Image
#     image = Image.open(img_path).convert('RGB')
#     boxes=[line[0] for line in result]
#     #print(boxes)
#     for box in boxes:  
#         print(box)  
#         top_left     = (int(box[0][0]), int(box[0][1]))
#         bottom_right = (int(box[2][0]), int(box[2][1]))
    
#         cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
#     cv2.imwrite('result.jpg', mat)
