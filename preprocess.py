import cv2 
import numpy as np
import utils 
import os 


def preprocess(img_path,height=720,width=540):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx = 0.3, fy = 0.3)  # RESIZE ảnh
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT ảnh thành GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)  # GAUSSIAN BLUR
    imgThreshold = cv2.Canny(imgBlur, 30, 50)  # CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations = 2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations = 1)  # APPLY EROSION
    
    #find  all countours

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))# cắt bìa sách từ BIGEST COUNTOUR tìm được
        return imgWarpColored
    else:
        # Nếu không tìm thấy COUNTOUR thì ta vẫn thêm ảnh gốc vào nhưng sẽ in ra màng hình ảnh không contour được để cảnh báo
        print('not scanner image ', img_path)
        img = cv2.resize(img, (width, height))
        return img