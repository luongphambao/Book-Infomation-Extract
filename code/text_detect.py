from paddleocr import PaddleOCR
import cv2
# Also switch the language by modifying the lang parameter


def Paddle_detection(ocr,img_path):
    
    text = ocr.ocr(img_path)
    return text
if __name__ == '__main__':
    ocr = PaddleOCR() # The model file will be downloaded automatically when executed for the first time
    img_path ='crop/0/0.jpg'
    result = ocr.ocr(img_path)

    for line in result:
        print(line)
    
    # Visualization
    mat = cv2.imread(img_path)
    

    from PIL import Image
    image = Image.open(img_path).convert('RGB')
    boxes=[line[0] for line in result]
    #print(boxes)
    for box in boxes:  
        print(box)  
        top_left     = (int(box[0][0]), int(box[0][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))
    
        cv2.rectangle(mat, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite('result.jpg', mat)