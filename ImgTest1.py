#print(im.format)
import cv2 
#import pytesseract

import numpy as np
from IMGs import JMCVImageLibrary_CV2 as jmcv2
#img = cv2.imread('../../pic/IMG_3742.jpeg')
dir='/Users/jerry/Documents/Air/AirSample/'
fileName='62.jpg'
#fileName='55.jpg'
fileName='82.jpg'
img = cv2.imread(dir+fileName)
#cv2.imshow('img',img)
#cv2.waitKey(0)
# add custom options

#h, w, c = img.shape
#boxes = pytesseract.image_to_boxes(img) 
#for b in boxes.splitlines():
#    b = b.split(' ')
#    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

img2=jmcv2.get_grayscale(img)
#img3=jmcv2.thresholding(img2)
img3=jmcv2.canny(img2)
#cv2.imshow(fileName, img3)
#cv2.waitKey(0)
cv2.imwrite(dir+'canny'+fileName,img3)
#custom_config = r'--oem 3 --psm 6'
#s=pytesseract.image_to_string(img, config=custom_config)

#print(s)


