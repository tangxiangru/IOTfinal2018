import cv2
import os
import sys
import dlib

imgs = sys.argv[1:-1]
k = float(sys.argv[-1])

for each in imgs:
    img = cv2.imread(each)
    h, w = img.shape[:2]
    print(each, img.shape)
    size = (int(w*k), int(h*k))
    img_processed = cv2.resize(img, size)
    #cv2.imshow(each, img_processed)
    cv2.imwrite(each, img_processed)