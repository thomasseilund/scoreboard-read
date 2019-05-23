import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread(sys.argv[1])
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template_1 = cv2.imread(sys.argv[2],0)
template_2 = cv2.imread(sys.argv[3],0)
W, H = img_gray.shape[::-1]
img_canvas = 255 * np.ones(shape=[H, W, 3], dtype=np.uint8)

wH, hH = template_1.shape[::-1]
wV, hV = template_2.shape[::-1]

expand = round(wH * float(sys.argv[5]))

#print(W, H, w, h)
res = cv2.matchTemplate(img_gray,template_1,cv2.TM_CCOEFF_NORMED)
threshold = float(sys.argv[4]) # 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    # cv2.rectangle(img_canvas, (pt[0] /*- hH*/, pt[1]), (pt[0] + wH /*+ hH*/, pt[1] + hH), (0,0,0), -1)
    # cv2.rectangle(img_canvas, (pt[0] - wV, pt[1]), (pt[0] + wH + wV, pt[1] + hH), (0,0,0), -1)
    cv2.rectangle(img_canvas, (pt[0] - expand, pt[1]), (pt[0] + wH + expand, pt[1] + hH), (0,0,0), -1)
    print(pt[0], pt[1])

#print(W, H, w, h)
res = cv2.matchTemplate(img_gray,template_2,cv2.TM_CCOEFF_NORMED)
threshold = float(sys.argv[4]) # 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    # cv2.rectangle(img_canvas, (pt[0], pt[1] /*- wV*/), (pt[0] /*+ wV*/, pt[1] + hV + wV), (0,0,0),-1)
    # cv2.rectangle(img_canvas, (pt[0], pt[1] - hH), (pt[0] + wV, pt[1] + hV + hH), (0,0,0),-1)
    cv2.rectangle(img_canvas, (pt[0], pt[1] - expand), (pt[0] + wV, pt[1] + hV + expand), (0,0,0),-1)
    print(pt[0], pt[1])

cv2.imwrite('res.png',img_canvas)
