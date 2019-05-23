import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

if len(sys.argv) != 7:
    print("call with:")
    print("scoreboard image")
    print("match template of horizontal digit part")
    print("match template of vertical digit part")
    print("threshold for template matching on scoreboard image")
    print("factor for enlargement of digit part")
    print("output image")
    print("ie. python3 ./digits_extract.py scoreboard/venue1/scoreboard.png scoreboard/venue1/horizontal.png scoreboard/venue1/vertical.png 0.85 0.3 temp/res.png && display temp/res.png")
    exit()

# test files exist
if not os.path.isfile(sys.argv[1]):
    print("can't find file " + sys.argv[1])
    exit()
if not os.path.isfile(sys.argv[2]):
    print("can't find file " + sys.argv[2])
    exit()
if not os.path.isfile(sys.argv[3]):
    print("can't find file " + sys.argv[3])
    exit()



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
    cv2.rectangle(img_canvas, (pt[0] - expand, pt[1]), (pt[0] + wH + expand, pt[1] + hH), (0,0,0), -1)
    print(pt[0], pt[1])

#print(W, H, w, h)
res = cv2.matchTemplate(img_gray,template_2,cv2.TM_CCOEFF_NORMED)
threshold = float(sys.argv[4]) # 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_canvas, (pt[0], pt[1] - expand), (pt[0] + wV, pt[1] + hV + expand), (0,0,0),-1)
    print(pt[0], pt[1])

cv2.imwrite(sys.argv[6],img_canvas)
