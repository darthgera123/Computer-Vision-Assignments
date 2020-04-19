import cv2
import numpy as np
import glob
import sys
import os
mypath = sys.argv[1]
if not os.path.exists(mypath):
    print("No such file")
    exit()
output = sys.argv[2]
frame = int(sys.argv[3])
img_array = []
name = []
for l in range(len(os.listdir(mypath))):
	k = mypath+'/frame'+str(l)+'.jpg'
	img = cv2.imread(k)
	name.append(k)
	height, width, layers = img.shape
	size = (width,height)
	img_array.append(img)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(output,fourcc, frame, size)
# print(name)
for img in img_array:
	out.write(img)
out.release()