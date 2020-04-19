import cv2
import sys
import os
import numpy as np
from tqdm import tqdm
# Opens the Video file
mypath = sys.argv[1]
webcam =0
if not os.path.exists(mypath):
    if int(mypath) == 1:
        webcam =1
    else:
        print("No such file")
        exit()
outputdir = sys.argv[2]
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

i=0
if webcam == 1:
    cap= cv2.VideoCapture(int(mypath))
    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        cv2.imwrite(outputdir+'/frame'+str(i)+'.jpg',frame)
        i+=1
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    cap= cv2.VideoCapture(mypath)
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(outputdir+'/frame'+str(i)+'.jpg',frame)
            i+=1
        except AttributeError as e:
            print("End of video")
 
