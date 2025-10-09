import shutil
import time
import cv2
import numpy as np
#from cvzone import HandTrackingModule
from HandTrackingModule import handDetector
import math
import tensorflow
from tensorboard import summary
from tensorflow.keras.models import *
from pathlib import Path
import keyboard

keyReset = True


cap = cv2.VideoCapture(0)
#use HandTrackingModule
detector = handDetector(maxHands=1)

#folder = "Data/Bulgarian/1"
folder = "Data/Bulgarian/30"
counter = 0

#create a loop and use the camera
def CollectImages(directory, num, let):
    keyReset = False
    counter = 0
    dir = Path(directory)
    if dir.exists() and dir.is_dir():
        shutil.rmtree(dir)
    dir.mkdir(parents= True, exist_ok=True)
    while counter < num:
        success, img = cap.read()
        data, img = detector.findHands(img)
        bboffset = 20
        imgSize = 300

        #use the output provided by HandTrackingModule
        if data:
            bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
            w, h = bbxmax - bbxmin, bbymax - bbymin
            cropImg = img[bbymin - bboffset : bbymax + bboffset, bbxmin - bboffset : bbxmax + bboffset]
            if cropImg is not None and cropImg.size != 0:
              cv2.imshow("CropedImage", cropImg)

            imgCropShape = cropImg.shape
            imgWhite  = np.ones([imgSize, imgSize, 3], np.uint8)*255
            #print(imgCropShape)

            #prepare the image in the correct format
            aspectRatio = h / w

            if cropImg is not None and cropImg.size != 0 and cropImg.shape[0] <= 300 and cropImg.shape[1] <= 300:
            #resize so that all images have the same shape --> important for the CNN
                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(cropImg, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil(((300-wCal) / 2))
                    imgWhite[:, wGap:wCal+wGap] = imgResize
                else:
                    k=imgSize/w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(cropImg, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil(((300 - hCal) / 2))
                    imgWhite[hGap:hCal+hGap, :] = imgResize


                cv2.imshow("WhiteImage", imgWhite)

        #save show the image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s") and keyReset:
            counter += 1
            cv2.imwrite(f"{directory}/Image_{time.time()}.jpg", imgWhite)
            print(counter)
        elif not keyboard.is_pressed("s"):
            keyReset = True

    return counter

