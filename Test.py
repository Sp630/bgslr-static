#this file connects all the modules together
#first some housekeeping must be done to load the dlls correctly
import os
import sys

base_path = os.path.dirname(os.path.abspath(sys.executable))
os.environ["PATH"] += os.pathsep + os.path.join(base_path, 'dlls')

import time
import cv2
import numpy as np
from docutils.nodes import classifier
from numpy.lib.polynomial import roots

from HandTrackingModule import handDetector
import math
import tensorflow as tf
from tensorboard import summary
from tensorflow.keras.models import *
import ClassificationModule
import gc
import tensorflow.keras.backend as K
import threading
import tkinter as tk
from PIL import Image, ImageTk
import CustomTrainer
from gtts import gTTS
from playsound import playsound

print("Available devices:", tf.config.list_physical_devices())

#ensure proper usage of physical devices
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CustomTrainer.StartCustomTrainer()

#threading synchronization
stop_event = threading.Event()
sharedData = None
sharedCount = None
globalText = ""
dataLock = threading.Lock()
useCustomModel = False
classifier = ClassificationModule.Classifier("Models/model13")

#videoCapture
cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)

#GUI
#Tkinter
counter = 0
def CVtoTK(videoLabel, root, text, counterText, wordText):
    success, img = cap.read()
    if success:
        data, img = detector.findHands(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wholeImg = Image.fromarray(img)
        tkimage = ImageTk.PhotoImage(image = wholeImg)
        videoLabel.tkimage = tkimage
        videoLabel.config(image=tkimage)


        with dataLock:
            global sharedData
            global counter
            global globalText
            #print(sharedData)
            text.config(text= sharedData)
            counterText.config(text= "")
            newText = ""
            for i in range(counter):
                newText += "*"
            counterText.config(text= newText)
            wordText.config(text= globalText)

    root.after(10, lambda: CVtoTK(videoLabel, root, text, counterText, wordText))


def StartTkinter():
    root = tk.Tk()
    root.title("BGSLR")
    root.geometry("800x800")
    videoLabel = tk.Label(root)
    videoLabel.pack()
    text = tk.Label(root, font=("Arial", 30))
    text.pack(side="top", pady=10)
    counterText = tk.Label(root, font=("Arial", 30))
    counterText.pack(side="top", pady=10)
    wordText = tk.Label(root, font=("Arial", 30))
    wordText.pack(side="top", pady=10)

    bottomFrame = tk.Frame(root)
    bottomFrame.pack(side="bottom", pady=10)
    quitButton = tk.Button(bottomFrame,
                           text="Излез",
                           command=lambda: Quit(root),
                           font=("Roboto", 14),
                           width=10,
                           height=5
                           )
    quitButton.pack(side="right", padx=5)

    switchButton = tk.Button(bottomFrame,
                             text="Смени Модел",
                             command=lambda: SwitchModel(),
                             font=("Roboto", 14),
                             width=15,
                             height=5
                             )
    switchButton.pack(side="left", padx=5)





    CVtoTK(videoLabel, root, text, counterText, wordText)
    root.mainloop()
def Quit(root):
    stop_event .set()

    #Ensure resources are properly released

    root.quit()
    root.destroy()
    sys.exit()

def SwitchModel():
    global useCustomModel
    global classifier
    if(useCustomModel == True):
        classifier = ClassificationModule.Classifier("Models/model13")
        useCustomModel = False
    else:
        classifier = ClassificationModule.Classifier("Models/CustomModels/model1")
        useCustomModel = True


def Train(root):
    t3 = threading.Thread(target=CustomTrainer.Train)
    t3.start()
    root.quit()
    root.destroy()


t2 = threading.Thread(target=StartTkinter)
t2.start()

#Load the model; done at the beginning to prevent slow-downs inside the loop
# if(useCustomModel):
#     classifier = ClassificationModule.Classifier("Models/model15")
# else:
#     classifier = ClassificationModule.Classifier("Models/model13")

globalImage = None
pred = None

#use this if you don't want GUI
def ShowVideo():


        success, img = cap.read()
        data, img = detector.findHands(img)
        globalImage = img
        #cv2.putText(text)
        cv2.putText((img), str(classes[np.argmax(pred)]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        


#t1 = threading.Thread(target=ShowVideo, daemon= True)
#t1.start()


#Text Control
var = None
handIsInFrame = False
counter = 0
globalText = " "
currentModel = "general"
if(useCustomModel):
    currentModel = "custom"
while not stop_event.isSet():
    # if(useCustomModel == False and currentModel == "custom"):
    #     classifier = ClassificationModule.Classifier("Models/model13")
    # elif(useCustomModel == True and currentModel == "general"):
    #     classifier = ClassificationModule.Classifier("Models/model15")

    success, img = cap.read()
    data = None
    data, img = detector.findHands(img)
    bboffset = 20
    imgSize = 300
    customSign = 23
    if classifier.result is not None:
        prediction = classifier.result
        if useCustomModel:
            classes = ["А", "Б", "В", "Г", "Д", "E", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т",
                       "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", ""]
            customSign = 29
        else:
            classes = ["А", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", "В", "", "Г", "Д", "E", "Ж", "З", "И"]
            customSign = 23
        #print(np.argmax(prediction))
        if classes[np.argmax(prediction)] == var and counter >= 10:
            if(np.argmax(prediction) == customSign):
                text = globalText
                if text is not None and text != "" and text != " ":
                    tts = gTTS(text=text, lang='bg')
                    tts.save("bulgarian.mp3")
                    playsound("bulgarian.mp3")
                    os.remove("bulgarian.mp3")
                    globalText = ""
            else:
                globalText = globalText + classes[np.argmax(prediction)]
            counter = 0
            var = classes[np.argmax(prediction)]
        elif classes[np.argmax(prediction)] is not var:
            var = classes[np.argmax(prediction)]
            counter = 0
        elif classes[np.argmax(prediction)] == var and handIsInFrame:
            counter += 1
        #print(np.argmax(prediction))
        #print(classes[np.argmax(prediction)])
        #print(globalText)
        sharedData = classes[np.argmax(prediction)]
        pred = prediction
#image recognition
    if data:
        handIsInFrame = True
        bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
        w, h = bbxmax - bbxmin, bbymax - bbymin
        cropImg = img[bbymin - bboffset: bbymax + bboffset, bbxmin - bboffset: bbxmax + bboffset]
        if cropImg is not None and cropImg.size != 0:
          cv2.imshow("CropedImage", cropImg)

        imgCropShape = cropImg.shape
        imgWhite = np.ones([imgSize, imgSize, 3], np.uint8) * 255
        #Reshaping so that the model can use it

        aspectRatio = h / w

        if cropImg.shape[0] <= 300 and cropImg.shape[1] <= 300 and cropImg is not None and cropImg.size != 0:

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(cropImg, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil(((300 - wCal) / 2))
                if imgResize.shape[0] <= 300 and imgResize.shape[1] <= 300:
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    classifier.getPrediction(imgWhite)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(cropImg, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil(((300 - hCal) / 2))
                if imgResize.shape[0] <= 300 and imgResize.shape[1] <= 300:

                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    classifier.getPrediction(imgWhite)

        else:
            sharedData = "Моля отдалечете се"
        cv2.imshow("WhiteImage", imgWhite)
        key = cv2.waitKey(1)
    else:
        handIsInFrame = False


cap.release()
