# This module is no longer supported. Android Development has moved to Android Studio.
# this file connects all the modules together
# <editor-fold desc="Imports">
import sys
import time
import cv2
import numpy as np
from HandTrackingModule import handDetector
import math
import tensorflow as tf
from tensorboard import summary
from tensorflow.keras.models import *
import ClassificationModule
import os
import gc
import tensorflow.keras.backend as K
import threading
import tkinter as tk
from PIL import Image, ImageTk
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from functools import partial
from kivy.clock import mainthread


# </editor-fold>

# ensure proper usage of physical devices
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# threading synchronization
stop_event = threading.Event()
sharedData = "sharedData"
sharedCount = 0
globalText = "globalText"
dataLock = threading.Lock()

# videoCapture
cap = cv2.VideoCapture(0)
detector = handDetector(maxHands=1)

# GUI
# Tkinter
counter = 0

# Text Control
def TextControl():
    # Load the model; done at the beginning to prevent slow-downs inside the loop
    classifier = ClassificationModule.Classifier("Models/model13")
    global sharedData, counter, globalText
    var = "None"
    handIsInFrame = False
    counter = 0
    globalText = " "
    while not stop_event.isSet():
        #print(handIsInFrame)
        success, img = cap.read()
        data = None
        data, img = detector.findHands(img)
        bboffset = 20
        imgSize = 300
        if classifier.result is not None:
            prediction = classifier.result
            classes = ["А", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Б", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", "В", "", "Г", "Д", "E", "Ж", "З", "И"]
            #classes = ["А", "Б", "В", "Г", "Д", "E", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У",
            #           "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", ""]
            # print(np.argmax(prediction))
            with dataLock:
                if classes[np.argmax(prediction)] == var and counter >= 10:
                    if (np.argmax(prediction) == 23):
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
                # print(np.argmax(prediction))
                print(classes[np.argmax(prediction)])
                # print(globalText)
                sharedData = classes[np.argmax(prediction)]
                pred = prediction
        # image recognition
        if data:
            handIsInFrame = True
            bbxmax, bbxmin, bbymax, bbymin = data["bbox"]
            w, h = bbxmax - bbxmin, bbymax - bbymin
            cropImg = img[bbymin - bboffset: bbymax + bboffset, bbxmin - bboffset: bbxmax + bboffset]
            if cropImg is not None and cropImg.size != 0:
                cv2.imshow("CropedImage", cropImg)

            imgCropShape = cropImg.shape
            imgWhite = np.ones([imgSize, imgSize, 3], np.uint8) * 255
            # Reshaping so that the model can use it

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

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.video_label = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 0.85))
        self.add_widget(self.video_label)

        self.text = Label(text= "text", font_size=30)
        self.add_widget(self.text)

        self.counter_text = Label(text= "counter-text", font_size=30)
        self.add_widget(self.counter_text)

        self.word_text = Label(text= "word_text",font_size=30)
        self.add_widget(self.word_text)

        self.quit_button = Button(text="Излез", size_hint=(1, 0.15), font_size=20)
        self.quit_button.bind(on_press=self.quit_app)
        self.add_widget(self.quit_button)

        # Start the CV processing loop

        red_img = np.ones((480, 640, 3), dtype=np.uint8) * np.array([0, 0, 255], dtype=np.uint8)  # Pure red
        red_buf = red_img.tobytes()
        red_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        red_texture.blit_buffer(red_buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_label.texture = red_texture
        threading.Thread(target=self.cv_loop, daemon=True).start()

    @mainthread
    def updateTexture(self, red_img):

        red_buf = red_img.tobytes()
        red_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        red_texture.blit_buffer(red_buf, colorfmt='rgb', bufferfmt='ubyte')
        self.video_label.texture = red_texture

    def cv_loop(self):



        global sharedData, counter, globalText

        while not stop_event.is_set():
            success, img = cap.read()
            if not success:
                continue


              #print("We made it, we have success = True, now what?")
            data, img = detector.findHands(img)
            img = cv2.flip(img, 0)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            red_img = np.ones((480, 640, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)  # Pure red
            self.updateTexture(img_rgb)

        #     # # Convert to texture for Kivy
        #     # buf = img_rgb.tobytes()
        #     # texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        #     # texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        #
        #     #Test
        #     red_img = np.ones((480, 640, 3), dtype=np.uint8) * np.array([0, 0, 255], dtype=np.uint8)  # Pure red
        #     red_buf = red_img.tobytes()
        #     red_texture = Texture.create(size=(640, 480), colorfmt='rgb')
        #     red_texture.blit_buffer(red_buf, colorfmt='rgb', bufferfmt='ubyte')
        #     Clock.schedule_once(lambda dt: self.video_label.setter('texture')(self.video_label, red_texture))
        #     # Update Kivy UI from main thread
        #     #Clock.schedule_once(lambda dt, tex=texture: self.video_label.setter('texture')(self.video_label, tex))
            with dataLock:
                # Update label texts
                Clock.schedule_once(lambda dt: self.text.setter('text')(self.text, sharedData))
                Clock.schedule_once(lambda dt: self.counter_text.setter('text')(self.counter_text, '*' * counter))
                Clock.schedule_once(lambda dt: self.word_text.setter('text')(self.word_text, globalText))

        #     #cv2.waitKey(10)

    def quit_app(self, instance):
        App.get_running_app().stop()
        stop_event.set()

        # Ensure resources are properly released
        sys.exit()



class BGSLRApp(App):
    def build(self):
        self.title = "BGSLR"
        t1 = threading.Thread(target=TextControl, daemon=True)  # Notice no parentheses after TextControl
        t1.start()
        return MainLayout()

if __name__ == '__main__':
    BGSLRApp().run()


def Quit(root):
    stop_event.set()

    # Ensure resources are properly released
    root.quit()
    root.destroy()
    sys.exit()

def StartKivy():
    BGSLRApp().run()

t1 = threading.Thread(target=StartKivy(), daemon= True)
t1.start()


globalImage = None
pred = None







cap.release()
