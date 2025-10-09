import tensorflow as tf
from numpy.ma.core import shape
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Dropout, Flatten, InputLayer
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import models
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.preprocessing as prep
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.python.platform import build_info
from gtts import gTTS
import os

text = "Здравей, това е тест на български език."
tts = gTTS(text=text, lang='bg')
tts.save("bulgarian.mp3")
os.system("start bulgarian.mp3")  # Use playsound if needed
