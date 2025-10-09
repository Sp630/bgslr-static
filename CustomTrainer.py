import cv2
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
from pathlib import Path
import shutil
import tkinter as tk
import threading

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import DataCollection

global text
text = "Моля изберете опция"
root = tk.Tk()
destr = False

def StartCustomTrainer():
    root.title("BGSLR")
    root.geometry("800x800")
    videoLabel = tk.Label(root)
    videoLabel.pack()
    text1 = tk.Label(root, font=("Arial", 30))
    text1.pack(side="top", pady=10)
    counterText = tk.Label(root, font=("Arial", 30))
    counterText.pack(side="top", pady=10)
    wordText = tk.Label(root, font=("Arial", 30))
    wordText.pack(side="top", pady=10)

    bottomFrame = tk.Frame(root)
    bottomFrame.pack(side="bottom", pady=10)
    quitButton = tk.Button(bottomFrame,
                           text="Премини към разпознаване",
                           command=lambda: Start(),
                           font=("Roboto", 14),
                           width=30,
                           height=5
                           )
    quitButton.pack(side="right", padx=5)

    startButton = tk.Button(bottomFrame,
                             text="Тренирай",
                             command=lambda: OpenTrainingThread(text1),
                             font=("Roboto", 14),
                             width=15,
                             height=5
                             )
    startButton.pack(side="left", padx=5)

    CVtoTK(text1)
    root.mainloop()

def Start():
    global root
    cv2.destroyAllWindows()
    #root.quit()
    root.after(0, root.destroy)
    return

def CVtoTK(textField):
    global text
    textField.config(text=text)
    root.after(10, lambda: CVtoTK(textField))



#root.after(10, lambda: CVtoTK(videoLabel, root, text, counterText, wordText))

def OpenTrainingThread(text1):
    t4 = threading.Thread(target=Train)
    t4.start()

def Train():
    global text
    parentDir = Path("Data/Testing")
    if parentDir.exists() and parentDir.is_dir():
        shutil.rmtree(parentDir)







    global let
    let = 1
    end = 3
    classes = ["А", "Б", "В", "Г", "Д", "E", "Ж", "З", "И", "Й", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т",
               "У", "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ю", "Я", ""]
    while let != end + 1:
        print(f"Let is {let}")
        text = f"Натиснете S, за да запазите \n снимки на букв. {classes[let-1]}"
        DataCollection.CollectImages(f"Data/Testing/{let}", 300, 1)
        let += 1
        print(text)

    # <editor-fold desc="Model">


    #get data from disk, split data into training and validation sets
    print(let)
    text = "Моля изчакайте докато моделът  тренира..."
    cv2.destroyAllWindows()
    train_dataset = prep.image_dataset_from_directory(

        directory="Data/Testing",
        image_size=(300, 300),
        validation_split=0.4,
        subset= "training",
        batch_size= 32,
        label_mode= "categorical",
        seed= 1

    )

    test_dataset = prep.image_dataset_from_directory(
        directory="Data/Testing",
        image_size=(300, 300),
        validation_split=0.4,
        subset= "validation",
        batch_size = 32,
        label_mode = "categorical",
        seed=1
    )


    #review classes
    print(train_dataset.class_names)

    print(test_dataset.class_names)

    #costruct the model
    model = tf.keras.models.Sequential([
        Input(shape=(300, 300, 3)),
        Conv2D(32, 3, activation= relu),
        Conv2D(32, 3, activation= relu),
        Dropout(0.2),
        MaxPool2D(2),

        Conv2D(64, 3, activation= relu),
        Conv2D(64, 3, activation= relu),
        Dropout(0.2),
        MaxPool2D(2),

        Conv2D(128, 3, activation= relu),
        Conv2D(128, 3, activation= relu),
        Dropout(0.2),
        MaxPool2D(2),

        Conv2D(256, 3, activation=relu),
        Conv2D(256, 3, activation=relu),
        Dropout(0.2),
        MaxPool2D(2),

        Flatten(),
        Dense(512, activation=relu),
        Dropout(0.2),
        Dense(256, activation=relu),
        Dense(let - 1, activation=softmax)


    ])

    #callbacks
    tensorboard_callback = TensorBoard(log_dir="C:\Sps things\Programing\PythonProjects\HandRecognition\Tensorboard\File1", histogram_freq=1)
    checkpoint = ModelCheckpoint(
        filepath= "Models/CustomModels/model1",
        save_best_only= True,
        monitor= "val_loss",
        verbose= 1
    )

    #compile the model
    model.compile(loss= CategoricalCrossentropy(), optimizer = Adam(), metrics = ['accuracy'])
    # </editor-fold>

    #train for 3 epochs/iterations
    history = model.fit(train_dataset, epochs = 3, validation_data = test_dataset, callbacks=[checkpoint])

    print("orange")
    Start()
#StartCustomTrainer()