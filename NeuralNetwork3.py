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



#get data from disk, split data into training and validation sets

train_dataset = prep.image_dataset_from_directory(

    directory="Data/Bulgarian",
    image_size=(300, 300),
    validation_split=0.4,
    subset= "training",
    batch_size= 32,
    label_mode= "categorical",
    seed= 1

)

test_dataset = prep.image_dataset_from_directory(
    directory="Data/Bulgarian",
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
    Dense(30, activation=softmax)

])

#callbacks
tensorboard_callback = TensorBoard(log_dir="C:\Sps things\Programing\PythonProjects\HandRecognition\Tensorboard\File1", histogram_freq=1)
checkpoint = ModelCheckpoint(
    filepath= "Models/model16",
    save_best_only= True,
    monitor= "val_loss",
    verbose= 1
)

#compile the model
model.compile(loss= CategoricalCrossentropy(), optimizer = Adam(), metrics = ['accuracy'])
model = tf.keras.models.load_model("Models/model13")

# Print details of each input layer
for input_tensor in model.inputs:
    print("Input Name:", input_tensor.name)
    print("Input Shape:", input_tensor.shape)
    print("Input Data Type:", input_tensor.dtype)
#train for 3 epochs/iterations
history = model.fit(train_dataset, epochs = 3, validation_data = test_dataset, callbacks=[tensorboard_callback, checkpoint])

print("banana")

