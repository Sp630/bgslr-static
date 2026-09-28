# bgslr-static: Bulgarian dactylic (fingerspelling) alphabet recognition

Real-time recognizer for the Bulgarian one-handed dactylic (fingerspelling) alphabet. You spell letters in front of a webcam, they build into words, and a dedicated "speak" sign reads the word aloud in Bulgarian.

This is v1. It works on single frames, which can't capture signs that involve motion. [bgslr-video](https://github.com/Sp630/bgslr-video) is the follow-up that works on video.

## Download

Windows build with GPU support: [Google Drive](https://drive.google.com/file/d/17hFBpcfheOSHMR3a1k-Y5rhoojfhq5_d/view?usp=sharing). Unzip and run the executable. The interface is in Bulgarian.

## How it works

1. **Hand detection.** MediaPipe finds the hand and its 21 landmarks, which give a bounding box around it.
2. **Preprocessing.** The hand is cropped with a margin and centered on a 300×300 white canvas, keeping its aspect ratio, so every input has the same shape.
3. **Classification.** A VGG-style CNN in TensorFlow/Keras (four blocks of paired 3×3 convolutions, 32 → 256 filters, with dropout and max pooling, then two dense layers) predicts one of 30 classes: 29 letters plus the "speak" sign.
4. **Letters to words.** A letter is added only after the same prediction holds for about 10 consecutive frames, which filters out flicker. Holding the speak sign reads the word aloud with Bulgarian text-to-speech (gTTS).
5. **Data.** Trained on about 12,000 images I collected myself.

## Custom trainer

On startup the app offers to train a personal model: it walks you through recording images for each letter, trains a new CNN on them, and lets you switch between the general and personal model while running. It's currently set up for the first 3 letters, but also works for the entire alphabet.

## Files

- `Test.py`: the main app (camera, GUI, recognition loop, text-to-speech)
- `HandTrackingModule.py`: MediaPipe hand detection and bounding box
- `ClassificationModule.py`: loads the model and runs predictions
- `DataCollection.py`: records training images
- `NeuralNetwork3.py`: model definition and training
- `CustomTrainer.py`: in-app personal model training
- Other scripts (`NeuralNetwork.py`, `NeuralNetwork2.py`, `Android*.py` etc.) are earlier experiments

The Windows build includes the trained model.

**Requirements:** Python 3, TensorFlow, mediapipe, opencv-python, numpy, Pillow, gTTS, playsound, keyboard
