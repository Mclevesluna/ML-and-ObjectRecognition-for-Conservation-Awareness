# British Birdsong Selector with Finger Counting and Voice Control

## Overview

This project presents a novel system enabling users to select one out of five British birdsongs by displaying the corresponding number of fingers and controlling the volume of the birdsong using speech commands. The system integrates two key components: object detection and speech recognition. It offers flexibility by accepting three types of inputs: still images, webcam videos, and live webcam streams.

## Project Structure

### Introduction and Background

This system allows users to select one out of 5 British birdsongs by holding up the number of fingers that corresponds to each birdsong. Furthermore, users can control the volume of the birdsong that is playing with their speech by saying the words "up" or "down". The machine learning system is composed of two key parts: 
1. Object detection 
2. Speech recognition

To provide greater flexibility, the system can receive three types of inputs:
1. A still image taken with the webcam
2. A video taken with the webcam
3. A live webcam stream used in real-time

### Methodology

#### General Logic

The system is trained on a dataset that includes color images of people displaying various numbers of fingers on both left and right hands. The model trained on these images can identify the number of fingers appearing in an image, with labels ranging from 0 to 4, corresponding to numbers 1 to 5 respectively.

#### System Components

1. **Import Libraries**: Import necessary libraries for data manipulation, image processing, model building, and training.
2. **Load Images**: Load images from specified folders (train and test) and their corresponding labels.
3. **Transform Images and Prepare for Training**: Use the Albumentations library to augment and preprocess images for training.
4. **Create Model Classes**: Define custom model classes using PyTorch, implementing an attention mechanism on top of a pre-trained VGG16 model.
5. **Adjust Loss and Optimizer**: Define the loss function and optimizer for training the model.
6. **Train Model on GPU**: Handle the training loop, running batches of training data through the model, calculating losses, and optimizing model parameters.
7. **Save Model**: Save the trained model's state dictionary.
8. **Load Model on CPU**: Load the model from the saved state dictionary and set it to evaluation mode.
9. **Capture Image**: Capture an image or video using a webcam.
10. **Make Prediction for an Image**: Make a prediction for the captured image using the trained model and play the corresponding birdsong.
11. **Launch Voice Recognition for Volume Control**: Use voice recognition for adjusting volume based on speech commands.
12. **Capture a Video**: Capture a video using a webcam and save it.
13. **Make Prediction for a Video**: Make predictions for each frame of the captured video.
14. **Start Live Webcam**: Initiate the live webcam feed and make predictions for each frame in real time.

### Experimentation

#### Challenges and Adjustments

The greatest challenge was achieving accuracy with real-life inputs. Initial models were inaccurate with image inputs other than the train/test data. Several iterations were made to improve real-life accuracy:

1. **Model Architecture Adjustments**: Adding convolutional layers and a fully connected layer.
2. **Dataset Changes**: Using more varied and realistic images with different backgrounds.
3. **Model and Environment Adjustments**: Switching to a VGG16 model type on Google Colab.
4. **Hyperparameter Tuning**: Adjusting epochs, learning rate, and batch size.
5. **Libraries and Environment Adjustments**: Switching to PyTorch for model complexity and using an attention mechanism.

The latest version of the model leverages an existing VGG16 model, achieving acceptable accuracy for real-life applications. Further development includes integrating sound playback and speech recognition functionalities.

### Results

The model functions well for finger counting in images, videos, and live webcam streams. It successfully plays the corresponding birdsong and adjusts volume based on speech commands. However, there are still some inaccuracies when functioning with real-life webcam inputs.

### Future Work

1. **Improve Accuracy**: Continue improving accuracy, especially for video and live webcam functions.
2. **Integrate Audio and Speech Recognition**: Incorporate audio and speech recognition elements into the video and live webcam functions.

## Dependencies

The following libraries are required to run the project:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
import cv2
import numpy as np
```
## How to run

1. **Clone the repository**
2. **Setup environment**
3. **Install dependencies**
```python
pip install -r requirements.txt
```
4. **Download and Prepare Datasets:** Download the datasets (e.g., British Birdsong Dataset, Selfie Finger Counting Dataset) and place them in the appropriate directories as specified in the code.

Here are the links to the datasets used:

[British Birdsong Dataset](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset)
[Selfie Finger Counting Dataset](https://www.kaggle.com/datasets/zakitaleb/selfie-finger-counting?resource=download)

5. **Train the Model:**

Use the provided Jupyter notebook to train the model on a GPU. Ensure you have a GPU-enabled environment set up (e.g., Google Colab).
Run the Application:

7. **Execute the main script (.ipynb) to start the application**

8. **Using the Application:**
Select the input type (image, video, or live stream) and display the number of fingers to select a birdsong.
Use speech commands ("up" or "down") to control the volume of the playback.

## References

[Difference Between a Batch and an Epoch in Neural Networks](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
[Image Classification with Attention](https://blog.paperspace.com/image-classification-with-attention/)
[MediaPipe Hands](https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md)
[Loading All Images Using imread from a Given Folder](https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder)
[Hand Fingers Detection CNN Tensorflow Keras](https://github.com/chauhanmahavir/Hand-Fingers-Detection-CNN-Tensorflow-Keras/blob/master/Fingers_Detection_CNN_Tensorflow_Keras.ipynb)
[Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
[British Birdsong Dataset](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset)
[Selfie Finger Counting Dataset](https://www.kaggle.com/datasets/zakitaleb/selfie-finger-counting?resource=download)



