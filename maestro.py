
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image
import time

from tensorflow.keras.models import load_model

def read_image(path):
    """ Method to read an image from file to matrix """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def plot_image(image, title=''):
    """ It plots an image as it is in a single column plot """
    # Plot our image using subplots to specify a size and title
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.set_title(title)
    ax1.imshow(image)
    while(True):
        image = cv2.resize(image, (800, 640))
        cv2.imshow("jaja",image)
        print("Mostrar imagen")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_faces(image):
    """
    It returns an array with the detected faces in an image
    Every face is defined as OpenCV does: top-left x, top-left y, width and height.
    """
    # To avoid overwriting
    image_copy = np.copy(image)
    
    # The filter works with grayscale images
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    # Extract the pre-trained face detector from an xml file
    face_classifier = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
    
    # Detect the faces in image
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)
    
    return faces 

def draw_faces(image, faces=None, plot=True):
    """
    It plots an image with its detected faces. If faces is None, it calculates the faces too
    """
    if faces is None:
        faces = get_faces(image)
    
    # To avoid overwriting
    image_with_faces = np.copy(image)
    
    # Get the bounding box for each detected face
    for (x,y,w,h) in faces:
        # Add a red bounding box to the detections image
        cv2.rectangle(image_with_faces, (x,y), (x+w,y+h), (255,0,0), 3)
        
        
    if plot is True:
        plot_image(image_with_faces)
    else:
        return image_with_faces
    
    

def plot_image_with_keypoints(image, image_info):
    """
    It plots keypoints given in (x,y) format
    """
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(111)
    
    for (face, keypoints) in image_info:
        for (x,y) in keypoints:
            ax1.scatter(x, y, marker='o', c='c', s=10)
 
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(image)
    

image = read_image('images/breaking_bad.jpg')
faces = get_faces(image)
print("Faces detected: {}".format(len(faces)))
draw_faces(image, faces)