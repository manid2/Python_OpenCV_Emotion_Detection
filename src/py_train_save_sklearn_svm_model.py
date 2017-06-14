"""
Mani experimenting with facial information extraction.

@purpose:      To extract all possible information from an image
               and present it in json or xml format for further processing.
@applications: 1. Enhancing the multiple object detection in Computer Vision field.
               2. Capturing a moment in the time based on the extracted information
                  and applying auto filters to enhace the image.
@based on: <a href="http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/">
              Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV
           </a>
"""

import cv2
import glob
import random
import math
import numpy as np
import dlib
# import itertools
from sklearn.svm import SVC

# Experimenting with the actual method used in the tutorial
__version__ = "1.0, 17/05/2017"
__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"

# Complete emotions lists:
#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Most important - positive emotions:
emotions = ["happy", "surprise"]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("..\\input\\shape_predictor_68_face_landmarks.dat")

# Set the classifier as a support vector machines with polynomial kernel
clf = SVC(kernel='linear', probability=True, tol=1e-3)

# Define function to get file list, randomly shuffle it and split 80/20


def get_files(emotion):
    files = glob.glob("..\\dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


# Mani refactoring to create the constant 180 / pi
# START
rad2deg_const_factor = 180 / math.pi
# END - Mani refactor


def get_landmarks(image):
    detections = detector(image, 1)

    # For all detected face instances individually
    for d in enumerate(detections):
        # Draw Facial Landmarks with the predictor class\
        shape = predictor(image, d)
        xlist = []
        ylist = []

        # Store X and Y coordinates in two lists
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        # Get the mean of both axes to determine centre of gravity
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)

        # Get distance between each point and the central point in both axes
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        # If x-coordinates of the set are the same, the angle is 0,
        # catch to prevent 'divide by 0' error in function
        if xlist[26] == xlist[29]:
            anglenose = 0
        else:
            # Mani refactoring to enhance the readability and simplify the equation.
            # START
            # anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)
            radians1 = math.atan((ylist[26] - ylist[29]) /
                                 (xlist[26] - xlist[29]))
            # since degrees = radians * rad2deg_const_factor
            anglenose = int(radians1 * rad2deg_const_factor)
            # END - Mani refactor

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            radians2 = math.atan((z - ymean) / (w - xmean))
            anglerelative = radians2 * rad2deg_const_factor - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1:
        landmarks_vectorised = "error"
    return landmarks_vectorised


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)

        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                # Append image array to training data list
                training_data.append(landmarks_vectorised)
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []
for i in range(0, 5):
    # Make sets by random sampling 80/20%
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    # Turn the training set into a numpy array for the classifier
    npar_train = np.array(training_data)
    npar_trainlabs = np.array(training_labels)

    # Train SVM
    print("training SVM linear %s" % i)
    clf.fit(npar_train, training_labels)

    # Use score() function to get accuracy
    print("getting accuracies %s" % i)
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print "linear: ", pred_lin

    # Store accuracy in a list
    accur_lin.append(pred_lin)

# Get mean accuracy of the i runs
print("Mean value lin svm: %.3f" % np.mean(accur_lin))
