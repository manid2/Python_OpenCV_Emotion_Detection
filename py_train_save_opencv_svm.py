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
import itertools
from sklearn.svm import SVC

# Experimenting with the actual method used in the tutorial
__version__ = "1.1, 25/05/2017"
__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"

# Complete emotions lists:
# emotionsList = ["neutral", "anger", "contempt", "disgust", "fear",
# "happy", "sadness", "surprise"]

# Most important - positive emotions:
emotionsList = ["happy", "surprise"]

claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

frontalFaceDetector = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
                       "..\\shape_predictor_68_face_landmarks.dat")


# Define function to get file list, randomly shuffle it and split 80/20
def get_files(emotion):
    files = glob.glob("..\\dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


# Constant factor to convert radians to degrees.
rad2degConvtFactor = 180 / math.pi


def get_landmarks(claheImage):
    detectedFaces = frontalFaceDetector(claheImage, 1)

    # For all detected face instances individually
    for faceCount, detectedFace in enumerate(detectedFaces):

        # Draw Facial Landmarks with the predictor class
        facialShape = facialShapePredictor(claheImage, detectedFace)
		
	xCoordinatesList = []
        yCoordinatesList = []

        # Store the X and Y coordinates of landmark points in two lists
        for i in range(1, 68):
            xCoordinatesList.append(float(facialShape.part(i).x))
            yCoordinatesList.append(float(facialShape.part(i).y))

        # Get the mean of both axes to determine centre of gravity
        xCoordMean = np.mean(xCoordinatesList)
        yCoordMean = np.mean(yCoordinatesList)

        # Get distance between each point and the central point in both axes
        xDistFromCentre = [(x - xCoordMean) for x in xCoordinatesList]
        yDistFromCentre = [(y - yCoordMean) for y in yCoordinatesList]

        # If x-coordinates of the set are the same, the angle is 0,
        # catch to prevent 'divide by 0' error in the function
        if xCoordinatesList[26] == xCoordinatesList[29]:
            anglenose = 0
        else:
            # anglenose = int(math.atan((yCoordinatesList[26]-yCoordinatesList[29])/
			#                (xCoordinatesList[26]-xCoordinatesList[29]))*180/math.pi)
            radians1 = math.atan(
                       (yCoordinatesList[26] - yCoordinatesList[29]) /
                       (xCoordinatesList[26] - xCoordinatesList[29]))
            # since degrees = radians * rad2degConvtFactor
            anglenose = int(radians1 * rad2degConvtFactor)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90
		
	landmarkVectorList = []
        for xdist, ydist, xcoord, ycoord in zip(xDistFromCentre,  yDistFromCentre,							   
						xCoordinatesList, yCoordinatesList):							  
            landmarkVectorList.append(xdist)
            landmarkVectorList.append(ydist)
            			
            xyCoordArray = np.asarray((ycoord, xcoord))
	    xyCoordMeanArray = np.asarray((yCoordMean, xCoordMean))
			
            dist = np.linalg.norm(xyCoordArray - xyCoordMeanArray)
            radians2 = math.atan((ycoord - yCoordMean) / (xcoord - xCoordMean))
            anglerelative = radians2 * rad2degConvtFactor - anglenose
            landmarkVectorList.append(dist)
            landmarkVectorList.append(anglerelative)

    if len(detectedFaces) < 1:
        landmarkVectorList = "error"
    return landmarkVectorList


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotionsList:
        training, prediction = get_files(emotion)

        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = claheObject.apply(gray)
            landmarkVectorList = get_landmarks(clahe_image)
            if landmarkVectorList == "error":
                pass
            else:
                # Append image array to training data list
                training_data.append(landmarkVectorList)
                training_labels.append(emotionsList.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = claheObject.apply(gray)
            landmarkVectorList = get_landmarks(clahe_image)
            if landmarkVectorList == "error":
                pass
            else:
                prediction_data.append(landmarkVectorList)
                prediction_labels.append(emotionsList.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


# Set the classifier as a opencv svm with SVM_LINEAR kernel
svm = cv2.SVM()
svm_params = dict(
    kernel_type=cv2.SVM_LINEAR,
    svm_type=cv2.SVM_C_SVC,
    C=2.67,
    gamma=5.383)

accur_lin = []
for i in range(0, 1):
    # Make sets by random sampling 80/20%
    print("Making sets %s" % i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    # Turn the training set into a numpy array for the classifier
    npar_train = np.array(training_data)
    npar_trainlabs = np.array(training_labels)

    # Train opencv SVM here.
    print("training SVM linear %s" % i)
    # clf.fit(npar_train, training_labels)
    svm.train(npar_train, npar_trainlabs, params=svm_params)
    svm.save('opencv_facial_landmarks.dat')

    '''
    # Use score() function to get accuracy
    print("getting accuracies %s" % i)
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print "linear: ", pred_lin
    '''

    # Store accuracy in a list
    accur_lin.append(pred_lin)

# Get mean accuracy of the i runs
print("Mean value lin svm: %.3f" % np.mean(accur_lin))
