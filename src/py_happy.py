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
# from sklearn.svm import SVC

# Experimenting with the actual method used in the tutorial
__version__ = "1.1, 01/06/2017"
__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"

# Complete emotions lists:
# emotionsList = ["neutral", "anger", "contempt", "disgust", "fear",
# "happy", "sadness", "surprise"]

# Training happy against neutral.
emotionsList = ["happy", "neutral"]

claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

frontalFaceDetector = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
    "..\\input\\shape_predictor_68_face_landmarks.dat")


# Define function to get file list, randomly shuffle it and split 80/20
def get_files(emotion):
    files = glob.glob("..\\dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


# Constant factor to convert radians to degrees.
rad2degCnvtFactor = 180 / math.pi

def get_landmarks(claheImage):
    detectedFaces = frontalFaceDetector(claheImage, 1)

    # For all detected face instances individually
    for faceCount, detectedFace in enumerate(detectedFaces):

        # Draw Facial Landmarks with the predictor class
        facialShape = facialShapePredictor(claheImage, detectedFace)

        xCoordinatesList = []
        yCoordinatesList = []

        # Store the X and Y coordinates of landmark points in two lists
        for i in range(0, 68):
            xCoordinatesList.append(facialShape.part(i).x)
            yCoordinatesList.append(facialShape.part(i).y)

        # Get the mean of both axes to determine centre of gravity
        xCoordMean = np.mean(xCoordinatesList)
        yCoordMean = np.mean(yCoordinatesList)

        '''
        # Mani - removing point coordinates distances
        # Get distance between each point and the central point in both axes
        xDistFromCentre = [(x - xCoordMean) for x in xCoordinatesList]
        yDistFromCentre = [(y - yCoordMean) for y in yCoordinatesList]
        '''

        # If x-coordinates of the set are the same, the angle is 0,
        # catch to prevent 'divide by 0' error in the function
        if xCoordinatesList[26] == xCoordinatesList[29]:
            noseBridgeAngleOffset = 0
        else:
            # noseBridgeAngleOffset = int(math.atan((yCoordinatesList[26]-yCoordinatesList[29])/
                        #                (xCoordinatesList[26]-xCoordinatesList[29]))*180/math.pi)
            radians1 = math.atan(
                (yCoordinatesList[26] - yCoordinatesList[29]) /
                (xCoordinatesList[26] - xCoordinatesList[29]))
            # since degrees = radians * rad2degCnvtFactor
            noseBridgeAngleOffset = int(radians1 * rad2degCnvtFactor)

        if noseBridgeAngleOffset < 0:
            noseBridgeAngleOffset += 90
        else:
            noseBridgeAngleOffset -= 90

        landmarkVectorList = []

        '''
        # Mani - removing point coordinates distances
        for xdist, ydist, xcoord, ycoord in zip(xDistFromCentre,  yDistFromCentre,
                        xCoordinatesList, yCoordinatesList):
        '''
        for xcoord, ycoord in zip(xCoordinatesList, yCoordinatesList):

            '''
            # Mani - removing point coordinates distances
            landmarkVectorList.append(xdist)
            landmarkVectorList.append(ydist)
            '''

            xyCoordArray = np.asarray((ycoord, xcoord))
            xyCoordMeanArray = np.asarray((yCoordMean, xCoordMean))

            pointDistance = np.linalg.norm(xyCoordArray - xyCoordMeanArray)
            
            # Prevent divide by zero error.
            denom = (xcoord - xCoordMean)
            if denom == 0:
                radians2 = 90
            else:
                radians2 = math.atan((ycoord - yCoordMean) / denom)
                
            pointAngle = (radians2 * rad2degCnvtFactor) - noseBridgeAngleOffset
            landmarkVectorList.append(pointDistance)
            landmarkVectorList.append(pointAngle)

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
            image = cv2.imread(item)  # read image
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

predictionAccuracy = []
for i in range(0, 1):

    # Make sets by random sampling 80/20%
    print "Making sets {0}".format(i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    #################### Training opencv SVM ####################
    
    print "\n#################### Training opencv SVM ####################\n"
    
    # Training data must be float32 matrix for the opencv svm.
    npArrTrainData = np.float32(training_data)
    npArrTrainLabels = np.float32(training_labels)

    print "npArrTrainData.shape = {0}.".format(npArrTrainData.shape)
    print "npArrTrainLabels.shape = {0}.".format(npArrTrainLabels.shape)
    
    print "Training opencv SVM linear {0} - Started.".format(i)
    svm.train(npArrTrainData, npArrTrainLabels, params=svm_params)    
    print "Training opencv SVM linear {0} - Completed.".format(i)
    
    # Save opencv SVM trained model.
    # svm.save('cv2_svm_happy_facial_landmarks.dat')    
    # print "Saving opencv SVM model to file - Completed."
    
    #################### Testing opencv SVM ####################
    
    print "\n#################### Testing opencv SVM ####################\n"
    
    # Testing data must be float32 matrix for the opencv svm.    
    npArrTestData = np.float32(prediction_data)
    npArrTestLabels = np.float32(prediction_labels)
        
    print "npArrTestData.shape = {0}.".format(npArrTestData.shape)
    print "npArrTestLabels.shape = {0}.".format(npArrTestLabels.shape)
    
    print "Testing opencv SVM linear {0} - Started.".format(i)
    results = svm.predict_all(npArrTestData)
    print "Testing opencv SVM linear {0} - Completed.".format(i)
    
    print "\n\t-> type(npArrTestLabels) = {}".format(type(npArrTestLabels))
    print "\t-> type(npArrTestLabels[0]) = {}".format(type(npArrTestLabels[0]))
    print "\t-> npArrTestLabels.size = {}".format(npArrTestLabels.size)
    
    print "\n\t-> type(results) = {}".format(type(results))
    print "\t-> type(results[0]) = {}".format(type(results[0]))
    print "\t-> type(results[0][0]) = {}".format(type(results[0][0]))
    print "\t-> results.size = {}, results.shape = {}".format(results.size, results.shape)

    #################### Check Accuracy ########################
    
    print "\n#################### Check Accuracy ########################\n"
    
    mask = results == npArrTestLabels
    correct = np.count_nonzero(mask)
    
    print "\t-> type(mask) = {}".format(type(mask))
    print "\t-> type(mask[0][0]) = {}".format(type(mask[0][0]))
    print "\t-> mask.size = {}, mask.shape = {}".format(mask.size, mask.shape)
    
    print "\nPrediction accuracy = %{0}.\n".format(correct*100.0/results.size)
    
    predictionAccuracy.append(correct)
    
# Get the mean accuracy of the i runs
print "Mean value of predict accuracy in five runs: {0:.4f}".format(np.mean(predictionAccuracy))

