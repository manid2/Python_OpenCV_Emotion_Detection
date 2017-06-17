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
    prediction = files[:int(len(files) * 0.4)]  # get first 40% of file list
    # training = files[-int(len(files) * 0.6):]  # get last 60% of file list
    return prediction


# Constant factor to convert radians to degrees.
rad2degCnvtFactor = 180 / math.pi


def get_landmarks(claheImage):
    detectedFaces = frontalFaceDetector(claheImage, 1)

    # For all detected face instances extract the features
    for detectedFace in detectedFaces:
        
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
    prediction_data = []
    prediction_labels = []
    
    for emotion in emotionsList:
        prediction = get_files(emotion)
        
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

    return prediction_data, prediction_labels


# Set the classifier as a opencv svm with SVM_LINEAR kernel
svm = cv2.SVM()
'''
svm_params = dict(
    kernel_type=cv2.SVM_LINEAR,
    svm_type=cv2.SVM_C_SVC,
    C=2.67,
    gamma=5.383)
'''

maxRuns = 4
runCount = 0
predictionAccuracyList = []

for runCount in range(0, maxRuns):

    # Get a set of samples for prediction
    print "\n\t\t<--- Making sets {0} --->".format(runCount)
    prediction_data, prediction_labels = make_sets()

    #################### Loading opencv SVM ####################
    
    print "\n#################### Loading opencv SVM ####################\n"
    
    # Load opencv SVM trained model.
    svm.load("..\\input\\cv2_svm_happy.dat")
    print "Loading opencv SVM model from file - Completed."
    
    #################### Testing opencv SVM ####################
    
    print "\n#################### Testing opencv SVM ####################\n"
    
    # Testing data must be float32 matrix for the opencv svm.    
    npArrTestData = np.float32(prediction_data)
    npArrTestLabels = np.float32(prediction_labels)
        
    print "npArrTestData.shape = {0}.".format(npArrTestData.shape)
    print "npArrTestLabels.shape = {0}.".format(npArrTestLabels.shape)
    
    print "Testing opencv SVM linear {0} - Started.".format(runCount)
    results = svm.predict_all(npArrTestData).reshape((-1,))
    print "Testing opencv SVM linear {0} - Completed.".format(runCount)
    
    print "\n\t-> type(npArrTestLabels) = {}".format(type(npArrTestLabels))
    print "\t-> type(npArrTestLabels[0]) = {}".format(type(npArrTestLabels[0]))
    print "\t-> npArrTestLabels.size = {}".format(npArrTestLabels.size)
    
    print "\n\t-> type(results) = {}".format(type(results))
    print "\t-> type(results[0]) = {}".format(type(results[0]))    
    print "\t-> results.size = {}, results.shape = {}".format(results.size, results.shape)
    
    #################### Check Accuracy ########################
    
    print "\n#################### Check Accuracy ########################\n"
    
    mask = results == npArrTestLabels
    correct = np.count_nonzero(mask)
    
    print "\t-> type(mask) = {}".format(type(mask))
    print "\t-> type(mask[0]) = {}".format(type(mask[0]))
    print "\t-> mask.size = {}, mask.shape = {}".format(mask.size, mask.shape)
    
    pred_accur = correct * 100.0 / results.size
    print "\nPrediction accuracy = %{0}.\n".format(pred_accur)
    print "---------------------------------------------------------------"
    
    predictionAccuracyList.append(pred_accur)
    
# Get the mean accuracy of the i runs
print "Mean value of predict accuracy in {0} runs: {1:.4f}".format(
    maxRuns, np.mean(predictionAccuracyList)) 
# sum(predictionAccuracyList) / len(predictionAccuracyList)
