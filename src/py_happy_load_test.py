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
import math
import dlib
import numpy as np
import datetime as dt

# Experimenting with the actual method used in the tutorial
__version__ = "2.0, 18/06/2017"
__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"

# Training happy against neutral.
emotionsList = ["happy", "surprise", "neutral"]

claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

frontalFaceDetector = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
    "..\\input\\shape_predictor_68_face_landmarks.dat")

# Constant factor to convert radians to degrees.
rad2degCnvtFactor = 180 / math.pi


def get_landmarks(claheImage):
    detectedFaces = frontalFaceDetector(claheImage, 1)

    # For all detected face instances extract the features
    for detectedFace in detectedFaces:
        
        # print "\nface found!\n"
        
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
        if xCoordinatesList[27] == xCoordinatesList[30]:
            noseBridgeAngleOffset = 0
            # radians1 = 1.5708 # 90 deg = 1.5708 rads
        else:           
            radians1 = math.atan(
                (yCoordinatesList[27] - yCoordinatesList[30]) /
                (xCoordinatesList[27] - xCoordinatesList[30]))
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
                radians2 = 1.5708 # 90 deg = 1.5708 rads
            else:
                radians2 = math.atan((ycoord - yCoordMean) / denom)
                
            pointAngle = (radians2 * rad2degCnvtFactor) - noseBridgeAngleOffset
            landmarkVectorList.append(pointDistance)
            landmarkVectorList.append(pointAngle)

    if len(detectedFaces) < 1:
        landmarkVectorList = "error"
    return landmarkVectorList

# Set the classifier as a opencv svm with SVM_LINEAR kernel
svm = cv2.SVM()
'''
svm_params = dict(
    kernel_type=cv2.SVM_LINEAR,
    svm_type=cv2.SVM_C_SVC,
    C=2.67,
    gamma=5.383)
'''

#################### Loading opencv SVM ####################
    
print "\n#################### Loading opencv SVM ####################\n"

# Load opencv SVM trained model.
svm.load("..\\input\\cv2_svm_happy_surprise.dat")
print "Loading opencv SVM model from file - Completed."

#################### Start Webcam ####################

video_capture = cv2.VideoCapture(0)  # Webcam object

print "\n#################### Starting Webcam ####################\n"

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_image = claheObject.apply(gray)
    landmarkVectorList = get_landmarks(clahe_image)
    
    if landmarkVectorList == "error":
        print "Feature extraction returns error!"
        break
    
    #################### Testing opencv SVM ####################
    
    # print "\n#################### Testing opencv SVM ####################\n"
    
    # Testing data must be float32 matrix for the opencv svm.    
    npArrTestData = np.float32(landmarkVectorList)    
    
    result = svm.predict(npArrTestData)
    
    #################### Print result ####################
    cv2.putText(frame, "You are {}.".format(emotionsList[int(result)]), (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1) 
    
    cv2.imshow("image", frame)  # Display the frame
    
    # Save the frame when the user presses 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = "..\\img_samples\\img_cap_{}.jpg".format(
            dt.datetime.today().strftime("%Y%m%d_%H%M%S"))
        cv2.imwrite(img_name, frame)
    
    # Exit program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

