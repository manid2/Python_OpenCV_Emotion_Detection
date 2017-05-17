"""
Mani experimenting with facial information extraction.
@purpose:      To extract all possible information from an image
               and present it in json or xml format for further processing.
@applications: 1. Enhancing the multiple object detection in Computer Vision field.
               2. Capturing a moment in the time based on the extracted information
                  and applying auto filters to enhace the image.
@Based on: <a href="http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/">
              Emotion Recognition With Python, OpenCV and a Face Dataset
           </a>
"""

import cv2
import glob

# No need modify this one as it is a helper script.
__version__ = "1.0, 01/04/2016"
__author__ = "Paul van Gent, 2016"

haar_files = "haarcascade_face_detection\\"

faceDet1 = cv2.CascadeClassifier(
    haar_files + "haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier(
    haar_files + "haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier(
    haar_files + "haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier(
    haar_files + "haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]  # Define emotions


def detect_faces(emotion):
    # Get list of all images with emotion
    files = glob.glob("sorted_set\\%s\\*" % emotion)

    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face using 4 different classifiers
        face1 = faceDet1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
            5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
            5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
            5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(
            5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        # Go over detected faces, stop at first detected face, return empty if
        # no face.
        if len(face1) == 1:
            facefeatures = face1
        elif len(face2) == 1:
            facefeatures = face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""

        # Cut and save face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            print "face found in file: %s" % f
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                # Resize face so all images have same size
                out = cv2.resize(gray, (350, 350))
                cv2.imwrite("dataset\\%s\\%s.jpg" %
                            (emotion, filenumber), out)  # Write image
            except:
                pass  # If error, pass file
        filenumber += 1  # Increment image number


for emotion in emotions:
    detect_faces(emotion)  # Call functiona
