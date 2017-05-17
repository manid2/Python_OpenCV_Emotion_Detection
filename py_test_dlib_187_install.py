"""
Mani experimenting with facial information extraction.
@purpose:      To extract all possible information from an image
               and present it in json or xml format for further processing.
@applications: 1. Enhancing the multiple object detection in Computer Vision field.
               2. Capturing a moment in the time based on the extracted information
                  and applying auto filters to enhace the image.
@Based on: <a href="http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/">
              Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV
           </a>
"""

import cv2
import dlib

# No need to modify this one as it is a helper script.
__version__ = "1.0, 17/05/2017"
__author__ = "Paul van Gent - 2016"

# Set up some required objects
video_capture = cv2.VideoCapture(0)  # Webcam object
detector = dlib.get_frontal_face_detector()  # Face detector

# Landmark identifier. Set the filename to whatever you named the
# downloaded file
predictor = dlib.shape_predictor("..\\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1)  # Detect the faces in the image

    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                       1, (0, 0, 255), thickness=2)
            # For each point, draw a red circle with thickness2 on the original
            # frame

    cv2.imshow("image", frame)  # Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit program when the user presses 'q'
        break
