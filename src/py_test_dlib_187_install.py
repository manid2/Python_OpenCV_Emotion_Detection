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
import datetime as dt

# No need to modify this one as it is a helper script.
__version__ = "1.0, 17/05/2017"
__author__ = "Paul van Gent - 2016"

# Set up some required objects
video_capture = cv2.VideoCapture(0)  # Webcam object
detector = dlib.get_frontal_face_detector()  # Face detector

# Landmark identifier. Set the filename to whatever you named the
# downloaded file
predictor = dlib.shape_predictor("..\\input\\shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1)  # Detect the faces in the image

    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            # For each point, draw circle with thickness = 2 on the original frame
            if i == 27 or i == 30:
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                       1, (0, 255, 0), thickness=2)
                cv2.putText(frame, "P{}".format(i), (shape.part(i).x, shape.part(i).y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1)      
            else:
                # pass                
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y),
                       1, (0, 0, 255), thickness=2)
                

    cv2.imshow("image", frame)  # Display the frame
    
    # Save the frame when the user presses 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = "..\\img_samples\\img_cap_{}.jpg".format(
            dt.datetime.today().strftime("%Y%m%d_%H%M%S"))
        cv2.imwrite(img_name, frame)
    
    # Exit program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
