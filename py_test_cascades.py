import cv2
import numpy as np

#The default webcam (0, the first webcam device detected).
#Change if you have more than one webcam connected and 
#want to use another one than the default one

video_capture = cv2.VideoCapture(0)

haar_files = "haarcascade_face_detection\\"
#load the trained classifier model
facecascade = cv2.CascadeClassifier(haar_files + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

#Create CLAHE - (Contrast Limited Adaptive Histogram Equalization) object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

fishface = cv2.createFisherFaceRecognizer()
#load the trained emotion classifier model
fishface.load("fishface_happy_surprise.xml")

facedict = {} #Create face dictionary

def crop_face(gray, face): #Crop the given face
    for (x, y, w, h) in face:
        faceslice = gray[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
	#append sliced face as a numbered face to the dictionary
    facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

while True:
	#Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
    ret, frame = video_capture.read()
    
	#Convert image to grayscale to improve detection speed and accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#Apply CLAHE to grayscale image from webcam
    clahe_image = clahe.apply(gray)

    #Run classifier on frame
    face = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in face: #Draw rectangle around detected faces
	    #draw it on the colour image "frame", with arguments: (coordinates), (size), (RGB color), line thickness 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face detected!', (x, y), font, 1, (0,255,0), 2, cv2.CV_AA)

    cv2.imshow("Face Detection", frame) #Display frame
	
	#Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
    if len(face) == 1:
        faceslice = crop_face(gray, face) #slice face from image
        cv2.imshow("face_slice", faceslice) #display sliced face

    if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition in order to work correctly, here it is bound to key 'q'
        break