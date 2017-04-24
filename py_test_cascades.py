import cv2
import numpy as np

#List of emotions to detect
emotions = ["happy", "surprise"]

#font used to add text to the detected face box
font = cv2.FONT_HERSHEY_SIMPLEX

#The default webcam (0, the first webcam device detected).
video_capture = cv2.VideoCapture(0)

#Create CLAHE - (Contrast Limited Adaptive Histogram Equalization) object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#load the trained face haar cascade model
haar_files = "..\\haarcascade_face_detection\\"
facecascade = cv2.CascadeClassifier(haar_files + "haarcascade_frontalface_default.xml")

#load the trained emotion classifier model
fishface = cv2.createFisherFaceRecognizer()
fishface.load("fishface_happy_surprise.xml")

#Create face dictionary
facedict = {}

#Crop the given face
def crop_face(clahe_image, face):
	for (x, y, w, h) in face:
		faceslice = clahe_image[y:y+h, x:x+w]
		faceslice = cv2.resize(faceslice, (350, 350))
		#append sliced face as a numbered face to the dictionary
	facedict["face%s" %(len(facedict)+1)] = faceslice
	return faceslice

def recognize_emotion():
	predictions = []
	confidence = []
	for x in facedict.keys():
		pred, conf = fishface.predict(facedict[x]) 
		predictions.append(pred)
		confidence.append(conf)
        print "max(): %s" %max(set(predictions), key=predictions.count)
        print("The detected emotion is: %s" %emotions[max(set(predictions), key=predictions.count)])

while True:
	#Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
	ret, frame = video_capture.read()  
	
	#Convert image to grayscale to improve detection speed and accuracy
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#Apply CLAHE to grayscale image from webcam
	clahe_image = clahe.apply(gray)
	
	#Run classifier on frame
	face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
	
	#Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
	if len(face) == 1:
		faceslice = crop_face(clahe_image, face) #slice face from image
	
	#Run the emotion recognizer when 10 faces are collected
	if len(facedict) == 10:
		recognize_emotion()
		# break

	#Draw rectangle around detected faces
	for (x, y, w, h) in face:
		#draw it on the colour image "frame", with arguments: (coordinates), (size), (RGB color), line thickness 2
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		cv2.putText(frame, 'Face detected!', (x, y), font, 1, (0,255,0), 2, cv2.CV_AA)

	#Display frame
	cv2.imshow("Face Detection", frame)
	
	#imshow expects a termination definition in order to work correctly, here it is bound to key 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
