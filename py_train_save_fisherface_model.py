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
import random
import numpy as np

# Modified to save the fisherface model trained on two emotions.
__version__ = "1.1, 17/05/2017"
__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"

# Complete emotions lists:
#emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Most important - positive emotions:
emotions = ["happy", "surprise"]

fishface = cv2.createFisherFaceRecognizer()  # Initialize fisher face classifier

data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


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
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # append image array to training data list
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    print "\ttraining fisher face classifier"
    print "\tsize of training set is:", len(training_labels), "images"
    fishface.train(training_data, np.asarray(training_labels))

    # print "\t--> Saving the trained model to xml file"
    # fishface.save("fishface_happy_surprise.xml")

    print "\tpredicting classification set"
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
            # print "\t\t--> Correct prediction, emotion:", pred
        else:
            incorrect += 1
            cnt += 1
    return ((100 * correct) / (correct + incorrect))


# Now run it
metascore = []
for i in range(0, 1):
    correct = run_recognizer()
    print "Iteration:", i, "--> got", correct, "percent correct!"
    metascore.append(correct)

print "\n\nend score:", np.mean(metascore), "percent correct!"
