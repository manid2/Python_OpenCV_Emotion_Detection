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

import glob
from shutil import copyfile

# No need to modify this one as it is a helper script.
__version__ = "1.0, 01/04/2016"
__author__ = "Paul van Gent, 2016"

# Define emotion order
emotions = ["neutral", "anger", "contempt", "disgust",
            "fear", "happy", "sadness", "surprise"]

# Returns a list of all folders with participant numbers
participants = glob.glob("source_emotions\\*")

# i = 1;

for x in participants:

    # i += 1;
    # store current participant number
    part = "%s" % x[-4:]

    # Store list of sessions for current participant
    for sessions in glob.glob("%s\\*" % x):
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            # Emotions are encoded as a float, readline as float,
            # then convert to integer.
            emotion = int(float(file.readline()))

            # get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob(
                "source_images\\%s\\%s\\*" % (part, current_session))[-1]

            # do same for neutral image
            sourcefile_neutral = glob.glob(
                "source_images\\%s\\%s\\*" % (part, current_session))[0]

            # Generate path to put neutral image
            dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]

            # Do same for emotion containing image
            dest_emot = "sorted_set\\%s\\%s" % (
                emotions[emotion], sourcefile_emotion[25:])

            # Copy file
            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)

    # if i == 10:
            # break;
