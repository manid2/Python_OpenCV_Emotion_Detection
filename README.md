# Python_OpenCV_Emotion_Detection
To capture a moment based on the detected emotion on a human face.

Developing using a set of tutorials from the following link:

[Pual Van Gent](http://www.paulvangent.com/).

## TO DO
1. Keep track of each face in the video/image and display their emotional state.
2. Monitor the facial state information when no face is detected.

#### Current trainData representation
```
Training Data
-------------
454 images with 68 * 2 features i.e 68 mag and 68 ang per image.
npArrTrainData.shape = (454L, 136L).

labels corresponding to the input images.
npArrTrainLabels.shape = (454L,).

Testing Data
------------
111 images with 68 * 2 features i.e 68 mag and 68 ang per image.
npArrTestData.shape = (111L, 136L).
npArrTestLabels.shape = (111L,).
```

### Prediction accuracy = 69.3694 % for 6 emotional states.
