# Python_OpenCV_Emotion_Detection
To capture a moment based on the detected emotion on a human face.

Developing using a set of tutorials from the following link:

[Pual Van Gent](http://www.paulvangent.com/).

## TO DO
Test for noseBridgeAngleOffset is pending.

#### Current trainData representation
```
npArrTrainData[316L, 136L] => 316 images with 68 * 2 features i.e 68 mag and 68 ang per image.
npArrTrainLabels[316L, ] => labels corresponding to the input images.

npArrTestData.shape[78L, 136L]
npArrTestLabels.shape[78L, ] 
```

##### Mean value of prediction accuracy in 10 runs: 98.4615
