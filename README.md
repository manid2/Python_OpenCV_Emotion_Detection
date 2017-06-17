# Python_OpenCV_Emotion_Detection
To capture a moment based on the detected emotion on a human face.

Developing using a set of tutorials from the following link:

[Pual Van Gent](http://www.paulvangent.com/).

## TO DO
Calculate the prediction accuracy, experiment with feature extraction and representation.

#### Current trainData representation
```
npArrTrainData[316L, 136L] => 316 images with 68 * 2 features i.e 68 mag and 68 ang per image.
npArrTrainLabels[316L, ] => labels corresponding to the input images.

npArrTestData.shape[78L, 136L]
npArrTestLabels.shape[78L, ] 

svm.predict_all() => results.shape[78L, 1L]
```