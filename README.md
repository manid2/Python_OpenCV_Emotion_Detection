# Python_OpenCV_Emotion_Detection
To capture a moment based on the detected emotion on a human face.

Developing using a set of tutorials from the following link:

[Pual Van Gent](http://www.paulvangent.com/).

## TO DO
### New error encountered:
```
Making sets 0
-------------------------------------------------------------------------------------------------
Traceback (most recent call last):
----------------------------------
  File "C:\Users\...\src\py_train_save_cv2_svm_happy.py", line 183, in <module>
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
  File "C:\Users\...\src\py_train_save_cv2_svm_happy.py", line 149, in make_sets
    landmarkVectorList = get_landmarks(clahe_image)
  File "C:\Users\...\src\py_train_save_cv2_svm_happy.py", line 61, in get_landmarks
    facialShape = facialShapePredictor(claheImage, detectedFace)
--------------------------------------------------------------------------------------------------
TypeError: No registered converter was able to produce a C++ rvalue of type class dlib::rectangle 
from this Python object of type tuple
--------------------------------------------------------------------------------------------------
```

#### Solution:
Yet to figure out.

#### Current trainData representation
```
npArrTrainData = np.float32(training_data).reshape(-1, 68)
npArrTrainLabels = np.float32(training_labels)
```
