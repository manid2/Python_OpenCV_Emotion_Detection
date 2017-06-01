# Python_OpenCV_Emotion_Detection
To capture a moment based on the detected emotion on a human face.

Developing using a set of tutorials from the following link:

[Pual Van Gent](http://www.paulvangent.com/).

## TO DO
### Corrected the error:
"OpenCV Error: Bad argument (train data must be floating-point matrix) in cvCheckTrainData"

### Solution:
Use "trainingData.convertTo(trainingData, CvType.CV_32FC1)"

### Pending - representation of trainData
Four ways to represent the trainData:
1. trainData [68] [3] = {x1, y1, ang1,
                         x2, y2, ang2,
                         ...
                         x68, y68, ang68
                         };
2. trainData [3] [68] = { x1, x2, x3, ... , x68,
                          y1, y2, u3, ... , y68,
                          ang1, ang2, ang3, ... , ang68
                        };
3. trainData [68] [2] = { euclPxyCenterDist1, euclPxyCenterDist2, ... euclPxyCenterDist68,
                          ang1, ang2, ang3, ... , ang68
                        };
4. trainData [2] [68] = { euclPxyCenterDist1, ang1,
                          euclPxyCenterDist2, ang2,
                          ...
                          euclPxyCenterDist68, ang68
                        };


