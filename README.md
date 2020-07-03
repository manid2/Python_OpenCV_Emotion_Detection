# Python_OpenCV_Emotion_Detection

To capture a moment based on the detected emotion on a human face.\
Developing using a set of tutorials from the following link:\
[Pual Van Gent](http://www.paulvangent.com/).

## TODO

1. Keep track of each face in the video/image and display their emotional state.
2. Monitor the facial state information when no face is detected.
3. Remove `contempt`.
4. Add `pout` and `disgust` faces.

## Current trainData representation

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

> *Prediction accuracy = 69.3694 % for 6 emotional states.*

## Update 2020-07-04_01:44:32

I am quite surprised that this project has become very popular in my profile.\
Since it is my original work it deserves to be popular. I am glad people found\
this very useful.

For this reason I am updating this repo with useful links to related projects.

* [Facial Features Recognizer][FFR_repo] -  Recognizes facial features such as\
  gender, age and emotion
* [Train Opencv 4 SVM][FFR_blog] - Discuss how to train a model of SVM in C++
* [Python utility package][PyUtilPkg] - Not so good scripts for image processing

### Note

Due to lack of expertise in git I had committed image and video data\
used in the project. Removed them in the master branch tip but retained in the\
git history. Please checkout older commits to access them if desired.

Currently not working in any Python, ML or Image processing projects.\
My primary skills are in C++ and Linux development.\
Hence not contributing to this repo until I can make time for fun projects.
:sweat_smile:

<!-- links -->

[FFR_repo]: https://github.com/manid2/FacialFeaturesRecognizer
[FFR_blog]: https://manid2.github.io/tutorials/how-to-train-opencv-4-svm/
[PyUtilPkg]: https://github.com/manid2/PyImageProcUility
