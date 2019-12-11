#  Racial and Facial expression recognition

## 1. Project proposal 
We will train a facial recognition system from scratch 
which will finally recognize the emotion by analysing images/videos. 
Our system will attempt to recognize six basic emotional expressions 
including fear, disgust, anger, surprise, happiness, and sadness 
introduced by Ekman<sup>[1]</sup>. 

Because the racial effect will occur in the process of facial 
recognition. We will also try to figure out how to detect races 
which may reduce the influence of the effect.

To make this system more interesting, we plan to transform the 
final result from the plain words into emoji. This means that 
the output will be an emoji that can represent the emotion and 
the race of the human.There may be other human-related information 
we can detect from static images and video sequences, like genders, 
ages, hair colors and so on. Facial expression and race will be 
the most important feature that we will focus on.

## 2. Program Implementation

### 2.1 Prepocessing

#### 2.1.1 get_train_data

a. tranverse all images

b. save features into feature.npy file, and format as below:
    | label | features |
    |   0   |   ...    |
    |   1   |   ...    |
    |  ...  |   ...    |
            ⬇️
    label : multiclass {0,1,...};
    features : get_face_feature in face_feature.py

#### 2.1.2 face_feature

a. get all features from path or camera

b. The fuction get_detector() and get_predictor() are dependent on 
dlib library

c. help function used on reading images from path

d.help function used on reading images from camera

#### 2.1.3 race_feature

a. extract features of different skin colors from camera capture

b. capture features from camera frame

### 2.2 Training Data

#### 2.2.1 get_train_data

a. get training data features from face_feature()

b. get race data features from race_feature()

#### 2.2.2 training_data

a. training data according to the input strategies

b. calculating error rate on testing data according to 
classifier and PCA method

c. randomly choose 5/6 as training data and rest 1/6 as 
testing data

### 2.3 Main functino: real time recognition from camera

#### 2.3.1 rt.rec

a. get trained model

b. get real time result from camera

## Reference
Documents of ML project: facial expression detection

1.database source: 
   http://app.visgraf.impa.br/database/faces
    
   CFD ver2.0.3: https://chicagofaces.org/     
    

2.face feature: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/


3.*Ekman P., Friesen W.V., Ellsworth P.Emotion in the Human Face: Guide-Lines 
for Research and an Integration of Findings: Guidelines for Research and an 
Integration of Findings.Pergamon; Berlin, Germany: 1972.*
