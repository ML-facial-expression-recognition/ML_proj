import face_feature, get_train_data, training_data,race_feature

import cv2
import numpy as np
import sys

# real time recognition from camera
def rt_rec():
    expression = ['neutral', 'joyful', 'sad', 'surprise', 'angry', 'disgusted', 'fearful']
    emoji_paths = ['../assets/neutral.png', '../assets/joyful.png', '../assets/sad.png', '../assets/surprise.png', '../assets/angry.png', '../assets/disgusted.png', '../assets/fearful.png']
    raceclass = ['Asian', 'Black', 'Latin','White']
    # get race trained model
    race_feature_path = '../img/race/feature.npy'
    race_strategy = 'ova' # {'ova', 'ovo'}
    race_classifier = 'decisionTree' # {'svm', 'decisionTree','logistic','mlp'}
    race_clf, race_pca = training_data.training_data(race_feature_path,race_strategy,race_classifier)
    # get trained model
    feature_path = '../img/facesdb/feature.npy'
    strategy = 'ova' # {'ova', 'ovo'}
    classifier = 'logistic' # {'svm', 'decisionTree','logistic','mlp'}
    clf, pca = training_data.training_data(feature_path, strategy, classifier)


    # get detector and predictor for getting face landmarks through camera
    detector = face_feature.get_detector()
    predictor = face_feature.get_predictor()

    # open camera!
    cam = cv2.VideoCapture(0)

    # get real time result
    while(True):
        _,img = cam.read()
        x_offset = 100
        y_offset = 250
        # face_ft, race_ft = face_feature.get_all_feature_from_camera(img,detector,predictor)
        race_ft = np.asarray(face_feature.get_all_feature_from_camera(img,detector,predictor)[1])
        face_ft = np.asarray(face_feature.get_all_feature_from_camera(img, detector, predictor)[0])
        if race_ft.shape[0] > 0:
            for i in range(race_ft.shape[0]):
                race_res = race_clf.predict(np.mat(race_ft[i,:]).A)
                face_res = clf.predict(np.mat(face_ft[i,:]).A)
                emoji = cv2.imread(emoji_paths[int(face_res)])
                emoji = cv2.resize(emoji, (int(150), int(150)))

                img = cv2.putText(img, raceclass[int(race_res)],(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),4)
                img[y_offset: y_offset + emoji.shape[0], x_offset: x_offset + emoji.shape[1]] = emoji

        cv2.imshow("camera", img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    rt_rec()