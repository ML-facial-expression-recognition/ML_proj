import face_feature, get_train_data, training_data

import cv2
import numpy as np
import sys

# real time recognition from camera
def rt_rec():
    expression = ['neutral','joyful','sad','surprise','angry','disgusted','fearful']

    # get trained model
    feature_path = '../img/facesdb/feature.npy'
    strategy = 'ova' # {'ova', 'ovo'}
    classifier = 'svm' # {'svm', 'decisionTree','logistic','mlp'}
    clf, pca = training_data.training_data(feature_path, strategy, classifier)

    # get detector and predictor for getting face landmarks through camera
    detector = face_feature.get_detector()
    predictor = face_feature.get_predictor()

    # open camera!
    cam = cv2.VideoCapture(0)

    # get real time result
    while(True):
        _,img = cam.read()
        ft = np.asarray(face_feature.get_face_feature_from_camera(img,detector,predictor))
        if ft.shape[0] >0:
            for i in range(ft.shape[0]):
                print(ft.shape)
                print(np.mat(ft[i,:]).shape)
                res = clf.predict(np.mat(ft[i,:]).A)
                img = cv2.putText(img, expression[int(res)],(40,40),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),2)

        cv2.imshow("camera", img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    rt_rec()