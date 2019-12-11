import face_feature, get_train_data, training_data

import cv2
import numpy as np
import sys

# real time recognition from camera
def rt_rec():
    expression = ['neutral', 'joyful', 'sad', 'surprise', 'angry', 'disgusted', 'fearful']
    emoji_paths = ['../assets/neutral.png', '../assets/joyful.png', '../assets/sad.png', '../assets/surprise.png', '../assets/angry.png', '../assets/disgusted.png', '../assets/fearful.png']

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
        x_offset = 100
        y_offset = 250
        ft = np.asarray(face_feature.get_face_feature_from_camera(img,detector,predictor))

        if ft.shape[0] > 0:
            for i in range(ft.shape[0]):
                res = clf.predict(np.mat(ft[i,:]).A)

                emoji = cv2.imread(emoji_paths[int(res)])
                emoji = cv2.resize(emoji, (int(150), int(150)))

                img = cv2.putText(img, expression[int(res)],(100,200),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),4)
                img[y_offset: y_offset + emoji.shape[0], x_offset: x_offset + emoji.shape[1]] = emoji

        cv2.imshow("camera", img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    rt_rec()