import face_feature

import cv2
import dlib
import numpy as np

def get_race_feature(path, detector, predictor):
    img = cv2.imread(path)
    res = get_race_feature_from_camera(img, detector, predictor)

    return res

def get_race_feature_from_camera(img, detector, predictor):
    # get feature of the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detect faces
    faces = detector(gray_img, 0)

    res = []
    # get features
    for (i, face) in enumerate(faces):
        shape = predictor(gray_img, face)
        fts = face_utils.shape_to_np(shape)
        # for (x,y) in fts:
        #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # square the faces and normalize the landmark feature
        norm = np.amax(fts, 0) - np.amin(fts, 0)
        fts_norm = (fts - np.amin(fts, 0)) / norm
        fts_ret = fts_norm.reshape((1, fts_norm.shape[0] * fts_norm.shape[1]), order='A').tolist()[0]
        res.append(fts_ret)

    return res

def race_feature(path):
    detector = face_feature.get_detector()
    predictor = face_feature.get_predictor()

    return get_race_feature(path, detector, predictor)


if __name__ == '__main__':
    path = '../img/face1.jpg'
    fts = race_feature(path)

