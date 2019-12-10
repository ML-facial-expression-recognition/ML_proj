import cv2
import dlib
import numpy as np
from imutils import face_utils

def get_detector():
    detector = dlib.get_frontal_face_detector()
    return detector

def get_predictor():
    p = "../img/face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
    return predictor

def get_face_feature(path, detector, predictor):
    # get the image
    img = cv2.imread(path)
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

        # square the faces and normalize the feature
        norm = np.amax(fts, 0) - np.amin(fts, 0)
        fts_norm = (fts - np.amin(fts, 0)) / norm
        fts_ret = fts_norm.reshape((1, fts_norm.shape[0] * fts_norm.shape[1]), order='A').tolist()[0]
        res.append(fts_ret)

    # display result
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    # print(fts_norm)
    return res

def get_face_feature_from_camera(img,detector,predictor):
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

        # square the faces and normalize the feature
        norm = np.amax(fts, 0) - np.amin(fts, 0)
        fts_norm = (fts - np.amin(fts, 0)) / norm
        fts_ret = fts_norm.reshape((1, fts_norm.shape[0] * fts_norm.shape[1]), order='A').tolist()[0]
        res.append(fts_ret)

    # display result
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    # print(fts_norm)
    return res


def face_feature(path):
    # set the predictor and detector
    detector = get_detector()
    predictor = get_predictor()

    # get feature
    return get_face_feature(path,detector,predictor)

if __name__ == '__main__':
    path = "../img/face1.jpg"
    fts = face_feature(path)
    # detector = get_detector()
    # predictor = get_predictor()
    # fts = get_face_feature(path, detector, predictor)
    print(fts)