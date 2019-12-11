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
    res = get_face_feature_from_camera(img, detector, predictor)

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
        print(fts.shape)
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

def S(a,b,c):
    s = (a[0]-c[0]) * (b[1]-c[1]) - (a[1]-c[1]) * (b[0]-c[0])
    if s != 0:
        s = int(s/abs(s))
    return s

def deter_in_tri(x,y,vers):
    # determine if the point is in the triangle with three vertexes in vers
    # get the result by determine whether the point is on the same side the with each point of the opposite arg
    for i in range(3):
        ab = vers.copy()
        c = ab[i]
        ab.pop(i)
        if S(ab[0],ab[1],c)*S(ab[0],ab[1],[x,y])<0:
            return 0

    return 1

def get_pts_in_tri(fts,tri):
    pts = []

    vers = np.concatenate((np.asarray([fts[tri[0]-1,:]]),np.asarray([fts[tri[1]-1,:]]),np.asarray([fts[tri[2]-1,:]])),axis = 0)
    xmin = np.min(vers,0)[0]
    xmax = np.max(vers,0)[0]
    ymin = np.min(vers,0)[1]
    ymax = np.max(vers,0)[1]
    vers = vers.tolist()

    # check if pts are in the triangle
    for x in range(xmin,xmax+1):
        for y in range(ymin,ymax+1):
            if deter_in_tri(x,y,vers) == 1:
                pts.append([x,y])

    return pts

def get_hsv_in_tri(img, pts):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    num = len(pts)
    h = 0
    s = 0
    v = 0

    for pt in pts:
        h = h + hsv_img[pt[0],pt[1],0]
        s = s + hsv_img[pt[0],pt[1],1]
        v = v + hsv_img[pt[0],pt[1],2]

    return [h/num,s/num,v/num]

def get_all_feature_from_camera(img, detector, predictor):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray_img, 0)

    res_landmark = []
    res_race = []
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
        res_landmark.append(fts_ret)

        # get skin color feature
        std_landmark = [[37,38,42],[38,39,42],[38,39,41],[39,40,41],[2,29,30],[2,30,31],[2,3,32],[3,4,32]]
        res_hsv = []
        for tri in std_landmark:
            # find a list of points that located inside the triangle
            pts = get_pts_in_tri(fts, tri)

            # find average hsv color space of points inside triangles
            hsv = get_hsv_in_tri(img, pts)

            for i in hsv:
                res_hsv.append(i)

        res_race.append(res_hsv)

    # display result
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    # print(fts_norm)
    return res_landmark, res_race

def get_all_feature(path,detector,predictor):
    img = cv2.imread(path)
    return get_all_feature_from_camera(img, detector, predictor)



def face_feature(path):
    # set the predictor and detector
    detector = get_detector()
    predictor = get_predictor()

    # get feature
    return get_face_feature(path,detector,predictor)

def all_face_feature(path):
    detector = get_detector()
    predictor = get_predictor()

    return get_all_feature(path,detector,predictor)

if __name__ == '__main__':
    path = "../img/face1.jpg"
    ftlm,ftr = all_face_feature(path)
    # detector = get_detector()
    # predictor = get_predictor()
    # fts = get_face_feature(path, detector, predictor)
    print(len(ftlm),len(ftr))
    print(ftr)
    print(len(ftr[0]))