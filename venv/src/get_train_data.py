import face_feature
import os
import numpy as np

def save_feature(path, res):
    try:
        np.save(path+'feature', res)
        return 1
    except:
        print("Error saving res")
        return 0

def save_race_feature(path, res):
    try:
        np.save(path+'race', res)
        return 1
    except:
        print("Error saving res")
        return 0



def get_train_data():
    file_path = '../img/facesdb/' # path of the file that stores the raw images

    file_num = 38 # total number of files
    type_num = 7

    res = []

    detector = face_feature.get_detector()
    predictor = face_feature.get_predictor()

    for i in range(1,file_num + 1):
        if i != 20 and i != 22:
            if i < 10:
                seq = '00' + str(i)
            else:
                seq = '0' + str(i)
            file_path_ex = file_path + 's' + seq + '/bmp/'

            for j in range(type_num):
                full_path = file_path_ex + 's' + seq +'-0' + str(j) +'_img.bmp'
                print(full_path)
                fts = face_feature.get_face_feature(full_path, detector, predictor)

                for ft in fts:
                    rj = [j]
                    for k in ft:
                        rj.append(k)
                    res.append(rj)
    res = np.asarray(res)

    save_path = '../img/facesdb/'
    if save_feature(save_path,res):
        print("feature saved!")
    return res

def get_race_data():
    file_path = '../img/CFDImages/'  # path of the file that stores the raw images
    res = []
    clss = ['A','B','L','W']

    detector = face_feature.get_detector()
    predictor = face_feature.get_predictor()

    for (root, dirs, files) in os.walk(file_path, ):
        for dir in dirs:
            img_root = os.path.join(root, dir)
            for img_name in os.listdir(img_root):
                if img_name.endswith('N.jpg'):
                    full_path = os.path.join(img_root,img_name)
                    print(full_path)
                    _,fts = face_feature.get_all_feature(full_path,detector,predictor)

                    for ft in fts:
                        cls = img_name[4]
                        rj = [clss.index(cls)]
                        for k in ft:
                            rj.append(k)
                        res.append(rj)
                        print(len(res))
    res = np.asarray(res)

    save_path = '../img/CFDImages/'
    if save_feature(save_path, res):
        print("feature saved!")
    return res





if __name__ == '__main__':
    res = get_race_data()
    print(len(res))
    print(len(res[0]))