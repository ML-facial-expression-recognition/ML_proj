import face_feature
import sys
import numpy as np

def save_feature(path, res):
    try:
        np.save(path+'feature', res)
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

if __name__ == '__main__':
    res = get_train_data()
    print(len(res))
    print(len(res[0]))