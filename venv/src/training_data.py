import get_train_data
import face_feature

import matplotlib.pyplot as plt
import numpy as np
import random as rd

from sklearn import ensemble, multiclass, tree, decomposition, neural_network
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



def load_data(path):
    try:
        res = np.load(path)
        print("successfully loaded the features!")
        return res
    except:
        print("Failed loading the features!")
        return 0

def select_data(raw_data):
    raw_data = raw_data.tolist()
    n = len(raw_data)
    # print(n)

    # get a random sequence:
    seq = np.arange(0,n,1)
    rd.shuffle(seq)

    # put data to each set
    train_data = []
    test_data = []
    for i in range(n):
        if seq[i] < n * 5/6:
            train_data.append(raw_data[i])
        else:
            test_data.append(raw_data[i])

    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    return train_data, test_data

def get_xy(data):
    y = data[:,:1]
    X = data[:,1:]
    return X, y

def lets_train(x,y,classifier,strategy):
    # decomposition PCA
    pca = decomposition.PCA(n_components=50).fit(x)
    # newx = pca.fit_transform(x)
    newx = x

    if strategy == 'ova':
        if classifier == 'svm':
            clf = multiclass.OneVsRestClassifier(SVC(C = 1e2, gamma = 'auto')).fit(newx,y)
        elif classifier == 'decisionTree':
            clf = multiclass.OneVsRestClassifier(ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=1),
                                                                            n_estimators=200)).fit(newx,y)
        elif classifier == 'logistic':
            clf = multiclass.OneVsRestClassifier(LogisticRegression(C=1e8, penalty='l2',solver='newton-cg')).fit(newx,y)
        elif classifier == 'mlp':
            clf = multiclass.OneVsRestClassifier(neural_network.MLPClassifier(solver='sgd',activation = 'relu',
                                                                              max_iter = 100,alpha = 1e-5,hidden_layer_sizes = (300,400,200,200),
                                                                              verbose = True)).fit(newx,y)
        else:
            print("wrong classifier type in ova")
            return 0,1,0

    elif strategy == 'ovo':
        if classifier == 'svm':
            clf = multiclass.OneVsOneClassifier(SVC(C = 1e2, gamma = 'auto')).fit(newx,y)
        elif classifier == 'decisionTree':
            clf = multiclass.OneVsOneClassifier(ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=1),n_estimators=200)).fit(newx,y)
        elif classifier == 'logistic':
            clf = multiclass.OneVsOneClassifier(LogisticRegression(C=1e8,penalty='l2',solver='newton-cg')).fit(newx,y)
        elif classifier == 'mlp':
            clf = multiclass.OneVsRestClassifier(neural_network.MLPClassifier(solver='sgd',activation = 'logistic',
                                                                              max_iter = 200,alpha = 1e-5,hidden_layer_sizes = (300,300),
                                                                              verbose = True)).fit(newx,y)
        else:
            print("wrong classifier type in ovo")
            return 0, 1, 0

    else:
        print("wrong classifier strategy")
        return 0, 1, 0


    err = np.sum(np.sign(abs(y.T-clf.predict(newx))))/y.shape[0]
    return clf, err, pca

def lets_test(x,y,clf,pca):
    # newx = pca.fit_transform(x)
    newx = x
    test_res = clf.predict(newx)
    err = np.sum(np.sign(abs(y.T-test_res)))/test_res.shape[0]
    return test_res,err

# the entrance function
def training_data(feature_path, strategy = 'ova', classifier = 'svm'):

    # load features stored as numpy
    raw_data = load_data(feature_path)
    if raw_data is 0:
        exit(1)
    print(raw_data.shape)

    # randomly select 2/3 as train set, 1/6 as verification set, 1/6 as test set
    train_data, test_data = select_data(raw_data)
    print(train_data.shape, test_data.shape)

    trainX,trainy = get_xy(train_data)
    testX, testy = get_xy(test_data)

    clf,train_err, pca = lets_train(trainX, trainy,classifier,strategy)
    test_res, test_err = lets_test(testX,testy,clf, pca)
    print("train error:", train_err, " test error:", test_err)

    return clf, pca


if __name__ == '__main__':
    feature_path = '../img/facesdb/feature.npy'
    training_data(feature_path)

