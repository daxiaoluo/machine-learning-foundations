import numpy as np
import random

class PLA(object):
    def __init__(self, dim):
        self._dim = dim

    def getDataSet(self, path):
        training_set = open(path)
        training_set = training_set.readlines()
        num = len(training_set)

        x_train = np.zeros((num, self._dim))
        y_train = np.zeros((num, 1))

        for i in range(num):
            trimData = training_set[i].strip()
            data = trimData.split('\t')
            y_train[i, 0] = np.int(data[1])
            items = data[0].strip().split(' ')
            x_train[i, 0] = 1.0
            x_train[i, 1] = np.float(items[0])
            x_train[i, 2] = np.float(items[1])
            x_train[i, 3] = np.float(items[2])
            x_train[i, 4] = np.float(items[3])

        return x_train, y_train

    def trainPLANaive(self, x, y, w, eta, rand=False):
        iterations = 0
        num = len(x)
        index = []
        for i in range(num):
            index.append(i)
        if rand:random.shuffle(index)
        flag = True
        while True:
            flag = True
            for i in index:
                if np.dot(x[i, :], w)[0] * y[i, 0] <= 0:
                    w = w + eta * y[i, 0] * x[i, :].reshape(5, 1)
                    iterations += 1
                    flag = False
            if flag:
                break
        return iterations
    
    def trainPocketPLA(self, x, y, w, updates, eta, rand=False, raw=False):
        num = len(x)
        index = []
        for i in range(num):
            index.append(i)
        if rand: random.shuffle(index)
        count = 0
        bestErr = 1.0
        bestW = np.zeros((self._dim, 1))
        while True:
            for i in index:
                if np.dot(x[i, :], w)[0] * y[i, 0] <= 0:
                    w = w + eta * y[i, 0] * x[i, :].reshape(5, 1)
                    count += 1
                    err = self.validate(x, y, w)
                    if err < bestErr: 
                        bestErr = err
                        bestW = np.copy(w)
                    if count >= updates:break
            if count >= updates:break
        return bestW, w

    def validate(self, x, y, w):
        count = 0.0
        for i in range(len(x)):
            if np.dot(x[i, :], w)[0] * y[i, 0] <= 0:
                count += 1
        return count / len(x)

if __name__ == '__main__':
    train_path = '/Users/tao.luo/MyProject/ml/hw1_15_train.dat'
    pla = PLA(5)
    x, y = pla.getDataSet(train_path)
    w = np.zeros((5, 1))
    print pla.trainPLANaive(x, y, w, 1)
    s = 0
    for i in range(2000):
        w = np.zeros((5, 1))
        s += pla.trainPLANaive(x, y, w, 1, True) 
    print s / 2000.0

    s = 0
    for i in range(2000):
         w = np.zeros((5, 1))
         s += pla.trainPLANaive(x, y, w, 0.5, True)
    print s / 2000.0

    pocket_train_path = '/Users/tao.luo/MyProject/ml/hw1_18_train.dat'
    pocket_test_path = '/Users/tao.luo/MyProject/ml/hw1_18_test.dat'
    x, y = pla.getDataSet(pocket_train_path)
    sum_best_err_rate = 0.0
    sum_raw_err_rate = 0.0
    for i in range(2000):
        w = np.zeros((5, 1))
        best_w, raw_w = pla.trainPocketPLA(x, y, w, 50, 1, True)
        t_x, t_y = pla.getDataSet(pocket_test_path)
        sum_best_err_rate += pla.validate(t_x, t_y, best_w)
        sum_raw_err_rate += pla.validate(t_x, t_y, raw_w)

    print sum_best_err_rate / 2000
    print sum_raw_err_rate / 2000

    sum_best_err_rate = 0.0
    sum_raw_err_rate = 0.0
    for i in range(2000):
         w = np.zeros((5, 1))
         best_w, raw_w = pla.trainPocketPLA(x, y, w, 100, 1, True)
         t_x, t_y = pla.getDataSet(pocket_test_path)
         sum_best_err_rate += pla.validate(t_x, t_y, best_w)
         sum_raw_err_rate += pla.validate(t_x, t_y, raw_w)

    print sum_best_err_rate / 2000
    print sum_raw_err_rate / 2000





