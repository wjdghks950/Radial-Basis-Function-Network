import numpy as np
import os
import random
import argparse
import matplotlib.pyplot as plt
from dataloader import load_data
from kmeans import kMeans

parser = argparse.ArgumentParser(description='Radial Basis Function - RBF: Gaussian')
parser.add_argument('-k', default=15, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-epochs', default=1000, type=int)
parser.add_argument('-std', default=0.2, type=float)
parser.add_argument('-project', default=False, type=bool)
parser.add_argument('-path', default='RBFN_train_test_files', type=str)
args = parser.parse_args()

def rbf_cis(x, centroid, s):
    diff = 0.0
    for i in range(x.shape[0]):
        diff += (x[i]-centroid[i])**2
    return np.exp(-1 / (2 * s**2) * diff)

class RBFNetCIS(object):
    def __init__(self, centers, std, k=5, lr=0.01, epochs=100, rbf=rbf_cis):
        self.centers = centers
        self.std = std
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.params = {}
        self.params['w1'] = np.random.randn(k,2) # Weights
        self.params['b'] = np.random.randn(1,2) # Biases

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def crossEntropy(self, y, t):
        return -(y*np.log(t+1e-7))

    def fit(self, x, y):
        N = x.shape[0] # Number of data samples
        D = x.shape[1] # Dimension
        a = np.zeros([N, self.k])
        F = np.zeros([N, 2])
        for epoch in range(self.epochs):
            loss = 0
            for i in range(N):
                a[i] = np.array([self.rbf(x[i], c, self.std) for c in self.centers])
                F[i] = a[i].T.dot(self.params['w1']) + self.params['b']
                F_softmax = self.softmax(F[i])
                # Update the weights using cross entropy
                loss0 = self.crossEntropy(y[i,0], F_softmax[0])
                loss1 = self.crossEntropy(y[i,1], F_softmax[1])
                self.params['w1'][:,0] = self.params['w1'][:,0] - self.lr * a[i] * (-loss0)
                self.params['w1'][:,1] = self.params['w1'][:,1] - self.lr * a[i] * (-loss1)
                self.params['b'][:,0] = self.params['b'][:,0] - self.lr * (-loss0)
                self.params['b'][:,1] = self.params['b'][:,1] - self.lr * (-loss1)
                loss += loss0 + loss1
            loss_avg = loss / float(N)
            if epoch % 100 == 0:
                print('[ Epoch: {}| '.format(epoch), end='')
                print('Loss: {:.4f} ]'.format(loss_avg))

    def predict(self, x):
        y_pred = np.zeros([x.shape[0], x.shape[1]])
        for i in range(x.shape[0]):
            a = np.array([self.rbf(x[i], c, self.std) for c in self.centers])
            F = a.T.dot(self.params['w1']) + self.params['b']
            y_pred[i] = F
        y_pred = self.softmax(y_pred)
        y_pred = np.argmax(y_pred, axis=1)
        return np.array(y_pred)

    def accuracy(self, y, t):
        return np.mean(np.equal(y, t))

    def generate_onehot(self, x, y):
        N = x.shape[0]
        D = x.shape[1]
        one_hot_target = np.zeros([N, D])
        for i in range(N):
            one_hot_target[i] = np.array([1,0]) if y[i] == 0 else np.array([0,1])
        return one_hot_target

    def plot(self, x0, x1, center=None):
        fig = plt.figure(1, figsize=(5,5))
        plt.scatter(x0[:,0], x0[:,1], c='black', label='class = 0')
        plt.scatter(x1[:,0], x1[:,1], c='white', label='class = 1')
        if center is not None:
            plt.scatter(center[:,0], center[:,1], c='red', label='centers')
        plt.legend()
        plt.title('RBFN for CIS [ k=' + str(self.k) + '| var=' + str(self.std) + ' ]')
        plt.grid(True)
        plt.show()

def rbf_fa (x, c, s):
    diff = 0.0
    diff = x - c
    return np.exp(-1 / (2 * s**2) * diff**2)   

class RBFNetFA(object):
    def __init__(self, centers, std, k=5, lr=0.01, epochs=100, rbf=rbf_fa):
        self.centers = centers
        self.std = std
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.params = {}
        self.params['w1'] = np.random.randn(k) # Weights
        self.params['b'] = np.random.randn(1) # Biases
        print('[ Std:', self.std ,' ]')

    def fit(self, x, y):
        N = x.shape[0]
        for epoch in range(self.epochs):
            loss_total = 0
            for i in range(x.shape[0]):
                a = np.array([self.rbf(x[i], c, s) for c, s in zip(self.centers, self.std)])
                F = np.squeeze(a.T.dot(self.params['w1']) + self.params['b'])

                loss = (y[i] - F).flatten() ** 2

                error = -(y[i] - F).flatten()

                self.params['w1'] = self.params['w1'] - self.lr * a * error
                self.params['b'] = self.params['b'] - self.lr * error
                #print('Loss/error: ', loss , '///', error)
                loss_total += loss
            loss_avg = float(loss_total) / float(N)
            if epoch % 100 == 0:
                print('[ Epoch: {}| '.format(epoch), end='')
                print('Loss: {0:.4f} ]'.format(loss_avg))

            if loss_avg < 0.01:
                break
    
    def predict(self, x):
        y_pred = []
        N = x.shape[0]
        for i in range(N):
            a = np.array([self.rbf(x[i], c, s) for c, s, in zip(self.centers, self.std)])
            F = a.T.dot(self.params['w1']) + self.params['b']
            y_pred.append(F)
        return np.array(y_pred)

    def sort(self, x, y):
        sort_idx = np.argsort(x, axis=0)
        sorted_x = x.copy()
        sorted_y = y.copy()
        sorted_x = x[sort_idx]
        sorted_y = y[sort_idx]
        return sorted_x, sorted_y

    def plot(self, x, y, y_pred, center=None):
        fig = plt.figure(1, figsize=(5,5))
        plt.plot(x, y, '-o', label='True')
        plt.plot(x, y_pred, '-o', label='Pred')
        if center is not None:
            print('center: ', center)
            plt.scatter(center, [0]*len(center), c='red', label='centers')
        plt.title('RBFN for FA [ k=' + str(self.k) + ' | var=' + str(len(self.std)) + ' ]')
        plt.legend()
        plt.grid(True)
        plt.show()

def selectCenters(data, k, x):
    if data == 'cis':
        # Random selection of centers from data samples
        centers = x[np.random.randint(x.shape[0], size=k), :] # return k random centers
        std = args.std
    elif data == 'fa':
        # Select centers using kMeans algorithm
        centers, std = kMeans(x, k)
    return centers, std

if __name__=='__main__':
    data = input('Enter the dataset to train (cis or fa): ')
    train_set, test_set = load_data(args.path, data)
    train_x, train_y = train_set
    test_x, test_y = test_set

    if data == 'cis': # dataset: cis
        centers, std = selectCenters(data, args.k, train_x)
        rbfn = RBFNetCIS(centers=centers, std=std, k=args.k, lr=args.lr, epochs=args.epochs)
        t = rbfn.generate_onehot(train_x, train_y) # one_hot_vector: t
        train_x0 = train_x[train_y == 0]
        train_x1 = train_x[train_y == 1]

        rbfn.fit(train_x, t) # train the network
        y_pred = rbfn.predict(test_x) # inference
        x0 = test_x[y_pred == 0]
        x1 = test_x[y_pred == 1]
        
        accuracy = rbfn.accuracy(y_pred, test_y)
        print('[ Accuracy : ', accuracy, ' ]')
        rbfn.plot(x0, x1, centers)

    elif data == 'fa': # dataset: fa (function approximation)
        centers, std = selectCenters(data, args.k, train_x)
        rbfn = RBFNetFA(centers=centers, std=std, k=args.k, lr=args.lr, epochs=args.epochs)
        # print('train_x: ', np.squeeze(train_x))
        # print('train_y: ', train_y)
        train_x = np.squeeze(train_x)
        train_x, train_y = rbfn.sort(train_x, train_y)
        rbfn.fit(train_x, train_y)
        #y_pred = rbfn.predict(train_x)
        #rbfn.plot(train_x, train_y, y_pred)
        rbfn.sort(test_x, test_y)
        y_pred = rbfn.predict(test_x)
        rbfn.plot(test_x, test_y, y_pred, centers)