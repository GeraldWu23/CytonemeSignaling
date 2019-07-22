# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:38:34 2019

@author: wukak
"""

from sklearn.svm import SVC
from readfile import *
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix as matrix
from cvxopt import solvers
import subprocess as sp



''' TwoClassifier 
    
    choose two labels from [0,1,2] and get the boundary of these two classes    

'''

class TwoClassifier:
    def __init__(self, path, lab0, lab1, kernel = 'linear', gamma='auto', C = 1, degree = 3): # lab0 and lab1 are labels of classes
        self.reader = Readdat(path)
        self.lab0 = lab0
        self.lab1 = lab1
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        
        # collect data by range and labels
        data0, label0 = self.reader.classify(self.lab0, data=self.reader.Xtr, label=self.reader.ttr) # train on training data
        data1, label1 = self.reader.classify(self.lab1, data=self.reader.Xtr, label=self.reader.ttr) # train on training data
        
        # stack two classes
        self.data = np.vstack([data0, data1])
        self.label = np.hstack([label0, label1])
        
        # shuffle the stack
        rand = np.random.permutation(len(self.label))
        self.data = self.data[rand]
        self.label = self.label[rand]
        self.label = [1 if i==lab0 else -1 for i in self.label] # map the original lab0 to 1 and lab1 to -1 for svm
        
        
    def train(self):
        # for linear, gamma doesn't affect the classifier
        svm = SVC(kernel = self.kernel, gamma = self.gamma, C = self.C, degree = self.degree)   
#        print(self.degree)
        svm.fit(self.data, self.label)     
        return svm
                          

''' ThreeClassifier

    get 3 classes classifier
    
    def map_back():
        pred_mapped(): the prediction with label 1 and -1 which are mapped from 0/1/2
        predict(): return the prediction table with the three classifier
'''      

class ThreeClassifier:
    def __init__(self, path, kernel = 'linear', gamma='auto', C = 1, degree = 3):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.degree = degree
        self.cla01 = TwoClassifier(path, 0, 1, self.kernel,gamma=self.gamma, C=self.C, degree=self.degree).train()
        self.cla02 = TwoClassifier(path, 0, 2, self.kernel,gamma=self.gamma, C=self.C, degree=self.degree).train()
        self.cla12 = TwoClassifier(path, 1, 2, self.kernel,gamma=self.gamma, C=self.C, degree=self.degree).train()
        self.reader = Readdat(path)
        self.Xte = self.reader.Xte
        self.tte = self.reader.tte
        
              
    def map_back(self, pred_mapped, lab0, lab1): # map 1,-1 to original label 0,1,2
        return [lab0 if i==1 else lab1 for i in pred_mapped]
    
    
    def predict(self):
        # build classifiers
        pred_mapped01 = self.cla01.predict(self.reader.Xte)
        pred_mapped12 = self.cla12.predict(self.reader.Xte)
        prediction = []
        
        result = np.vstack([self.map_back(pred_mapped01,0,1.5), self.map_back(pred_mapped12,0.5,2)]) # for cla01, bigger than 0 can be 1 or 2, for cla12, less than 2 can be 0 or 1
    
        # overall prediction
        correct = 0
        size = len(result[0]) # the length of test data
        
        for i in range(size):

            cell_pred = round(np.average(result[:,i])) # return the most approximate int from the result
            prediction.append(cell_pred) 
            if cell_pred == self.reader.tte[i]:
                correct += 1
        
        return np.vstack([result, prediction, self.reader.tte]) # return the prediction and the real labels 
    
    
    def view_predict(self):
        prediction = np.column_stack((self.Xte, self.predict()[2,:]))  # get the data and their prediction
        
        prediction_0 = prediction[np.where(prediction[:,2] == 0)]
        prediction_1 = prediction[np.where(prediction[:,2] == 1)]
        prediction_2 = prediction[np.where(prediction[:,2] == 2)]
        
        plt.figure(figsize=(7,7))
        plt.scatter(prediction_0[:,0], prediction_0[:,1], c='C0', marker='.', label = '0')
        plt.scatter(prediction_1[:,0], prediction_1[:,1], c='C1', marker='.', label = '1')
        plt.scatter(prediction_2[:,0], prediction_2[:,1], c='C2', marker='.', label = '2')
        plt.legend()
        plt.title('PREDICTION')
        plt.xlim(-1,2)
        plt.ylim(-1,2)
#        plt.clf()
        
        return 
        
        
''' plot hyperplane of linear svm 
    
    svm = svm
    min_x = minimum x
    max_x = maximum x
    linestyle = linestyle(default='dashed')

    svm has two coefficient w[0] and w[1] with which we can get the k and the b of the margin(a straight line)
'''
def plot_hyperplane(svm, min_x, max_x, linestyle='dashed'): # plot the boundary with svm, only valid when kernel is linear
    w = svm.coef_[0]
    intercept = svm.intercept_[0]
    k = -w[0]/w[1]
    
    xx = np.linspace(min_x, max_x)  # make sure the line is long enough
    yy = k * xx - intercept/w[1]
    plt.plot(xx, yy, linestyle=linestyle) # add 'k' for black line
    


''' svm script '''

def linearKrl(a, b):
    return np.dot(a, b)

def rbfKrl(x, y, gamma=1):
    """RBF kernel with precision gamma."""
    d = x-y
    return np.exp(-gamma*np.dot(d, d))


class svm_sc:
    def __init__(self, Xtr, ttr, Xte, kernel = 'linearKrl', C = 1, graph = False):
        self.Xtr = Xtr
        self.ttr = ttr
        self.Xte = Xte
        self.kernel = kernel
        self.C = C
        self.threshold = 1e-4
        self.slabel = min(self.ttr)  # smaller label before mapped
        self.blabel = max(self.ttr)  # bigger label before mapped
        
        
        # labels have to be 1 or -1 in svm
        if self.blabel != 1 or self.slabel != -1:
            self.ttr = np.array([1 if i == self.blabel else -1 for i in self.ttr])
        
        
        lenX = len(self.Xtr)     # length of training set
        
        # set the parametres in cvxopt
        K = np.zeros((lenX,lenX))  # kernel<xn, xm>
        for i in range(lenX):
            for j in range(i,lenX):
                K[i,j] = self.kernel(self.Xtr[i], self.Xtr[j])
                K[j,i] = self.kernel(self.Xtr[j], self.Xtr[i])
        P = matrix(np.outer(self.ttr, self.ttr) * K)  # tntm<xn, xm>
        q = matrix((-1) * np.ones(lenX))  # -1s
        G = matrix(np.vstack((((-1) * np.eye(lenX)),np.eye(lenX))))  # -1, 1  (C >= an >= 0)
        h = matrix((np.hstack((np.zeros(lenX),(self.C*np.ones(lenX))))))  # 0s, Cs
        A = matrix(self.ttr, (1,lenX), 'd')  # Zigma at = 0
        b = matrix(np.zeros(1))  # 0
        
        
        # solve and get alphas(the weight of punishment of each point)
        try:
            sol = solvers.qp(P,q,G,h,A,b)
        except:
            print(A,lenX,A.typecode)
            return
        alphas = np.array(sol['x'])
        
        
        # get support vectors
        # a point with punishment weight larger than a 
        edge_ind = (alphas>self.threshold).reshape(-1,) # indicator of sv
        ind = np.arange(len(alphas))[edge_ind] # index of sv

        try:
            t_sv = self.ttr[edge_ind]  # label of sv
        except:
            return edge_ind
        
        print('\n\nnumber of support vectors is: ' + str(np.sum(edge_ind)))
        print('number of points is: ' + str(len(self.ttr)) + '\n')
        
        alp_sv = alphas[edge_ind]  # weight of sv(punishment)
        X_sv = self.Xtr[edge_ind]  # sv

        # get the bias
        b_lst = []
        for i in ind:
            bi = self.ttr[i] - alp_sv * self.ttr[i] * K[i,edge_ind] # sigma(an) * tn * K(x) + b = 1
            b_lst.append(bi)
        b = np.mean(b_lst)
        norm = np.linalg.norm(b)
        b /= norm
        
        
        # calculate weight if linear 
        if self.kernel == linearKrl:
            w = (alp_sv * t_sv).dot(X_sv).sum(axis=0)
        else:
            w = None
        
        
        # for test data nonlinear hyperplane is not a line
        pred_val = []
        if self.kernel == linearKrl:
            pred_val = (self.Xte * w).sum(axis=1) + b
        else:   
            for i_te in range(len(self.Xte)):
                pred = 0
                for alp, t, X in zip(alp_sv, t_sv, X_sv):
                    pred += (alp * t)* self.kernel(self.Xte[i_te], X)
                pred += b
                pred_val.append(pred)
                
        
        # get sig
        sig = np.sign(pred_val).reshape(-1,)
            
        # visualise the result
        if graph == True:
            plt.figure(figsize=(7,7))
            plt.plot(self.Xte[sig==-1,0], self.Xte[sig==-1,1], '.', color ='#591D67', label = self.slabel)
            plt.plot(self.Xte[sig== 1,0], self.Xte[sig== 1,1], '.', color ='#FDD225', label = self.blabel)
            plt.legend()
        
        
        return            
            
            
            


if __name__ == '__main__':
    path4 = 'D:/CytonemeSignaling/testDataStudySharpness_linear/SameSharpness/coords_00100.dat'
#    test = Readdat(path4)
#    
#    Xtr = test.Xtr
#    ttr = test.ttr
#    Xte = test.Xte
#    tte = test.tte
#    data1, label1 = test.classify(1, Xtr, ttr)
#    data2, label2 = test.classify(2, Xtr, ttr)
#    Xtr12 = np.vstack([data1, data2])
#    ttr12 = np.hstack([label1,label2])
#    
#    
#
#    rand = np.random.permutation(len(ttr12))
#    Xtr12 = Xtr12[rand]
#    ttr12 = ttr12[rand]
#    ttr12_mapped = np.array([1 if i==1 else -1 for i in ttr12]) # map the original lab0 to 1 and lab1 to -1 for svm
#
#    a = svm_sc(Xtr12, ttr12, Xte, linearKrl, C=0.1, graph=True)
#    
#    test.scatter(Xtr12, ttr12)
#    
#    # clear memory
#    tmp = sp.call('cls',shell=True)





