# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:09:19 2019

@author: wukak
"""

import numpy as np
import matplotlib.pyplot as plt



class Readdat:
    ''' Read data from the raw file 
    
        self.data = x and y of the data, in list
        self.label = labels of the data, in list
        self.Xtr = x and y of trainning data
        self.ttr = labels of training data
        self.Xte = x and y of testing data
        self.tte = labels of testing data
        self.scatte() : scatter the data in different colour according to their labels
        self.classify() :　return data with spacific label
    '''
    def __init__(self, path):       
        with open(path) as file:
            linelist = file.readlines()
            lst = [line.split() for line in linelist]
            self.data = []
            self.label = []
            for line in lst:
                self.data.append([float(line[0]), float(line[1])])
                self.label.append(int(float(line[2])))
            self.data  = np.array(self.data)
            self.label = np.array(self.label)
            
            # cut dataset to training data and testing data
            size = int(len(self.data)/2)
            self.Xtr = self.data[:size]
            self.ttr = self.label[:size]
            self.Xte = self.data[size:]
            self.tte = self.label[size:]


    
    def scatter(self, data=None, label=None): 
        ''' plot the points with different colours '''
        plt.figure(figsize = (7,7))
        if data is not None and label is not None: # if a spacific class is specialised
            plt.scatter(data[:,0], data[:,1], c = label, marker = '.')
        else:
            # classify with the label from the dataset
            data0,label0 = self.classify(0)
            data1,label1 = self.classify(1)
            data2,label2 = self.classify(2)
            
            # scatter them with labels
            plt.scatter(data0[:,0], data0[:,1], c=['C0']*len(data0), marker = '.', label = 0)
            plt.scatter(data1[:,0], data1[:,1], c=['C1']*len(data1), marker = '.', label = 1)
            plt.scatter(data2[:,0], data2[:,1], c=['C2']*len(data2), marker = '.', label = 2)
            plt.legend()
            plt.show()
        
    
    
    def classify(self, lab, data = None, label = None): 
        ''' return classified data and label using the actual labels '''  
        if (data is None) and (label is None): # if data or label is not specialised, the whole data set is used
            data = self.data
            label = self.label
            
        data_classified = []
        label_classified = []
        for i in range(len(data)):
            if (label[i] == lab):
                data_classified.append(data[i])
                label_classified.append(label[i])
        return np.array(data_classified), np.array(label_classified) # return classified data and classified label
        

        
        
        
if __name__ == '__main__':
    path = 'D:/CytonemeSignaling/testFateCoords.dat'
    
    
    
#    test = Readdat(path)
#    print(test.classify(0))
#    print('\n-------------------\n')
#    print(pred)   
    
    
    
    
    
    
    
    
    
    