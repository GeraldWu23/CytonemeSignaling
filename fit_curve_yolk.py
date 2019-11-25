# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:38:59 2019

@author: wukak
"""

import numpy as np
from numpy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt





class fit_model_yolk:
    ''' model fitting with given points 
    
        self.Boltzmann(): give Boltzmann output with specified x, sigma, b
        self.fit(): 
    '''
    def __init__(self, xlist, ylist):
        self.xlist = xlist
        self.ylist = ylist
        
    def Boltzmann(self, x, sigma, b):
        '''
        sigma stands for threshold
        b stands for sharpness
        '''
        return 1 / (1 + exp( -(x-sigma) / b))
    
    
    def fit(self,func):
        ''' Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized '''
        xlist, ylist = self.xlist, self.ylist
        try:
            # The estimated covariance of popt
            popt, pcov = curve_fit(func, np.array(xlist), np.array(ylist), p0=(132, 68))
        except:
            print("Can't fit")
            return False
        return popt
    
    
    def plot_model(self, func, start, end, num = 50, title = None, graph = True, inputc='C0', predictc='C1', inputname='input', predictname='predict'):     
        ''' train the model and plot the model with input points '''
        xlist, ylist = self.xlist, self.ylist
        popt = self.fit(func)
        
        try: # False will be not subscriptable
            popt[0]
        except:
            return False

        
        if graph:
            pred_y = func(np.array(np.linspace(start,end,num)), popt[0], popt[1])
            plt.plot(xlist, ylist, marker = 'o',c=inputc, label = inputname + 
                     '  threshold : '+str(round(popt[0],3)))
            plt.plot(np.linspace(start, end, num), pred_y,c = predictc, marker = '.',
                     label = predictname+'  sharpness: '+str(round(popt[1],3)))
            plt.legend(fontsize = 12)
            if title:
                plt.title(title)
        
        return popt[0],  popt[1]
        
        
        
if __name__ == '__main__':
    
    pass