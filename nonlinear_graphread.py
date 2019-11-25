# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:28:31 2019

@author: wukak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 20:09:55 2019

@author: wukak
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
MAXINT = 10000
BGintensity = 190
from fit_curve_yolk import fit_model_yolk




def showimg(imgBGR):
    plt.figure(figsize=(12,8))
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRGB, origin='lower')
    return imgRGB
    
def showimggray(imgBGR):
    plt.figure(figsize=(12,8))
    imgGRAY = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    plt.imshow(imgGRAY, cmap=plt.cm.gray)
    return imgGRAY


    



'''
    get the img of the yolk
    
    yolk_thr needs to be decided according to the original img
    
    NOTE: the x y in plot() is different from x y in imshow()
'''
def getyolk(img):
    imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yolk_thr = 170
    maxp = 170
    minp = min(map(min, imgGRAY))
    xtrain = []
    ytrain = []
    minx = MAXINT
    miny = MAXINT
    maxx = 0
    maxy = 0
    
    # label the pixels and get target points
    for row in range(len(imgGRAY)):
        for col in range(len(imgGRAY[row])):
            if imgGRAY[row][col] > yolk_thr:
#                imgGRAY[row][col] = 255
                pass
            elif imgGRAY[row][col] > (maxp+minp)/2:
#                imgGRAY[row][col] = 180  # the colour of the light part
                minx = min(row, minx)
                miny = min(col, miny)
                maxx = max(row, maxx)
                maxy = max(col, maxy)
            else:
#                imgGRAY[row][col] = 30  # the colour of the dark part
                xtrain.append(row)
                ytrain.append(col)
                minx = min(row, minx)
                miny = min(col, miny)
                maxx = max(row, maxx)
                maxy = max(col, maxy)
                
                
    #yolk_centre = ((maxx+minx)/2, (maxy+miny)/2)
    return xtrain, ytrain, imgGRAY, (minx, miny, maxx, maxy)


'''
    linear regression
'''
def train(xtrain, ytrain):
    linreg = LinearRegression()
    linreg.fit(np.array(xtrain).reshape(-1,1), ytrain)
    return linreg.intercept_, linreg.coef_
                
    
'''
    rotate images
'''
def rotate(img, centre, angle, show = False):
    rotm = cv2.getRotationMatrix2D(centre, angle, 1)
    rotimg = cv2.warpAffine(img, rotm, (len(img[0]),len(img)))
    
    # clean the corners
    for row in range(len(img)):
        for col in range(len(img[0])):
            if rotimg[row][col] == 0:
                rotimg[row][col] = BGintensity
    
    if show:
        showimg(rotimg)
        
    return rotimg



'''
    get intensities of a range
'''
def intensity(imgGRAY, meetpointx, meetpointy, ori_coef, length=100):
    line_x, line_y = orthogonalline(meetpointx, meetpointy, ori_coef, length, graph = False)
    inten_li = []

    for i in range(len(line_x)):
        inten_li.append(imgGRAY[int(line_y[i])][int(line_x[i])])
    
    return inten_li

'''
    get an orthogonal line
'''
def orthogonalline(meetpointx, meetpointy, ori_coef, length=100, graph = True):
        ortho_coef = -1/ori_coef
        ortho_intercept = meetpointx - meetpointy * ortho_coef
        
        line_y = np.linspace(int(meetpointy - length/2), int(meetpointy + length/2), num = length)
        line_x = [i * orth_coef + ortho_intercept for i in line_y]
        
        if graph:
            plt.plot(line_x, line_y, 'r')
        
        return np.array(line_x).reshape((-1,)), line_y



'''
    get average intensities
'''

def average_intensity(imgGRAY, ytop, ybottom, ori_coef, ori_intercept, num=6, length=100):
    xtop = ytop * ori_coef + ori_intercept
    xbottom = ybottom * ori_coef + ori_intercept
    
    # lines
    xlist = np.linspace(xbottom, xtop, num = num + 2)
    ylist = np.linspace(ybottom, ytop, num = num + 2)
    
    # intensity
    inten_stat = np.array([0.0]*length)
   
    for i in range(len(xlist)):
        # ignore the first and the last line
        if i > 0 and i < (len(xlist) - 1):
            inten_stat += np.array(intensity(imgGRAY, xlist[i], ylist[i], ori_coef, length = length)) / num
    return inten_stat[::-1] # order from left to right in img
    
    

'''
    fit the intensity list with two boltzmann models
'''
def bound_sharpness(intensity_list):
    
    minvalue = max(intensity_list)
    minpoint = 0
    
    # get lowest point
    for i in range(len(intensity_list)):
        if intensity_list[i] < minvalue:
            minvalue = intensity_list[i]
            minpoint = i
    intensity_list = [(i - min(intensity_list)) / (max(intensity_list) - min(intensity_list)) for i in intensity_list]
    
    firsthalf = fit_model_yolk([i for i in range(minpoint)][::5], intensity_list[:minpoint][::-5])
    sigma0, b0 = firsthalf.fit(firsthalf.Boltzmann)
    
    
    secondhalf = fit_model_yolk([i for i in range(minpoint,len(intensity_list))][::5], intensity_list[minpoint:][::5])
    sigma1, b1 = secondhalf.fit(secondhalf.Boltzmann)
#    plt.plot(intensity_list[minpoint:][::5])

    return b0, b1, sigma0, sigma1, minpoint


#def analyse(image_path):
#    img = cv2.imread(image_path) # BGR image
##    showimg(img)
#    
#    # yolk info
#    xtrain, ytrain, imgGRAY, scale = getyolk(img)
#    b, a = train(xtrain, ytrain) # intercept, coef
#    angle = -np.arctan(a)/np.pi*180
#    minx, miny, maxx, maxy = scale
#    centre = ((minx+maxx)/2, (miny+maxy)/2)
#    imgGRAY_rot = rotate(imgGRAY, centre, angle, show=False)
#    #showimg(imgGRAY_rot)
#    
#    plt.figure()
#    
#    # intensity list
#    inten_li = intensity(imgGRAY_rot, minx, miny, maxx, maxy)
#    inten_li_norm = [(i - min(inten_li)) / (max(inten_li) - min(inten_li)) for i in inten_li] 
#    
#    # fit boltzmann
#    b0, b1, sigma0, sigma1, minpoint = bound_sharpness(inten_li_norm)
#    threshold0 = (round(minpoint - sigma0 +miny,1), round(max(inten_li[:minpoint])/2 + min(inten_li[:minpoint])/2,1))
#    threshold1 = (round(sigma1 + miny,1), round(max(inten_li[minpoint:])/2 + min(inten_li[minpoint:])/2),1)
#    minpoint += miny # adjust to practical scale
#    print('threshold0: ' + str(threshold0))
#    print('threshold1: ' + str(threshold1))
#    
#    
#    # visualisation
#    plt.plot(inten_li_norm)
#    
#    
#    plt.xticks(np.arange(0, len(inten_li_norm), 200), np.arange(miny, maxy, 200))
#    plt.yticks(np.arange(0,1.2,0.2), [round(i*(max(inten_li) - min(inten_li))+min(inten_li),2) for i in np.arange(0,1.2,0.2)])
# 
#    plt.text(0,0,'sharpness0 is '+str(round(b0,3))+'\nand sharpness1 is '+str(round(b1,3)))
#    plt.title('normalise gray level')
#    plt.plot(threshold0[0]-miny, inten_li_norm[int(threshold0[0]-miny)], 'ro')
#    plt.plot(threshold1[0]-miny, inten_li_norm[int(threshold1[0]-miny)], 'go')
#    plt.annotate(s=str(threshold0), xy = (threshold0[0]-miny, inten_li_norm[int(threshold0[0]-miny)]), xytext=(threshold0[0]-miny-200, 0.45))
#    plt.annotate(s=str(threshold1), xy = (threshold1[0]-miny, inten_li_norm[int(threshold1[0]-miny)]), xytext=(threshold1[0]-miny-50, 0.45))
#    
#    
#    return True
    


'''
    __main__
'''

if __name__ == '__main__':

    img = cv2.imread('D://CytonemeSignaling//1b.tif')
    xtrain, ytrain, imgGRAY, (minx, miny, maxx, maxy) = getyolk(img)
    intercept,coef = train(xtrain, ytrain)
    
    showimg(img)
       
    
    for i in range(len(xlist)):
        # ignore the first and the last line
        if i > 0 and i < (len(xlist) - 1):
            orthogonalline(xlist[i], ylist[i], coef)

    plt.figure(figsize=(15,9))  
    plt.plot(average_intensity(imgGRAY, ytop, ybottom, coef, intercept, num=6, length=400))           
    