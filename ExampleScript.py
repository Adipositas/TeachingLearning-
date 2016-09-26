# -*- coding: utf-8 -*-
"""
For presentation in teaching and learning, showing image filtering with kernels,
and different effects imposable on an image.

Created on Wed Aug 31 10:20:35 2016

@author: David Norsk
"""
#%%

import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
from skimage import feature
from scipy import ndimage

image = mh.imread("C:\\Users\\dnor\\Desktop\\Stuff\\portrait2.png")[:,:,0]
image = mh.resize.resize_to(image,[512, 508] )
#plt.imshow(image, cmap='Greys_r')


id_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.int32)
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gauss_kernel = (1/16) * np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]])

def image_filtered(image, kernel):
      
    rows = np.size(image,1)
    cols = np.size(image,0)
    filtered_image = np.zeros([np.size(image,0), np.size(image,1)])
    temp_val = 0
    
    for k in range(0+1, rows-1):
        for i in range(0+1, cols-1):
            temp_val = (image[i-1,k-1] * kernel[0,0] +
            image[i-1, k] * kernel[0,1] + image[i-1, k + 1] * kernel[0,2] +
            image[i, k - 1] * kernel[1,0] + image[i,k] * kernel[1,1] +
            image[i, k + 1] * kernel[1,2] + image[i + 1, k - 1] * kernel[2,0] +
            image[i + 1, k] * kernel[2, 1] + image[i + 1,k + 1] * kernel[2,2])
            
            if temp_val < 0:
                temp_val = 0
            filtered_image[i, k] = temp_val
                
    return filtered_image
            #filtered_image[k,i] = np.sum(image[k-1:k+2, i-1:i+2] * kernel)
            #np.sum(np.multiply(image[k-1:k+2, i-1:i+2], kernel))
     #       filtered_image = [image[k-1:k+2, i-1:i+2]]
                                             

rows = np.size(image,1)
cols = np.size(image,0)

filtered_image = image_filtered(image, edge_kernel)
plt.imshow(filtered_image, cmap='Greys_r')

#%% Make edge map
impose_image = np.copy(image)

for k in range(np.size(filtered_image,0)):
    for i in range(np.size(filtered_image, 1)):
        if filtered_image[k, i] > 30:
            impose_image[k, i] = 255

edges = feature.canny(image, sigma = 8)

impose_image = np.copy(image)

for i in range(np.size(image,0)):
    for k in range(np.size(image, 1)):
        if edges[i,k] == True:
            impose_image[i, k] = 255
plt.figure()
plt.imshow(impose_image, cmap='Greys_r')


#%% Range Filtering


rows = np.size(image,1)
cols = np.size(image,0)
filtered_image = np.zeros([np.size(image,0), np.size(image,1)])
temp_val = 0

kernel = gauss_kernel

ranges = [[222, 228], [224, 300]]
x_range = 15
y_range = 4
for ran in ranges:
    for k in range(ran[1]-x_range, ran[1]+x_range):
        for i in range(ran[0] - y_range, ran[0] + y_range):
            temp_val = (image[i-1,k-1] * kernel[0,0] +
            image[i-1, k] * kernel[0,1] + image[i-1, k + 1] * kernel[0,2] +
            image[i, k - 1] * kernel[1,0] + image[i,k] * kernel[1,1] +
            image[i, k + 1] * kernel[1,2] + image[i + 1, k - 1] * kernel[2,0] +
            image[i + 1, k] * kernel[2, 1] + image[i + 1,k + 1] * kernel[2,2])
            
            if temp_val < 0:
                temp_val = 0
            filtered_image[i, k] = temp_val
 
filtered_image = filtered_image/(filtered_image.max()/255.0)
           
for i in range(np.size(image,0)):
    for k in range(np.size(image,1)):
        if filtered_image[i,k] == 0:
            filtered_image[i,k] = image[i,k]
            

filtered_image = ndimage.gaussian_filter(image, sigma=5)          
plt.figure()

plt.imshow(filtered_image, cmap='Greys_r')

#%%

filtered_image = ndimage.sobel(image)          
plt.figure()

plt.imshow(filtered_image, cmap='Greys_r')