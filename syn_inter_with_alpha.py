# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:40:14 2018

@author: sensetime
"""
import os
#import sys
from custom_layer import bilinear_interp, readPFM, gene_occ, gene_mask
import matplotlib.pyplot as plt
import matplotlib.image as img
from scipy.interpolate import griddata
import numpy as np
import tensorflow as tf

pwd = os.getcwd()
print pwd
#read all files needed
l_disp_file = '/data/driving/disparity_35mm/left/0001.pfm'
r_disp_file = '/data/driving/disparity_35mm/right/0001.pfm'
l_img_file = '/data/driving/finalpass_35mm/left/0001.png'
r_img_file = '/data/driving/finalpass_35mm/right/0001.png'

l_img = img.imread(l_img_file)
r_img = img.imread(r_img_file)
l_disp_t = readPFM(l_disp_file)
l_disp = l_disp_t[0]
r_disp_t = readPFM(r_disp_file)
r_disp = r_disp_t[0]

#generate the coordinate
height, width, channels = l_img.shape
h_indx = np.linspace(0,height-1,height)
w_indx = np.linspace(0, width-1, width)

grid_x, grid_y = np.meshgrid(w_indx, h_indx)
'''
frames = 50
alpha = np.linspace(0,1,frames+1)
i=0
#for item in alpha:
  '''

item = 0.5
lint = grid_x - item * l_disp
lint = np.round(lint)
tl_disp = griddata((lint.flatten(), grid_y.flatten()),item*l_disp.flatten(), \
                        (grid_x, grid_y), method='linear', fill_value=0)
#tl_disp = np.round(tl_disp)

rint = grid_x + (1 - item) * r_disp
rint = np.round(rint)
tr_disp = griddata((rint.flatten(), grid_y.flatten()), (1-item)*r_disp.flatten(), \
                        (grid_x, grid_y), method='linear', fill_value=0)
#tr_disp = np.round(tr_disp)

coor_xinl = grid_x + tl_disp
coor_yinl = grid_y
syn_l_to_t = bilinear_interp(l_img, coor_xinl, coor_yinl,'left_to_target')

coor_xinr = grid_x - tr_disp
coor_yinr = grid_y
syn_r_to_t = bilinear_interp(r_img, coor_xinr, coor_yinr, 'right_to_target')

run_config = tf.ConfigProto()
with tf.Session(config = run_config) as sess:
    syn_l_to_t_val = sess.run(syn_l_to_t)
    syn_r_to_t_val = sess.run(syn_r_to_t)

occ_tl = gene_occ(height, width, np.round(lint))
mask_tl = occ_tl.reshape((height, width, 1))
mask_tl = np.concatenate((mask_tl, mask_tl, mask_tl, mask_tl), axis = 2)
occ_tr = gene_occ(height, width, np.round(rint))
mask_tr = occ_tr.reshape((height, width, 1))
mask_tr = np.concatenate((mask_tr, mask_tr, mask_tr, mask_tr), axis = 2)
'''
occ_tl = gene_mask(item * l_disp, tl_disp, 0.01, 1050, 0.01, 1)
mask_tl = occ_tl.reshape((height, width, 1))
mask_tl = np.concatenate((mask_tl, mask_tl, mask_tl, mask_tl), axis = 2)
occ_tr = gene_mask(tr_disp, (1-item)*r_disp, 0.01, 1050, 0.01, 0)
mask_tr = occ_tr.reshape((height, width, 1))
mask_tr = np.concatenate((mask_tr, mask_tr, mask_tr, mask_tr), axis = 2)
'''
Z = (1 - item) * mask_tl + item * mask_tr
syn_final = ((1 - item) * mask_tl * syn_l_to_t_val*mask_tl + \
            item * mask_tr * syn_r_to_t_val*mask_tr) / Z
plt.figure(1)
plt.imshow(syn_l_to_t_val)
plt.figure(2)
plt.imshow(syn_r_to_t_val)
plt.figure(3)
plt.imshow(syn_final)

plt.figure(4)
plt.subplot(221)
plt.imshow(occ_tl,cmap='gray')
plt.title('mask from target to left')
plt.subplot(222)
plt.imshow(occ_tr,cmap='gray')
plt.title('mask from target to right')
plt.subplot(223)
plt.imshow(mask_tl * syn_l_to_t_val)
plt.title('syn from the left with mask')
plt.subplot(224)
plt.imshow(1-mask_tr * syn_r_to_t_val)
plt.title('syn from the right with mask')
plt.show()