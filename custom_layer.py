from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
 

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
      
    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def bilinear_interp(im, x, y, name):
  """Perform bilinear sampling on im given x, y coordinates
  
  This function implements the differentiable sampling mechanism with
  bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
  (5).
 
  Args:
    im: Tensor of size [height, width, depth] 
    x: Tensor of size [height, width, 1]
    y: Tensor of size [height, width, 1]
    name: String for the name for this opt.
  Returns:
    Tensor of size [height, width, depth]
  """
  with tf.variable_scope(name):
    x = tf.reshape(x, [-1]) 
    y = tf.reshape(y, [-1]) 

    # constants
    #num_batch = tf.shape(im)[0]
    #height, width, channels = im.get_shape().as_list()
    height, width, channels = im.shape
    x = tf.to_float(x)
    y = tf.to_float(y)

    #height_f = tf.cast(height, 'float32')
    #width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)

    max_x = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_y = tf.cast(tf.shape(im)[0] - 1, 'int32')
    #x = (x + 1.0) * (width_f - 1.0) / 2.0
    #y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width 
    dim1 = width * height

    # Create base index
    base = tf.range(1) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2 
    base_y1 = base + y1 * dim2 
    idx_a = base_y0 + x0 
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Use indices to look up pixels
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    # Interpolate the values 
    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
    wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

    output = tf.add_n([wa*pixel_a, wb*pixel_b, wc*pixel_c, wd*pixel_d]) 
    output = tf.reshape(output, shape=tf.stack([height, width, channels]))
    return output
    
def gene_occ(height, width, coor_x):
    coor_x = np.clip(coor_x,0,width-1);
    occ = np.zeros((height,width));
    for i in range(height):
        max_value = coor_x[i,0]
        for j in range(width):
            if coor_x[i,j] > max_value:
                occ[i,j] = 1 
                max_value = coor_x[i,j]
            else:
                occ[i,j] = 0
    return occ

def gene_mask(left_disp, right_disp, threshold, f, baseline, flag):
    ltx_disp = left_disp
    rtx_disp = right_disp
    ltx_depth = f * baseline / ltx_disp
    rtx_depth = f * baseline / rtx_disp
    if flag == 1:
        diff_depth = rtx_depth - ltx_depth
    else:
        diff_depth = ltx_depth - rtx_depth
    diff_depth[diff_depth > threshold] = 1
    diff_depth[diff_depth < threshold] = 0
    return 1 - diff_depth
             
def gene_visi_l(ori_x, trans_x, height, width, l_depth, r_depth):   
    ori_x = ori_x.flatten()
    trans_x = trans_x.flatten()
    visi_ltr = np.zeros(len(ori_x)) 
    for indx in range(len(ori_x)):
        if trans_x[indx] < 0:
            visi_ltr[indx] = 1
        elif trans_x[indx] > width:
            visi_ltr[indx] = 0
        else:
            rx = np.floor(trans_x[indx])
            y  = indx // width
            lx = ori_x[indx]
            l_value = l_depth[y, int(lx)]
            r_value = r_depth[y, int(rx)]
            if l_value - r_value < 0:
                visi_ltr[indx] = 1
            else:
                visi_ltr[indx] = 0
    return visi_ltr

def gene_visi_r(ori_x, trans_x, height, width, l_depth, r_depth):
    ori_x = ori_x.flatten()
    trans_x = trans_x.flatten()
    visi_rtl = np.zeros(len(ori_x))
    for indx in range(len(ori_x)):
        if trans_x[indx] < 0:
            visi_rtl[indx] = 0
        elif trans_x[indx] > width:
            visi_rtl[indx] = 1
        else:
            rx = np.floor(trans_x[indx])
            y  = indx // width
            lx = ori_x[indx]
            l_value = l_depth[y, int(lx)]
            r_value = r_depth[y, int(rx)]
            if r_value - l_value < 0:
                visi_rtl[indx] = 1
            else:
                visi_rtl[indx] = 0
    return visi_rtl
    
def line_sample(x_old, x_new, img):
    des_value = np.zeros(img.shape)
    height, width = img.shape
    for i in range(height):
        line_x = x_old[i,:]
        for j in range(width):
            point_x = x_new[i,j] 
            if point_x >= line_x.max():
                des_value[i,j] = img[i, width-1]
            elif point_x <= line_x.min():
                des_value[i,j] = img[i, 0]
            else:
                l_point = line_x[line_x < point_x]
                l_indx = np.repeat(point_x, len(l_point))
                l_dist = np.abs(l_indx - l_point)
                l_nearest = np.where(l_dist == l_dist.min())
                l_value = img[i, l_nearest]
                r_point = line_x[line_x > point_x]
                r_indx = np.repeat(point_x, len(r_point))
                r_dist = np.abs(r_indx - r_point)
                r_nearest = np.where(r_dist == r_dist.min())
                r_value = img[i, r_nearest]
                #gap = line_x[r_nearest] - line_x[l_nearest]
                gap = r_dist[r_nearest] + l_dist[l_nearest]
                des_value[i,j] = (r_dist[r_nearest] * l_value \
                                    + l_dist[l_nearest] * r_value) / gap
    return des_value
    