import numpy as np
import math
from scipy.signal import convolve2d
from numpy import pi
from PIL import ImageFilter

def compute_angles(points):
    """ compute angles between a set of points """
    angles = np.zeros((len(points), len(points)))
    for i in xrange(len(points)):
        p1 = points[i,:]
        for j in xrange(i+1, len(points)) :
            p2 = points[j,:]
            angles[i,j] = math.atan2((p1[1] - p2[1]),(p2[0] - p1[0]))
            angles[j,i] = -angles[i,j]
    #angles between 0, 2*pi
    angles = np.fmod(np.fmod(angles, 2 * pi) + 2 * pi, 2 * pi)
    return angles

def edge(img):
    """ Simple edge detection and filtering """
    #Here I could have improve the classification accuracy by removing noise of the image and the blurred checkerboard.  
    img = img.convert('L')
    img = img.filter(ImageFilter.FIND_EDGES)
    img = np.asarray(img)
    if len(img.shape) > 2:
        img = img[:,:,1]
        img = (img < 100) & img
    img = img *  255
    return img

