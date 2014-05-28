import numpy as np
from numpy import pi
from scipy.signal import convolve2d
from scipy.misc.pilutil import toimage
from scipy.ndimage import *
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from utils import *

class ShapeContext(object):

    def __init__(self, image, mean_dist=0, r1=0.125, r2=2, nbins_theta=12, nbins_r=5):
        self.image = image
        self.r1 = r1
        self.r2 = r2
        self.nbins_theta = nbins_theta
        self.nbins_r = nbins_r
        self.desc_size = nbins_r * nbins_theta
        self.mean_dist = mean_dist

    def get_samples(self, max_sample = 100.0):
        """ Uniforme sampling """
        #Here I could have improved the sampling if I had more time. I could have probably used a function to find contours and then sample from every other point.  
        img = self.image
        sob = edge(img)
        #optimize
        x, y = np.nonzero(sob > 0)
        points = zip(x,y)
        if len(x) < max_sample:
            max_sample = float(len(x)) 
        sample = np.zeros((1, len(x)))
        sep = np.ceil(len(x)/max_sample)
        sample[::sep, 0::sep] = 1
        self.points = [points[idx] for idx in xrange(len(x)) if sample[0, idx]]
        self.points = np.asarray(self.points)
        self.num_points = self.points.shape[0]
    
    def compute_histogram(self):
        """ quantize angles and radious for all points in the image """
        # compute distance between points 
        distmatrix = np.sqrt(pdist(self.points))
        if not self.mean_dist:
            self.mean_dist = np.mean(distmatrix)
        distmatrix = distmatrix/self.mean_dist
        distmatrix = squareform(distmatrix)
        #compute angles between points
        angles = compute_angles(self.points)
        #quantize angles to a bin
        tbins = np.floor(angles / (2 * pi / self.nbins_theta))
        lg = np.logspace(self.r1, self.r2, num=5)
        #quantize radious to bins
        rbins = np.ones(angles.shape) * -1
        for r in lg:
            counts = (distmatrix < r)  
            rbins = rbins + counts.astype(int)  
        return rbins, tbins
        
    def compute_features(self, samp=100):
        """ construct shape context feature for all sampled points in the image"""
        self.get_samples(samp)
        rbins, tbins = self.compute_histogram()
        inside = (rbins > -1).astype(int)
        features = np.zeros((self.num_points, self.desc_size))
        #construct the feature
        for p in xrange(self.num_points):
            rows = []
            cols = []
            for i in xrange(self.num_points):
                if inside[p,i]:
                    rows.append(tbins[p,i])
                    cols.append(rbins[p,i])
            bins = np.ones((len(rows)))
            a = csr_matrix((bins,(np.array(rows), np.array(cols))), shape=(self.nbins_theta, self.nbins_r)).todense()
            features[p, :] = a.reshape(1, self.desc_size) / np.sum(a)
        self.features = features

    def show_points(self, imsize):
        """ Auxiliary function to visualize sampled points """
        mask = np.zeros(imsize)
        for t in self.points:
            mask[t] = 1
        toimage(mask).show()


