import numpy as np
import cv2
import os
from sklearn import mixture as mix
from matplotlib import pyplot as plt

ProcessedSet = '../DataSet/GF/TRN/'
OriginalSet = '../DataSet/ORI_TRN/'
imagenames = os.listdir(OriginalSet)
n_components = 200
b = 8
num_of_patches_per_image = 500
total_num_of_patches = num_of_patches_per_image * len(imagenames)
patches = np.empty((total_num_of_patches, b * b))
j = 0

for name in imagenames:
    img = cv2.imread(ProcessedSet + name,  0)
    s = img.shape
    patch_idx = -np.ones((num_of_patches_per_image, 2), np.uint8)
    for i in xrange(num_of_patches_per_image):
        while True:
            # don't use same patch twice
            m = np.random.randint(0, s[0] - b + 1)
            n = np.random.randint(0, s[1] - b + 1)
            if [m, n] not in patch_idx:
                patch_idx[i] = [m, n]
                break        
        patches[j] = np.reshape(img[m:m + b, n:n + b], (1, b*b))
        j += 1

GMM = mix.GaussianMixture(n_components=n_components)
GMM.fit(patches)
