
# coding: utf-8

# In[1]:

import numpy as np


### Load training data

# In[14]:

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68
from menpofast.utils import convert_from_menpo

training_images = []
for i in mio.import_images('/data/PhD/DataBases/faces/lfpw/trainset/', verbose=True, max_images=None):
    
    # convert the image from menpo Image to cvpr2015 Image (channels at front)
    i = convert_from_menpo(i)
    
    labeller(i, 'PTS', ibug_face_68)
    i.crop_to_landmarks_proportion_inplace(0.5, group='PTS')
    if i.n_channels == 3:
        i = i.as_greyscale(mode='average')
    training_images.append(i)


### Load test data

# In[16]:

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68
from menpofast.utils import convert_from_menpo

test_images = []
for i in mio.import_images('/data/PhD/DataBases/faces/lfpw/testset/', verbose=True, max_images=10):
    
    # convert the image from menpo Image to fg2015 Image (channels at front)
    i = convert_from_menpo(i)
    
    i.rescale_landmarks_to_diagonal_range(200)
    i.crop_to_landmarks_proportion_inplace(0.5)
    labeller(i, 'PTS', ibug_face_68)
    if i.n_channels == 3:
        i = i.as_greyscale(mode='average')
    test_images.append(i)



### Active Pictorial Structure

##### Build

# In[18]:

# Star tree
adjacency_array = np.empty((67, 2), dtype=np.int32)
for i in range(68):
    if i < 34:
        adjacency_array[i, 0] = 34
        adjacency_array[i, 1] = i
    elif i > 34:
        adjacency_array[i-1, 0] = 34
        adjacency_array[i-1, 1] = i

root_vertex = 34


# In[19]:

# MST tree
adjacency_array = np.array([[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6], [ 6,  7], [ 7,  8], [ 8,  9], 
                            [ 8, 57], [ 9, 10], [57, 58], [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67], 
                            [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53], [13, 14], [62, 61], [62, 51],
                            [53, 64], [14, 15], [61, 49], [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
                            [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35], [29, 30], [29, 28], [28, 27],
                            [27, 22], [27, 21], [22, 23], [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
                            [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38], [45, 46], [38, 39], [46, 47],
                            [39, 40], [47, 42], [40, 41], [41, 36]])
root_vertex = 0


# In[20]:

from menpofast.feature import no_op
from antonakoscvpr2015.builder import APSBuilder

aps = APSBuilder(adjacency_array=adjacency_array, 
                 root_vertex=root_vertex, 
                 patch_shape=(17, 17),
                 features=no_op, 
                 normalize_patches=False,
                 normalization_diagonal=100,
                 n_levels=2, 
                 downscale=2, 
                 scaled_shape_models=False,
                 max_shape_components=25,
                 n_appearance_parameters=50).build(training_images, group='ibug_face_68', verbose=True)


##### Test

# In[23]:

from antonakoscvpr2015.fitter import LucasKanadeAPSFitter
from antonakoscvpr2015.algorithm import Forward, Inverse

sampling_mask = np.require(np.zeros((17, 17)), dtype=np.bool)
#sampling_mask[1::4, 1::4] = True
sampling_mask[:] = True
                  
fitter = LucasKanadeAPSFitter(aps, algorithm=Inverse, n_shape=[3, 10], 
                              sampling_mask=sampling_mask)


# In[24]:

np.random.seed(seed=1)

fitting_results = []

for j, i in enumerate(test_images):
    
    gt_s = i.landmarks['ibug_face_68'].lms
    s = fitter.perturb_shape(gt_s, noise_std=0.04)
    
    fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=20)
    
    fitting_results.append(fr)
    
    print 'Image: ', j
    #print fr


# In[25]:

from menpofit.visualize import visualize_fitting_results

visualize_fitting_results(fitting_results)


# In[21]:




# In[ ]:


