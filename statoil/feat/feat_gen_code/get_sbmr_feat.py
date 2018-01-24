import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_json('../../input/train.json')
test = pd.read_json('../../input/test.json')
from subprocess import check_output
# Isolation function.
def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])

# Size in number of pixels of every isolated object.
def size(arr):     
    return np.sum(arr<-5)

# Isolation function.
def isoval(arr):
    image = img_as_float(np.reshape(np.array(arr), [75,75]))
    image = gaussian_filter(image,2)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image 
    dilated = reconstruction(seed, mask, method='dilation')
    return image-dilated
#Adding up every pixel value is equivalent to the volumen under the surface. 
def volume(arr):
    return np.sum(arr)

# Feature engineering iso1 and iso2.
#train['isoval1'] = train.iloc[:, 0].apply(isoval)
#train['isoval2'] = train.iloc[:, 1].apply(isoval)
train['vol'] = (train.iloc[:, 0] + train.iloc[:, 1]).apply(volume)
# Feature engineering iso1 and iso2.
train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
train['s1'] = train.iloc[:,5].apply(size)
train['s2'] = train.iloc[:,6].apply(size)
train['size'] = train.s1+train.s2
train[['id','s1','s2','size','vol']].to_csv('../sbmr_train.csv',index=False)

# Feature engineering iso1 and iso2.
#test['isoval1'] = test.iloc[:, 0].apply(isoval)
#test['isoval2'] = test.iloc[:, 1].apply(isoval)
test['vol'] = (test.iloc[:, 0] + test.iloc[:, 1]).apply(volume)
test['iso1'] = test.iloc[:, 0].apply(iso)
test['iso2'] = test.iloc[:, 1].apply(iso)
# Feature engineering s1 s2 and size.
test['s1'] = test.iloc[:,5].apply(size)
test['s2'] = test.iloc[:,6].apply(size)
test['size'] = test.s1+test.s2
test[['id','s1','s2','size','vol']].to_csv('../sbmr_test.csv',index=False)
