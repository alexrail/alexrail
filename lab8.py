import numpy as np 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.signal import convolve2d

# Gabor filters
# build gabor filter
# size of the filter
# wavelength amount of waves through it 
# orientation 

def gabor_filter(size, wavelength, orientation, monitor):
    # size = filter size in pixels (w = h)
    # wavelength = wavelength of sinusoid relative to half the size of the filter
    # orientation = desired orientation of filter
        
    # set parameters (based on neural data from V1)
    lambda_ = size * 2. / wavelength # wavelength of sinusoidal plane wave (how many waves 'fit' in the filter)
    sigma = lambda_ * 0.8 # standard deviation of gaussian kernel
    gamma = 0.3  # spatial aspect ratio of gaussian kernel
    theta = np.deg2rad(orientation + 90) # orientation  
    
    # gaussian
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    gauss = np.exp(-(rotx**2 + gamma**2 * roty**2) / (2 * sigma ** 2))
    
    # sinusoid 
    sinusoid = np.cos(2 * np.pi * rotx / lambda_)
    
    # gabor
    ### create gabor filter from gaussian and sinusoid (1 line of code) ###
    filt = gauss * sinusoid
    filt[np.sqrt(x**2 + y**2) > (size / 2)] = 0
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))
    
    # show 
    if monitor:
        plt.figure(); plt.imshow(gauss, cmap=plt.gray()); plt.title('2D Gaussian kernel')
        plt.figure(); plt.imshow(sinusoid, cmap=plt.gray()); plt.title('Sinusoidal plane wave')
        plt.figure(); plt.imshow(filt, cmap=plt.gray()); plt.title('Gabor filter')
    
    
    return filt 

# generate a filter bank
sizes = np.arange(7, 39, 2) # 16 sizes
wavelengths = np.arange(4, 3.2, -0.05) # 16 associated wavelengths
orientations = np.arange(-45, 135, 45) # 4 orientations
params = []
i = 0;
for s in sizes:
    i = i + 1
    for o in orientations:        
        w = wavelengths[i-1]
        params.append((s,w,o))
filterBank = []
gaborParams = []
for (size, wavelength, orientation) in params:
    gaborParam = {'size':size, 'wavelength':wavelength, 'orientation':orientation, 'monitor':0}
    filt = gabor_filter(**gaborParam)
    filterBank.append(filt)
    gaborParams.append(gaborParam)

# show filter bank
plt.figure()
n = len(filterBank)
for i in range(n):
    plt.subplot(16,4,i+1)
    plt.axis('off'); plt.imshow(filterBank[i],cmap=plt.gray())

plt.show()

# read in images
face = rgb2gray(plt.imread('images/image19_66440000_rs.jpg'))
zebra = rgb2gray(plt.imread('images/image32_41110000_rs.jpg'))

# show image and filter
plt.figure(); plt.imshow(zebra, cmap=plt.gray()); plt.title('image') 
filt = filterBank[60] 
plt.figure(); plt.imshow(filt, cmap=plt.gray()); plt.title('filter') 

# convolve filter with image (convolution = dot product)
#time 
output = convolve2d(zebra, filt, mode='valid') 
plt.figure(); plt.imshow(output, cmap=plt.gray()); plt.title('response') 
plt.show()

# # show image and filter
# plt.figure(); plt.imshow(face, cmap=plt.gray()); plt.title('image') 
# filt = filterBank[30] 
# plt.figure(); plt.imshow(filt, cmap=plt.gray()); plt.title('filter') 

# # convolve filter with image (convolution = dot product)
# #time 
# output = convolve2d(face, filt, mode='valid') 
# plt.figure(); plt.imshow(output, cmap=plt.gray()); plt.title('response') 
# plt.show()


# filt = gabor_filter(23, 3.6, -45, 1) to test the function


# generate a filter with the following parameters: 
# size = 23, wavelength = 3.6, orientation = -45 degrees
### 1 line of code ###
	