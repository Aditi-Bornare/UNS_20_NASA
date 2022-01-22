import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.fftpack import ifft
from scipy import fft
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from itertools import chain 
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
# Reading images in grayscale
oilydata = cv2.imread("oily(6).jpeg",cv2.IMREAD_GRAYSCALE)
drydata = cv2.imread("dry(35).jpg",cv2.IMREAD_GRAYSCALE)
fig , axs = plt.subplots(1,2,figsize=(10,10))
axs[0].imshow(drydata,cmap='gray')
axs[0].set_title('dry')
axs[1].imshow(oilydata,cmap='gray')
axs[1].set_title('oily')
plt.show()

# To transform images into signals we reduce image sizes 
# and used a flatten fucntion to convert them from 2-D into 1-D array

resize_dimg = cv2.resize(drydata,(50,50))
resize_oimg = cv2.resize(oilydata,(50,50))
dflatten_list = list(chain.from_iterable(resize_dimg)) 
oflatten_list = list(chain.from_iterable(resize_oimg)) 
fig,axs = plt.subplots(2,1,figsize=(12,4),sharex=True)
axs[0].plot(dflatten_list)
axs[0].set_title('DRY')
axs[1].plot(oflatten_list)
axs[1].set_title('OILY')
plt.show()

# comparing probability distributions
fig,axs = plt.subplots(1,2,figsize=(12,4),sharex=True)
sns.distplot(dflatten_list,ax=axs[0],color='Red').set_title('DRY')
sns.distplot(oflatten_list,ax=axs[1],color='Green').set_title('OILY')
plt.show()

# We are looking for small hiden informations on these 
# images so the ideal type of filter is LOWPASS FILTER.
# PSD: Power Spectral Density
dfreqs, dpsd = signal.welch(dflatten_list)
ofreqs, opsd = signal.welch(oflatten_list)
fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
axs[0].semilogx(dfreqs, dpsd,color ='r')
axs[1].semilogx(ofreqs, opsd,color ='g')
axs[0].set_title('PSD: DRY')
axs[1].set_title('PSD: OILY')
axs[0].set_xlabel('Frequency')
axs[1].set_xlabel('Frequency')
axs[0].set_ylabel('Power')
axs[1].set_ylabel('Power')
plt.show()
sos = signal.iirfilter(3, Wn=0.01, rs=0.06 ,fs=100,btype='lp',output='sos',
                       analog=False, ftype='cheby2')
w, h = signal.sosfreqz(sos, worN=100)

# Freq response
plt.subplot(2, 1, 1)
db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
plt.plot(w, db)
plt.ylim(-75, 5)
plt.grid(True)
plt.yticks([0, -20, -40, -60])
plt.ylabel('Gain [dB]')
plt.title('Frequency Response')
plt.subplot(2, 1, 2)
plt.plot(w/np.pi, np.angle(h))
plt.grid(True)
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
           [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
plt.xlabel('Normalized frequency (1.0 = Nyquist)')
plt.show()

# Step response
t, s = signal.step(sos)
fig,axs = plt.subplots(1,1,figsize=(7,3),sharey=True)
axs.semilogx(t, s,color ='g')
axs.set_title('PSD')
axs.set_xlabel('Frequency')
axs.set_ylabel('Power')
plt.show()
