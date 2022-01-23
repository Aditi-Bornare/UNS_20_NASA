from django.http import HttpResponse
from django.shortcuts import render, redirect
from . import forms
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.fftpack import ifft
from scipy import fft
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns
from itertools import chain
from sklearn.model_selection import train_test_split


# Create your views here.

def plots_view(request):
    return render(request, 'ClassifySkin/plots.html')

def index_view(request):
    if request.method == 'POST':
        form = forms.SkinForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            return redirect('classification_view')
    else:
        form = forms.SkinForm()
    return render(request, 'ClassifySkin/index.html', {'form' : form})

def classification_view(request):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    IM_SIZE = 90
    from tensorflow import keras
    model = keras.models.load_model('static/model')

    def autocorrelation(x):
        xp = np.fft.fftshift((x - np.average(x))/np.std(x))
        n, = xp.shape
        xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
        f = fft.fft(xp)
        p = np.absolute(f)**2
        pi = ifft(p)
        return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

    def kLargest(arr, k):
        max_ks = []
        # Sort the given array arr in reverse
        # order.
        arr = np.sort(arr)[::-1]
        #arr.sort(reverse = True)
        # Print the first kth largest elements
        for i in range(k):
            max_ks.append(arr[i])

        return max_ks

    def rms(x):
        return np.sqrt(np.mean(x**2))

<<<<<<< HEAD
=======
    # Autocorrelation
    print(f'Tree max values of autocorrelation\noily FILTERED:   {kLargest(autocorrelation(ofiltered), 3)}\n\ndry FILTERED:{kLargest(autocorrelation(dfiltered), 3)}')

    # Finding Peaks
    ppeaks, _  = signal.find_peaks(dpsd)
    npeaks, _  = signal.find_peaks(opsd)
    print(f'Ten max peaksof filtered PSD\ndry:   {kLargest(ppeaks,10)}\noily peaks:{kLargest(npeaks,10)}')

    # Root Mean Square and Mean of filtered signals
    print(f'Root Mean Square\n--------------------\noily : {rms(ofiltered)}\ndry : {rms(dfiltered)}')
    print('--'*10)
    print(f'Mean \n--------------------\noily : {np.mean(dflatten_list)}\ndry : {np.mean(ofiltered)}')

    # Inverse Fast Fourier Transform
    fig,axs = plt.subplots(1,2,figsize=(12,4),sharey=True)
    offt = fft.fft(oflatten_list)
    dfft = fft.fft(dflatten_list)
    axs[0].plot(dfft.real, 'b-')
    axs[0].plot(dfft.imag, 'r--')
    axs[0].set_title('DRY')
    axs[1].plot(offt.real, 'b-')
    axs[1].plot(offt.imag, 'r--')
    axs[1].set_title('OILY')
    plt.legend(('real', 'imaginary'))
    #plt.savefig("plot10")

    # Magnitude Spectrum for oily
    dft = cv2.dft(np.float32(oilydata),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.figure(figsize=(11,6))
    plt.subplot(121),plt.imshow(oilydata, cmap = 'gray')
    plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.savefig("plot11")

    # Magnitude Spectrum for dry
    dft = cv2.dft(np.float32(drydata),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.figure(figsize=(11,6))
    plt.subplot(121),plt.imshow(drydata, cmap = 'gray')
    plt.title('ORIGINAL'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #plt.savefig("plot13")

    # Apply Signal Processing to all images
    mpath = "static/images/dataset/"
    IM_SIZE = 90
    def get_img(folder):
        X = []
        y = []
        for xr in os.listdir(folder):
            if not xr.startswith('.'):
                if xr in ['oily']:
                    label = 0
                elif xr in ['dry']:
                    label = 1
                for filename in tqdm(os.listdir(folder + xr)):
                    im_array = cv2.imread(folder + xr +'/'+ filename,cv2.IMREAD_GRAYSCALE)
                    if im_array is not None:
                        img = cv2.resize(im_array,(IM_SIZE,IM_SIZE))
                        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
                        dft_shift = np.fft.fftshift(dft)
                        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
                        msd = np.std(magnitude_spectrum)
                        new_image = cv2.Laplacian(img,cv2.CV_64F)
                        lpvar = abs(np.max(new_image) - np.min(new_image))/np.max(new_image)
                        #flatten the image
                        flatten_list = list(chain.from_iterable(img))
                        #filtering
                        sos = signal.iirfilter(3, Wn=0.01, rs=0.5 ,fs=100,btype='lp',output='sos',
                           analog=False, ftype='cheby2')
                        filtered = signal.sosfilt(sos, flatten_list)
                        #power Spectral density
                        _, psd = signal.welch(filtered)
                        #find peaks of PSD
                        peaks, _  = signal.find_peaks(psd)
                        maxPeaks  = kLargest(peaks, k=6)
                        #mean and rms
                        Mean = np.mean(flatten_list)
                        Rms = rms(filtered)
                        # autocorrelation
                        auto= autocorrelation(filtered)
                        maxauto = kLargest(auto, k=5)
                        #fft
                        invfft = fft.fft(filtered)
                        vfl = np.std(flatten_list)
                        invfft_r_peaks, _  = signal.find_peaks(invfft.real)
                        invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)
                        maxinvfft_r_peaks  = kLargest(invfft_r_peaks, k=6)
                        maxinvfft_imag_peaks  = kLargest(invfft_imag_peaks, k=6)
                        #peaks of periodogram filtered
                        _, Pxx_den = signal.periodogram(filtered,100)
                        Perio_Peaks, _  = signal.find_peaks(Pxx_den)

                        maxPerio_Peaks  = kLargest(Perio_Peaks, k=6)
                        total = maxPeaks + [Rms,Mean,lpvar,msd,vfl]
                        total = total + maxPerio_Peaks
                        total = total + maxinvfft_r_peaks
                        total = total + maxinvfft_imag_peaks
                        total = total + maxPeaks
                        total = total + maxauto
                        X.append(total)
                        y.append(label)
        y = np.array(y)
        return np.array(X),y
    X,y = get_img(mpath)

    X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size = 0.2)
    #Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mmscaler = MinMaxScaler()
    X_train_01 = mmscaler.fit_transform(X_train)
    X_test_01 = mmscaler.transform(X_test)

    # Training the model
    import tensorflow as tf
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train_01, y_train, epochs=200)

    val_loss, val_acc = model.evaluate(X_test_01, y_test)
    print(f'Validation loss: {val_loss}')
    print(f'Validation accuracy: {val_acc}')

    train_loss, train_acc = model.evaluate(X_train_01, y_train)
    print(f'Training loss: {train_loss}')
    print(f'Training accuracy: {train_acc}')
>>>>>>> b35fb698533ec9b7f7f8523349602fb9b75923be

    def get_img2(folder):
        X=[]
        filename=os.listdir(folder)
        # print(filename)
        img=cv2.imread(folder + '/' + filename[0],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IM_SIZE,IM_SIZE))
        dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        msd = np.std(magnitude_spectrum)
        new_image = cv2.Laplacian(img,cv2.CV_64F)
        lpvar = abs(np.max(new_image) - np.min(new_image))/np.max(new_image)
        #flatten the image
        flatten_list = list(chain.from_iterable(img))
        #filtering
        sos = signal.iirfilter(3, Wn=0.01, rs=0.5 ,fs=100,btype='lp',output='sos',
           analog=False, ftype='cheby2')
        filtered = signal.sosfilt(sos, flatten_list)
        #power Spectral density
        _, psd = signal.welch(filtered)
        #find peaks of PSD
        peaks, _  = signal.find_peaks(psd)
        maxPeaks  = kLargest(peaks, k=6)
        #mean and rms
        Mean = np.mean(flatten_list)
        Rms = rms(filtered)
        # autocorrelation
        auto= autocorrelation(filtered)
        maxauto = kLargest(auto, k=5)
        #fft
        invfft = fft.fft(filtered)
        vfl = np.std(flatten_list)
        invfft_r_peaks, _  = signal.find_peaks(invfft.real)
        invfft_imag_peaks, _  = signal.find_peaks(invfft.imag)
        maxinvfft_r_peaks  = kLargest(invfft_r_peaks, k=6)
        maxinvfft_imag_peaks  = kLargest(invfft_imag_peaks, k=6)
        #peaks of periodogram filtered
        _, Pxx_den = signal.periodogram(filtered,100)
        Perio_Peaks, _  = signal.find_peaks(Pxx_den)

        maxPerio_Peaks  = kLargest(Perio_Peaks, k=6)
        total = maxPeaks + [Rms,Mean,lpvar,msd,vfl]
        total = total + maxPerio_Peaks
        total = total + maxinvfft_r_peaks
        total = total + maxinvfft_imag_peaks
        total = total + maxPeaks
        total = total + maxauto
        X.append(total)
        return np.array(X)

    #Make the prediction
    npath = "media/images"
    uX= get_img2(npath)
    # print(uX)
    scaler = StandardScaler()
    uX = scaler.fit_transform(uX)
    mmscaler = MinMaxScaler()

    X_train_01 = mmscaler.fit_transform(uX)
    X_test_01 = mmscaler.transform(uX)
    y_pred = model.predict(X_test_01)
    print(y_pred)
    if y_pred[0][0]>=y_pred[0][1]:
        result="Dry"
    else:
        result="Oily"
    filename=os.listdir(npath)
    os.remove(npath + '/' +filename[0])
    return render(request, 'ClassifySkin/success.html', {'result' : result})
