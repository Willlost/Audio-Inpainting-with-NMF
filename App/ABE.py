import librosa, librosa.display, soundfile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF

class ABE:
    def __init__(self,X,Y,GroundTruth):
        self.x = X[0] #test audio signal
        self.y = Y[0] #training audio signal
        self.x_sr = X[1] #test audio sample rate
        self.y_sr = Y[1] #training audio sample rate
        self.x_mel = librosa.feature.melspectrogram(y=self.x,sr=self.x_sr) #test melspectrogram
        self.y_mel = librosa.feature.melspectrogram(y=self.y,sr=self.y_sr) #training melspectrogram
        self.yhat = self.x #inpainted audio signal
        self.yhat_mel = self.x_mel #inpainted melspectrogram
        self.GT = GroundTruth[0] #ground truth audio signal
        self.GT_mel = librosa.feature.melspectrogram(y=self.GT,sr=GroundTruth[1]) #ground truth melspectrogram
    
    def apply(self,mode='NMF'):
        if mode == 'NMF':
            #resample test data at 44100hz
            x = librosa.resample(self.x,orig_sr=self.x_sr,target_sr=self.y_sr)
            #transform the resampled test data into a Mel-spectrogram
            self.x_mel = librosa.feature.melspectrogram(y=x,sr=self.y_sr)

            #define the NMF model
            model = NMF(
                init='random',
                random_state=1,
                solver='mu',
                beta_loss='kullback-leibler',
                max_iter=1000,
                tol=1e-4
            )
            #apply the model on both Mel-spectrograms
            W = model.fit_transform(self.y_mel)
            _ = model.fit_transform(self.x_mel)

            #derive the low-rank approximation matrix of the training data
            self.yhat_mel = W@model.components_
            #transform the low-rank approximation matrix into time series with the GL algorithm
            self.yhat = librosa.feature.inverse.mel_to_audio(self.yhat_mel,sr=self.y_sr)

            #pad lost samples with 0
            self.yhat = np.pad(self.yhat, (0,self.y.size-self.yhat.size), 'constant')
            #transform the inpainted time series into Mel-spectrogram
            self.yhat_mel = librosa.feature.melspectrogram(y=self.yhat,sr=self.y_sr)

            return (self.yhat_mel, self.yhat, self.y_sr)
        
        else: #if the specified inpainting mode is not supported
            raise ValueError(f'Mode {mode} is not supported. \nCurrent available options are: \n\tNMF')
    
    def save(self,path,sig,sr=44100):
        soundfile.write(path,sig,int(sr))

    def specshow(self, save=False):
        plt.subplot(1,3,1)
        spec = librosa.display.specshow(librosa.power_to_db(self.GT_mel),sr=self.y_sr)
        cmap = spec.get_cmap()

        plt.subplot(1,3,2)
        librosa.display.specshow(librosa.power_to_db(self.x_mel),sr=self.x_sr,cmap=cmap)
        
        plt.subplot(1,3,3)
        librosa.display.specshow(librosa.power_to_db(self.yhat_mel),sr=self.y_sr,cmap=cmap)
        
        if save == True:
            plt.savefig('log_mel_spectrograms.png')
        
        plt.show()
    
    def loss(self,mode=['l1']):
        if not isinstance(mode,list):
            raise TypeError(f"\'mode\' should be a list, but {type(mode)} was given")
        
        wave_loss = {}
        spec_loss = {}

        GT_mel = librosa.power_to_db(self.GT_mel)
        yhat_mel = librosa.power_to_db(self.yhat_mel)

        for i in mode:
            if i == 'l1':
                n = self.y.size
                wave_loss['L1 Loss'] = np.sum(np.absolute(self.GT-self.yhat))/n
                spec_loss['L1 Loss'] = np.sum(np.absolute(GT_mel-yhat_mel))/n

            elif i == 'l2':
                n = self.y.size
                wave_loss['L2 Loss'] = np.sum(np.absolute(self.GT-self.yhat)**2)/n
                spec_loss['L2 Loss'] = np.sum(np.absolute(GT_mel-yhat_mel)**2)/n
            
            elif i == 'esr':
                n = self.y.size
                wave_loss['ESR'] = np.sum(np.absolute(self.yhat-self.GT)**2)/np.sum(np.absolute(self.GT)**2)
                spec_loss['ESR'] = np.sum(np.absolute(yhat_mel-GT_mel)**2)/np.sum(np.absolute(GT_mel)**2)
            
            else:
                raise ValueError(f'Mode {i} is not supported. \nCurrent available options are: \n\tl1\n\tl2\n\tesr')
        
        return (wave_loss,spec_loss)

