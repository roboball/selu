#!/usr/bin/env python
#coding: utf-8

# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
'''values: -32768 to 32767, 16 bit, max 0dB, SNR 96.33 dB'''
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import pyaudio
import audioop

def wavplayer(audio,fs):
	"""pyaudio wave player for numpy arrays"""
	p = pyaudio.PyAudio()
	stream = p.open(format=p.get_format_from_width(2),
					channels=1,
					rate=int(fs),
					output=True)
	bytestream = audio.tobytes()
	stream.write(bytestream)
	stream.stop_stream()
	stream.close()
	p.terminate()

def plotting(audio, fs, plotnum, subplotnum, plotname, 
			 plotx='time [s]',ploty='amplitude', colors ='b'):
	"""plot audio"""
	plt.figure(plotnum, figsize=(8,8))
	plt.subplots_adjust(wspace=0.5,hspace=0.5)
	plt.subplot(subplotnum)
	plt.plot(range(len(audio)), audio, color = colors)
	plt.axis([0,len(audio),int(np.min(audio))-1,int(np.max(audio))+1])
	if plotnum == 1:
		plt.xticks([w* fs for w in range(int(len(audio)/fs)+1)],
		           [w for w in range(int(len(audio)/fs)+1)])
	plt.title(plotname)
	plt.xlabel(plotx)
	plt.ylabel(ploty)
  
def preemphasis(signal,coeff=0.95):
    """perform pre-emphasis"""    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


def winfft(audio, fs, frame_time=25,step_time=10,numfft=None, window=np.hanning):
	"""apply sliding window, compute FFT"""
	magfft_list = []
	powfft_list = []
	audio_len = len(audio)
	frame_size = int(fs * (frame_time/1000))
	frame_step = int(fs * (step_time/1000))
	win = window(frame_size)
	
	pos = 0
	while(pos * frame_step + frame_size < audio_len):
		pos +=1
	diff = pos * frame_step + frame_size - audio_len
	audio_padded = np.append(audio, np.zeros(diff,np.int16))

	for posx in range(pos+1):
		audio_chunk = audio_padded[posx * frame_step : 
			posx * frame_step + frame_size].astype(np.float64)
		audio_chunk *= win
		magfft_chunk = np.absolute(np.fft.rfft(audio_chunk,numfft))
		powfft_chunk = (1.0/numfft) * np.square(magfft_chunk)
			
		magfft_list.append(magfft_chunk)
		powfft_list.append(powfft_chunk)
		
	return magfft_list,powfft_list, win

def melbank(fs, numbins=26,lowfreq=0, numfft=None):
	"""create Mel-filterbank (26 standard, or 26-40 bins)"""
	highfreq = fs/2
	mel_low = 2595 * np.log10(1+lowfreq/700.)
	mel_high  = 2595 * np.log10(1+highfreq/700.)
	mel_peaks = np.linspace(mel_low,mel_high,numbins+2)
	hz_peaks = 700*(10**(mel_peaks/2595.0)-1)
	bin = np.floor((numfft+1)*hz_peaks/fs)
	#create mel-bank
	melbank = np.zeros([numbins,numfft//2+1])
	for j in range(0,numbins):
		for i in range(int(bin[j]), int(bin[j+1])):
			melbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
		for i in range(int(bin[j+1]), int(bin[j+2])):
			melbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
	return melbank 

#~ def dctopt(cepstra, L=22):
    #~ """increase magnitude of high frequency DCT coeffs """
    #~ if L > 0:
        #~ nframes,ncoeff = numpy.shape(cepstra)
        #~ n = numpy.arange(ncoeff)
        #~ lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        #~ return lift*cepstra
    #~ else:
        #~ # values of L <= 0, do nothing
        #~ return cepstra  
	
def mfcc(powfft_list, melbank, numcep=13):
	"""compute 12 MFCC + 1 log-energy features"""
	spectrogram = np.zeros([5,5]) #noch falsch
	for powfft in powfft_list:
		mfcc = np.dot(powfft,melbank.T) 
		mfcc = np.where(mfcc == 0,np.finfo(float).eps,mfcc)
		log_mffc = np.log(mfcc)
		#print(mfcc.shape)
		dct_mfcc = dct(log_mffc, type=2, axis=0, norm='ortho')[:numcep]
		print(dct_mfcc.shape)
		print(dct_mfcc)
		
		#optimize dct:
		 #~ dctopt_mfcc = dctopt(dct_mfcc,lift_fac)

		energy = np.sum(powfft,0) 
		energy = np.where(energy == 0, np.finfo(float).eps,energy)
		log_energy = np.log(energy)	
				
	return spectrogram

#~ def delta(feat, N):
    #~ """compute 13 delta + 13 delta_delta features"""
    #~ if N < 1:
        #~ raise ValueError('N must be an integer >= 1')
    #~ NUMFRAMES = len(feat)
    #~ denominator = 2 * sum([i**2 for i in range(1, N+1)])
    #~ delta_feat = numpy.empty_like(feat)
    #~ padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    #~ for t in range(NUMFRAMES):
        #~ delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    #~ return delta_feat


# init params
frame_time = 25
step_time = 10
num_fft = 512
num_bins = 26
low_freq = 0

#=======================================================================
# MFCC processing pipeline
#=======================================================================

# read in audio
(fs,audio) = wav.read('artos_ofenrohr_8k.wav')
# preemphasis
audio_pre = preemphasis(audio)
# windowing, fft
magfft_list,powfft_list, win = winfft(audio, fs, frame_time, step_time, num_fft)
# mel filterbank
melbank = melbank(fs, num_bins, low_freq, num_fft)
# 12 MFCC + 1 energy features
features = mfcc(powfft_list, melbank)
# 13 delta + 13 delta_delta features
#~ delta(features, N)

#=======================================================================
# play audio
#=======================================================================

#~ wavplayer(audio,fs)
#~ wavplayer(audio3,fs)

#=======================================================================
#plot files:
#=======================================================================

#~ plotting(audio, fs,1, 211, 'audio .wav file')
#~ plotting(audio_pre, fs,1, 212,'pre-emphasis')
#~ plotting(win, fs,2,211,'hanning window','samples')
#~ plotting(magfft_list[0], fs,3,211,'fft magnitude','samples')
#~ plotting(powfft_list[0], fs,3,212,'fft power','samples')

plt.show()
plt.clf()
