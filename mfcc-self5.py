#!/usr/bin/env python
#coding: utf-8
'''

# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
values: -32768 to 32767, 16 bit, max 0dB, SNR 96.33 dB

'''
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import sounddevice as sd
import time


# init global params
frame_time = 25
step_time = 10
num_bins = 26 #26-40 Mel bins
low_freq = 0
num_fft = 512
lift_factor = 22
delta_factor = 2
fs = 16000  #Hz
duration = 5  # seconds



def loadrawdata(filename):
	"""load data from binary .raw format"""
	with open(filename, 'rb') as fid:
		datafile = np.fromfile(fid, dtype=np.int16) #get frames
	return datafile
	
def wavrecord(fs,duration = 3, ch = 1):
	"""wave recorder from mic to numpy arrays"""
	print('start speaking for '+str(duration)+' sec')	
	recording = sd.rec(int(duration * fs), samplerate=fs, channels=ch, blocking=True, dtype='int16')
	sd.wait()
	return recording

def wavplayer(audio,fs):
	"""wave player for numpy arrays"""
	sd.play(audio, fs, blocking=True)
	sd.stop()
	
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
  
def preemphasis(signal,coeff=0.97):
    """perform pre-emphasis"""    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def winfft(audio, fs, frame_time=25,step_time=10, window=np.hamming):
	"""apply sliding window, compute FFT"""
	magfft_list = []
	powfft_list = []
	audio_len = len(audio)
	frame_size = int(fs * (frame_time * (1.0/1000)))
	frame_step = int(fs * (step_time* (1.0/1000)))
	
	win = window(frame_size)
	num_fft = 0
	k = 0
	#get fft size
	while frame_size > num_fft:
		num_fft = np.power(2,k) 
		k+=1	
	pos = 0
	#zero pad ftt
	while((pos * frame_step + frame_size) < audio_len):
		pos +=1
	diff = pos * frame_step + frame_size - audio_len
	audio_padded = np.append(audio, np.zeros(diff,np.int16))
	#compute fft for each chunk
	for posx in range(pos+1):
		audio_chunk = audio_padded[posx * frame_step : 
			posx * frame_step + frame_size].astype(np.float64)
		#audio_chunk *= win
		magfft_chunk = np.absolute(np.fft.rfft(audio_chunk,num_fft))
		powfft_chunk = (1.0/num_fft) * np.square(magfft_chunk)	
		magfft_list.append(magfft_chunk)
		powfft_list.append(powfft_chunk)	
	return magfft_list,powfft_list, win, num_fft 

def melbank(fs, numbins=26,lowfreq=0, num_fft=None):
	"""create Mel-filterbank (26 standard, or 26-40 bins)"""
	highfreq = fs/2
	mel_low = 2595 * np.log10(1+lowfreq/700.)
	mel_high  = 2595 * np.log10(1+highfreq/700.)
	mel_peaks = np.linspace(mel_low,mel_high,numbins+2)
	hz_peaks = 700*(10**(mel_peaks/2595.0)-1)
	bin = np.floor((num_fft+1)*hz_peaks/fs)
	#create mel-bank
	melbank = np.zeros([numbins,num_fft//2+1])
	for j in range(0,numbins):
		for i in range(int(bin[j]), int(bin[j+1])):
			melbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
		for i in range(int(bin[j+1]), int(bin[j+2])):
			melbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
	return melbank 
	
def melspec(powfft_list, melbank):
	"""compute 26-40 log-mel spectral + 1 log-energy features"""
	melspec_list = []
	energy_list = []
	for powfft in powfft_list:
		#log-melspectrals
		melspec = np.dot(powfft,melbank.T) 
		mfcc = np.where(melspec == 0,np.finfo(float).eps,melspec)
		log_melspec = np.log(melspec)
		melspec_list.append(log_melspec)		
		#log-energy
		energy = np.sum(powfft,0) 
		energy = np.where(energy == 0, np.finfo(float).eps,energy)
		log_energy = np.log(energy)
		energy_list.append(log_energy)		
	return melspec_list, energy_list 
	
def mfcc(melspec_list,numcep=12):
	"""compute 12 MFCC features"""
	mfcc_list = []
	for melspec in melspec_list:
		mfcc = dct(melspec, type=2, axis=0, norm='ortho')[:numcep]
		mfcc_list.append(mfcc)
	return mfcc_list

def dctopt(mfcc_list, lift_factor=22):
	"""increase magnitude of high frequency DCT coeffs"""
	mfcclift_list = []
	for mfcc in mfcc_list:
		if lift_factor > 0:
			ncoeff = len(mfcc)
			n = np.arange(ncoeff)
			lift = 1 + (lift_factor/2.)*np.sin(np.pi*n/lift_factor)
			mfcclift_list.append(lift*mfcc)
		else:
			# values of lift_fac <= 0, do nothing
			mfcclift_list.append(mfcc)
	return mfcclift_list

def spectrogram(energy_list, mfcclift_list):
	"""init spectrogram with 13 MFCC features"""
	mfcc_len = len(energy_list)
	spectrogram = np.zeros([39,mfcc_len])
    #fill in 13 MFCCs
	for pos in range(len(mfcclift_list)):
		#mfccs = np.append(energy_list[pos],mfcclift_list[pos])
		mfcclift_list[pos][0] = energy_list[pos]
		mfccs = mfcclift_list[pos]
		spectrogram[26:39,[pos]] = np.rot90(mfccs[:,np.newaxis],2)
	return spectrogram
		
	
def delta(spectogram, N):
	"""compute 13 delta + 13 delta_delta features"""
	
	
	#features = np.rot90(spectrogram[26:39,[-1]],2)
	features = spectrogram[26:39,[-1]]
	#~ print(features)
	#~ print(features.shape)
	if N < 1:
		raise ValueError('N must be an integer >= 1')
	frame_num = 13
	denominator = 2 * sum([i**2 for i in range(1, N+1)])
	delta_feat = np.zeros_like(spectrogram[26:39,:].T)
	#~ print(frame_num)
	#~ print('denom:',denominator)
	print('delta_feat',delta_feat.shape)
	
	padded = np.pad(spectrogram[26:39,:].T, ((N, N), (0, 0)), mode='edge')   # padded version of feat
	print(padded)
	print(padded.shape)
	
	# numerical gradient:
	for t in range(frame_num):
		delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
		#~ print(np.arange(-N, N+1))
		#~ print(padded[t : t+2*N+1])
	print(delta_feat[t])	
		
	return delta_feat


#=======================================================================
# read in input
#=======================================================================

# read in audio: .wav files
#wavname = 'artos_ofenrohr_16k.wav'
wavname = 'english.wav'
(fs,audio) = wav.read(wavname)

# read in audio: .raw files	
filename = 'is1a0001_043.raw'
#filename = 'rolladen_runter.raw'
#~ audio = loadrawdata(filename)

# record audio: extranal mic
#~ audio = wavrecord(fs, duration)

#=======================================================================
# MFCC processing pipeline
#=======================================================================	

# preemphasis
audio_pre = preemphasis(audio)
# windowing, fft
magfft_list,powfft_list, win, num_fft = winfft(audio_pre, fs, frame_time, step_time)
# mel filterbank
melbank = melbank(fs, num_bins, low_freq, num_fft)
# log-melspec + log-energy features
melspec_list, energy_list  = melspec(powfft_list, melbank)
# 12 MFCC features
mfcc_list = mfcc(melspec_list,numcep=13)
#optimize MFCC features:
mfcclift_list = dctopt(mfcc_list,lift_factor)

#init spectrogramm with 13 MFCC features:
spectrogram = spectrogram(energy_list, mfcclift_list)
print(spectrogram[:,[-2]])
print(spectrogram.shape)
# 13 delta + 13 delta_delta features
delta(spectrogram, delta_factor)

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
#plotting(win, fs,2,211,'hamming window','samples')
#~ plotting(magfft_list[0], fs,3,211,'fft magnitude','samples')
#~ plotting(powfft_list[0], fs,3,212,'fft power','samples')

plt.show()
plt.clf()
