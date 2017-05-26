import numpy as np
import os
import pyaudio


def loadRawData(filename):
	"""load data from binary .raw format"""
	with open(filename, 'rb') as fid:
		datafile = np.fromfile(fid, dtype=np.int16) #get frames
	return datafile

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

filename = 'is1a0001_043.raw'
fs = 16000
audio = loadRawData(filename)
wavplayer(audio,fs)
