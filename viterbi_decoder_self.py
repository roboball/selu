#!/usr/bin/env python
#coding: utf-8

'''
links:
https://en.wikipedia.org/wiki/Viterbi_algorithm
https://github.com/phvu/misc/blob/master/viterbi/viterbi.py
https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
'''

import numpy as np

########################################################################
# init HMM params:
########################################################################

# num of states: 2
states = 2
# num of observations: 3
obs = 3
# pi = prior start probs: 2
start_p = np.array([[ 0.6],[0.4]])
# A = transition probs: (2x2)
trans_p = np.array([[ 0.7, 0.3],[ 0.4, 0.6]])
# B = emission probs: (2states x 3obs)
emit_p = np.array([[0.5, 0.4, 0.1],[0.1, 0.3, 0.6]])



########################################################################
# init functions:
########################################################################
                     
                     
def viterbi(obs, states, start_p, trans_p, emit_p):
	'''Viterbi Decoder'''
	
	# create trellis matrix
	trellis = np.copy(emit_p)
	# calculate start states
	trellis[:,[0]] = np.multiply(start_p, emit_p[:,[0]])
	print(trellis)
	backpt = np.ones_like(trellis, 'int32') * -1
	print(backpt)
	#~ for i in range(1,2):
		#~ trellis[:,[i]] = np.multiply(trans_p[:,[i]], emit_p[:,[i]])
		#~ print(trellis)
		
