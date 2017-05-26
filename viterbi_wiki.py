#!/usr/bin/env python
#coding: utf-8

'''
links:
https://en.wikipedia.org/wiki/Viterbi_algorithm
https://github.com/phvu/misc/blob/master/viterbi/viterbi.py
https://phvu.net/2013/12/06/sweet-implementation-of-viterbi-in-python/
'''

########################################################################
# init HMM params:
########################################################################

# num of states: 2
states = ('Healthy', 'Fever')
# num of observations: 3
obs = ('normal', 'cold', 'dizzy')
# pi = prior start probs: 2
start_p = {'Healthy': 0.6, 'Fever': 0.4}
# A = transition probs: (2x2)
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
# B = emission probs: (2states x 3obs)
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }
                   

########################################################################
# init functions:
########################################################################
                     
                     
def viterbi(obs, states, start_p, trans_p, emit_p):
	'''Viterbi Decoder'''
	V = [{}]
	for st in states:
		V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
	# Run Viterbi when t > 0
	for t in range(1, len(obs)):
		V.append({})
		for st in states:
			max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
			for prev_st in states:
				if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
					max_prob = max_tr_prob * emit_p[st][obs[t]]
					V[t][st] = {"prob": max_prob, "prev": prev_st}
					break
	print ('Dynamic Programming Table')
	for line in dptable(V):
		print (line)
	opt = []
	# The highest probability
	max_prob = max(value["prob"] for value in V[-1].values())
	previous = None
	# Get most probable state and its backtrack
	for st, data in V[-1].items():
		if data["prob"] == max_prob:
			opt.append(st)
			previous = st
			break
	
	# Follow the backtrack till the first observation
	for t in range(len(V) - 2, -1, -1):
		opt.insert(0, V[t + 1][previous]["prev"])
		previous = V[t + 1][previous]["prev"]
	print(' ')
	print ('The steps of states are: ' + ' '.join(opt) + '\nwith highest probability of: %s' % max_prob)

def dptable(V):
    '''Print a table of steps from dictionary'''
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


#start viterbi decoding
viterbi(obs, states, start_p, trans_p, emit_p)


