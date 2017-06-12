#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
name:    selu for tensorflow
purpose: activation function (for neural networks)
@author: Roboball
links:   https://arxiv.org/abs/1706.02515
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import os
# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#init global params
ALPHA = 1.6733
LAMBDA = 1.0507

def selu(x):
	'''forward pass'''
	if x <= 0.0:
		return LAMBDA * (ALPHA * np.exp(x) - ALPHA)
	else:
		return LAMBDA * x
		
def d_selu(x):
	'''forward pass'''
	if x <= 0.0:
		return LAMBDA * ALPHA * np.exp(x)
	else:
		return LAMBDA

#vectorize functions
np_selu = np.vectorize(selu)
np_d_selu = np.vectorize(d_selu)
np_selu_32 = lambda x: np_selu(x).astype(np.float32)
np_d_selu_32 = lambda x: np_d_selu(x).astype(np.float32)

def tf_selu(x, name=None):
	'''forward pass for tf'''
	with ops.name_scope(name, "selu",[x]) as name:
		y = py_func(np_selu_32,
						[x],
						[tf.float32],
						name=name,
						grad=selugrad)  # call the gradient
		return y[0]

def tf_d_selu(x,name=None):
	'''backward pass for tf'''
	with ops.name_scope(name, "d_selu",[x]) as name:
		y = tf.py_func(np_d_selu_32,
						[x],
						[tf.float32],
						name=name,
						stateful=False)
		return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
	'''get gradients from graph'''
	# Need to generate a unique name to avoid duplicates:
	rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
	tf.RegisterGradient(rnd_name)(grad)  
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": rnd_name}):

		return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        
def selugrad(op, grad):
	'''gradient function'''
	x = op.inputs[0]
	n_gr = tf_d_selu(x)
	return grad * n_gr  


########################################################################
#final test the function
########################################################################
#init all and start session:
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#init data
x = tf.constant([0.2,-0.7,1.2,-1.7])
y = tf_selu(x)
#print tensorflow functions
print('tensorflow functions:')
print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())


#optional: print numpy functions
x2 = np.array([0.2,-0.7,1.2,-1.7]).astype(np.float32)
print('numpy functions:')
print(x2, np_selu(x2), np_d_selu(x2))





        
        
        
        
        
        
        
        
        
        
        
        
