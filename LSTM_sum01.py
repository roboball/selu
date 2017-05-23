#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''=======================================================================
#Purpose: Acoustic Model (Speech Recognition)
#
#Model:   LSTM
#         500 nodes each layer, trained with Adam
#
#Inputs:  Bavarian Speech Corpus (German)
#		  training utterances: 		56127
#		  validation utterances: 	 7012
#         test utterances: 		 	 7023
#         --------------------------------
#		  total					    70162
#
#		  shape of input: 39 MFCCcoeff vector
#
#Output:  135 classes (45 monophones with 3 HMM states each)
#Version: 4/2017 Roboball (MattK.)

#Start tensorboard via bash: 	tensorboard --logdir /logfile/directory
#Open Browser for tensorboard:  localhost:6006
#tensorboard --logdir /home/praktiku/korf/speechdata/tboard_logs/MLP_5layer_2017-05-10_14:04
#======================================================================='''


import numpy as np
import tensorflow as tf
import random
import re
import datetime
import matplotlib.pyplot as plt
import os
import subprocess
try:
	import cPickle as pickle
except:
   import _pickle as pickle

#remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

########################################################################
#define functions:
########################################################################

def loadNamefiles(filenames):
	'''load dataset-names from .txt files:'''
	return np.genfromtxt(filenames, delimiter=" ",dtype='str')

def loadPatData(filename, headerbytes):
	'''load Pattern files: 39MFCC coefficients, 12byte header'''
	with open(filename, 'rb') as fid:
		frames = np.fromfile(fid, dtype=np.int32) #get frames
		#print (frames[0])
		fid.seek(headerbytes, os.SEEK_SET)  # read in without header offset
		datafile = np.fromfile(fid, dtype=np.float32).reshape((frames[0], 39)).T
	return datafile

def createRandomvec(randomint, epochlength):
	'''create random vector'''
	np.random.seed(randomint)
	randomvec = np.arange(epochlength)
	np.random.shuffle(randomvec)
	return randomvec

def randomShuffleData(randomint, dataset_name):
	'''based on createRandomvec shuffle data randomly'''
	datasetlength = len(dataset_name)
	#init random list
	dataset_rand = []
	#create random vector
	randomvec = createRandomvec(randomint, datasetlength)
	#fill random list
	for pos in range(datasetlength):
		dataset_rand.append(dataset_name[randomvec[pos]])
	return dataset_rand

def padUtterance(framenum, labelfile, datafile):
	'''pad utterance on both sides'''
	#extract start and end vectors
	startveclabel = labelfile[0]
	startvecdata = datafile[:,0]
	endveclabel = labelfile[-1]
	endvecdata = datafile[:,-1]

	#compose pads
	startmatlabel = np.tile(startveclabel,((framenum-1)/2))
	startmatdata = np.tile(startvecdata,(((framenum-1)/2),1)).T
	endmatlabel = np.tile(endveclabel,((framenum-1)/2))
	endmatdata = np.tile(endvecdata,(((framenum-1)/2),1)).T

	#pad utterance
	paddedlabel = np.concatenate((startmatlabel, labelfile, endmatlabel), axis=0)
	paddeddata = np.concatenate((startmatdata,datafile, endmatdata), axis=1)
	return paddedlabel, paddeddata

def slidingWin(framenum, paddedlabel, paddeddata, modtype):
	'''create sliding windows to extract chunks'''
	#init: empty lists
	labellist=[]
	datalist=[]

	#sliding window over utterance
	for pos in range(paddedlabel.shape[0]-framenum+1):
		chunklabel = paddedlabel[pos:pos+framenum]
		label = chunklabel[(framenum-1)/2]
		chunkdata = paddeddata[:,pos:pos+framenum]
		if modtype == "MLP":
			chunkdata_vec = np.reshape(chunkdata, (39*framenum), order='F')
			labellist.append(label)
			datalist.append(chunkdata_vec)
		else:
			labellist.append(label)
			datalist.append(chunkdata)
	return labellist, datalist

def MinibatchData(minibatchsize, framenum, databuffer, modtype):
	'''create minibatches from databuffer'''
	#init data with zeros
	if modtype == "MLP":
		minibatchdata = np.zeros(shape=(minibatchsize,framenum * 39))
		#save vectors to minibatch array
		for vectorpos in range(minibatchsize):
			minibatchdata[vectorpos][:] = datavectorbuffer[vectorpos]

	elif modtype == "CNN":
		minibatchdata = np.zeros(shape=(minibatchsize,framenum, 39, 1 ))
		#save vectors to minibatch array
		for vectorpos in range(minibatchsize):
			datatensor = databuffer[vectorpos][np.newaxis,..., np.newaxis]
			minibatchdata[vectorpos][:][:][:] = datatensor
	else:
		print("Error, enter correct modtype: MLP, CNN, RNN")
	return minibatchdata

def MinibatchLabel(minibatchsize , classnum, labelbuffer):
	'''create one hot encoding from labels'''
	#init labels with zeros
	minibatchlabel = np.zeros(shape=(minibatchsize,classnum))

	#one-hot-encoding
	for labelpos in range(minibatchsize):
		label = int(labelbuffer[labelpos])
		minibatchlabel[labelpos][label-1] = 1.0
	return minibatchlabel

def Preprocess(framenum, randbatnum, dataset_name,datapath, labelpath, modtype):
	'''load data and label files, pad utterance, create chunks'''
	#init params, paths:
	headerbytes = 12

	#load data file:
	datafilename = datapath + dataset_name[randbatnum]
	datafile = loadPatData(datafilename, headerbytes)

	#load label file:
	dataset_namelab = dataset_name[randbatnum][:-4]+".txt"
	inputlabel = labelpath + dataset_namelab
	labelfile = np.loadtxt(inputlabel) #read in as float

	#pad utterance:
	paddedlabel, paddeddata = padUtterance(framenum, labelfile, datafile)

	#extract vector and labels by sliding window:
	labellist, datavectorlist = slidingWin(framenum, paddedlabel, paddeddata, modtype)

	return labellist, datavectorlist

def Databuffer(buffersamples, framenum, dataset_name, data_path, label_path, modtype):
	'''create buffer for minibatch'''
	#init buffer
	labelbuffer = []
	datavectorbuffer = []

	for pos in range(buffersamples):
		labellist, datavectorlist = Preprocess(framenum, pos, dataset_name, data_path, label_path, modtype)
		labelbuffer = labelbuffer+labellist
		datavectorbuffer = datavectorbuffer+datavectorlist
	return labelbuffer, datavectorbuffer


########################################################################
#import data .txt files: pickle
########################################################################

print('=============================================')
print("load training data")
print('=============================================\n')

#fix inintial randomness:
randomint = 1
modtype = "LSTM"
modlayer = 5

#get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

#init paths:
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
path = "/home/korf/Desktop/BA_05/" #on lynx5
dir_filenames = "data/dataset_filenames/"
dir_data = "data/pattern_hghnr_39coef/"
dir_label = "data/nn_output/"
data_path = path + dir_data
label_path = path + dir_label

tboard_logdir   = "runlogs/tboard_logs/"
tboard_path = path + tboard_logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

#init filenames:
trainset_filename = 'trainingset.txt'
validationset_filename = 'validationset.txt'
testset_filename = 'testset.txt'

#load filenames:
trainset_name = loadNamefiles(path + dir_filenames + trainset_filename)
valset_name = loadNamefiles(path + dir_filenames + validationset_filename)
testset_name = loadNamefiles(path + dir_filenames + testset_filename)

#random shuffle filenames:
valset_rand = randomShuffleData(randomint, valset_name)


########################################################################
#init parameter:
########################################################################

#define hyperparameter:
epochs = 1
learnrate=1e-4 #normal:1e-4
train_size  = len(trainset_name)
val_size = len(valset_name)
test_size = len(testset_name)

batsize_train = 100
batsize_test = 100
batches_train = int(train_size / batsize_train) #round down
batches_test = int(test_size / batsize_test) 	#round down
buffersamples = 10 		#choose number utterances in buffer

nodes = 500								#number of nodes per layer
classnum = 135 							#number of output classes
framenum = 39						 	#number of frames
display_step = 100 						#to show progress
randomint = 2 							#fix randomness
tolerance = 0.01

#init cost, accuracy:
crossval_history = np.empty(shape=[0],dtype=float)
cost_history = np.empty(shape=[0],dtype=float)
train_acc_history = np.empty(shape=[0],dtype=float)
test_acc_history = np.empty(shape=[0],dtype=float)


########################################################################
#init and define model: MLP model
########################################################################

#init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None, 39,39,1])
y_target = tf.placeholder(tf.float32, shape=[None, classnum])

#init model:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def flattening(x, shape):
	return tf.reshape(x, shape)

def matmul(x, W, b):
  return tf.add(tf.matmul(x, W), b)

def activfunction(x, acttype):
	if acttype == "tanh":
		activation_array = tf.tanh(x)
	elif acttype == "relu":
		activation_array = tf.nn.relu(x)
	else:
		activation_array = tf.sigmoid(x)
	return activation_array

#init activation function:
actdict = ["tanh", "relu", "sigmoid"]
acttype = actdict[1]

#init weights:

#1. hidden layer
with tf.name_scope(name="conv1"):
	W1 = weight_variable([4, 4, 1, 96])
	b1 = bias_variable([96])
	tf.summary.histogram("weights", W1)
	tf.summary.histogram("biases", b1)
#2. hidden layer
with tf.name_scope(name="conv2"):
	W2 = weight_variable([3, 3, 96, 192])
	b2 = bias_variable([192])
	tf.summary.histogram("weights", W2)
	tf.summary.histogram("biases", b2)
#3. hidden layer
with tf.name_scope(name="conv3"):
	W3 = weight_variable([3, 3, 192, 384])
	b3 = bias_variable([384])
	tf.summary.histogram("weights", W3)
	tf.summary.histogram("biases", b3)
#4. hidden fc-layer
with tf.name_scope(name="fc1"):
	W4 = weight_variable([384*3*3, 4096])
	b4 = bias_variable([4096])
	tf.summary.histogram("weights", W4)
	tf.summary.histogram("biases", b4)
#5. hidden fc-layer
with tf.name_scope(name="fc2"):
	W5 = weight_variable([4096, 4096])
	b5 = bias_variable([4096])
	tf.summary.histogram("weights", W5)
	tf.summary.histogram("biases", b5)
#6. output layer
with tf.name_scope(name="fc3"):
	W6 = weight_variable([4096, classnum])
	b6 = bias_variable([classnum])
	tf.summary.histogram("weights", W6)
	tf.summary.histogram("biases", b6)

#define model: 5 hidden layer model

#1.lstm layer
h_conv1 = activfunction((conv2d(x_input, W1) + b1),acttype)
h_pool1 = max_pool_2x2(h_conv1)
#2.lstm layer
h_conv2 = activfunction((conv2d(h_pool1, W2) + b2),acttype)
h_pool2 = max_pool_2x2(h_conv2)
#3.lstm layer
h_conv3 = activfunction((conv2d(h_pool2, W3) + b3),acttype)
h_pool3 = max_pool_2x2(h_conv3)
#4.fc layer
layer_flat = flattening(h_pool3, [-1,384*3*3])
layer_4 = activfunction(matmul(layer_flat, W4, b4),acttype)
#5.fc layer
layer_5 = activfunction(matmul(layer_4, W5, b5),acttype)
#6. output layer
layer_6 = matmul(layer_5, W6, b6)


#define classifier, cost function:
softmax = tf.nn.softmax(layer_6)

with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits= layer_6,labels = y_target))
	tf.summary.scalar("cross_entropy", cross_entropy)

#define optimizer, accuracy:
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)
with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(layer_6,1), tf.argmax(y_target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", accuracy)

#merge all summaries for tensorboard:
summ = tf.summary.merge_all()


#init tf session :
sess = tf.InteractiveSession()
#save and restore all the variables:
saver = tf.train.Saver()
#start session:
sess.run(tf.global_variables_initializer())
#init tensorboard
writer = tf.summary.FileWriter(tboard_path + tboard_name)
writer.add_graph(sess.graph)

#print out model info:
print("*******************************************")
print("model: " +str(modlayer)+" hidden layer "+str(modtype))
print("*******************************************")
print("5 layer, trained with Adam")
print("activation function: "+str(acttype))
print("-------------------------------------------")
print("data name: Bavarian Speech Corpus")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size)+"\n")


########################################################################
#train model:
########################################################################

print('=============================================')
print('start '+str(modtype)+' training')
print('=============================================')
t1_1 = datetime.datetime.now()

#epoch loop:
for epoch in range(1,epochs+1):

	#random shuffle filenames for each epoch:
	randomint1 = epoch
	trainset_rand = randomShuffleData(randomint1, trainset_name)

	#training loop: length = int(train_size/buffersamples)
	for buffernum in range(int(train_size/buffersamples)):

		#grab linear utterances (buffersamples) from random trainset:
		trainset_buffernum = trainset_rand[buffernum * buffersamples:(buffernum * buffersamples) + buffersamples]

		#create buffer:
		labelbuffer, datavectorbuffer = Databuffer(buffersamples, framenum,
		trainset_buffernum, data_path, label_path, modtype)
		bufferlength = len(labelbuffer)

		#get number of minibatches from buffer:
		minibatches = int(bufferlength/batsize_train)

		#minibatch loop per buffer:
		for batch in range(minibatches):

			#grab linear minibatch datavectors from databuffer:
			datavector = datavectorbuffer[batch * batsize_train: (batch * batsize_train)+batsize_train]
			#grab linear minibatch labels from labelbuffer:
			labelvector = labelbuffer[batch * batsize_train: (batch * batsize_train)+batsize_train]

			#minibatch data:
			trainbatchdata = MinibatchData(batsize_train, framenum, datavector, modtype)
			#minibatch labels: one hot encoding
			trainbatchlabel = MinibatchLabel(batsize_train, classnum, labelvector)

			'''feed minibatch into model:'''

			#start feeding data into the model:
			feedtrain = {x_input: trainbatchdata, y_target: trainbatchlabel}
			optimizer.run(feed_dict = feedtrain)

			#log history for tensorboard
			if batch % 5 == 0:
				[train_accuracy, s] = sess.run([accuracy, summ], feed_dict=feedtrain)
				writer.add_summary(s, batch)

			#get cost_history, accuracy history data:
			cost_history = np.append(cost_history,sess.run(cross_entropy,feed_dict=feedtrain))
			train_acc_history = np.append(train_acc_history,sess.run(accuracy,feed_dict=feedtrain))

		#check progress: training accuracy
		if  buffernum%10 == 0:
			train_accuracy = accuracy.eval(feed_dict=feedtrain)
			crossvalidationloss = sess.run(cross_entropy,feed_dict=feedtrain)
			crossval_history = np.append(crossval_history,crossvalidationloss)
			t1_2 = datetime.datetime.now()
			print('epoch: '+ str(epoch)+'/'+str(epochs)+
			' -- training utterance: '+ str(buffernum * buffersamples)+'/'+str(train_size)+
			" -- cross validation loss: " + str(crossvalidationloss))
			print('training accuracy: %.2f'% train_accuracy +
			" -- training time: " + str(t1_2-t1_1)[0:-7])

		#~ #stopping condition:
		#~ if abs(crossval_history[-1] - crossval_history[-2]) < tolerance:
			#~ break
	print('=============================================')

print('=============================================')
print("overall training accuracy %.3f"%np.mean(train_acc_history, axis=0))

#Measure Training Time:-------------------------------------------------
t1_3 = datetime.datetime.now()
print("total training time: " + str(t1_3-t1_1)[0:-7]+'\n')


########################################################################
#start testing:
########################################################################

print('=============================================')
print('start '+str(modtype)+' testing')
print('=============================================')
t2_1 = datetime.datetime.now()

#random shuffle filenames:
randomint2 = 1
testset_rand = randomShuffleData(randomint2, testset_name)

#test loop: length = int(test_size/buffersamples)
for buffernum2 in range(int(test_size/buffersamples)):

		#grab linear minibatch datavectors from databuffer:
		testset_buffernum = testset_rand[buffernum2 * buffersamples: (buffernum2 * buffersamples)+buffersamples]

		#create buffer:
		labelbuffer, datavectorbuffer = Databuffer(buffersamples, framenum,
		testset_buffernum, data_path, label_path, modtype)
		bufferlength = len(labelbuffer)

		#get number of minibatches from buffer:
		minibatches = int(bufferlength/batsize_test)


		#minibatch loop per buffer:
		for batch in range(minibatches):

			#grab minibatch datavectors from databuffer:
			datavector = datavectorbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]
			#grab minibatch labels from labelbuffer:
			labelvector = labelbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]

			#minibatch data:
			testbatchdata = MinibatchData(batsize_test, framenum, datavector, modtype)
			#minibatch labels: one hot encoding
			testbatchlabel = MinibatchLabel(batsize_test, classnum, labelvector)

			'''feed minibatch into model:'''

			#start feeding data into the model:
			feedtest = {x_input: testbatchdata, y_target: testbatchlabel}
			predictions = accuracy.eval(feed_dict = feedtest)
			test_acc_history = np.append(test_acc_history,predictions)

		#check progress: test accuracy
		if  buffernum2%10 == 0:
			test_accuracy = accuracy.eval(feed_dict=feedtest)
			t2_2 = datetime.datetime.now()
			print('test utterance: '+ str(buffernum2*buffersamples)+'/'+str(test_size))
			print('test accuracy: %.2f'% test_accuracy +
			" -- test time: " + str(t2_2-t2_1)[0:-7])

print('=============================================')
print('test utterance: '+ str(test_size)+'/'+str(test_size))
total_testacc = np.mean(test_acc_history, axis=0)
print("overall test accuracy %.3f"%total_testacc)

#Measure Training Time:-------------------------------------------------
t2_3 = datetime.datetime.now()
print("total test time: " + str(t2_3-t2_1)[0:-7]+'\n')

#info to start tensorboard:
print('=============================================')
print("tensorboard")
print('=============================================')
print("to start tensorboard via bash enter:")
print("tensorboard --logdir " + tboard_path + tboard_name)
print("open your Browser with:\nlocalhost:6006")


########################################################################
#start export:
########################################################################

print('=============================================')
print("start export")
print('=============================================')

#define storage path for model:
model_path = path + "SpeechCNN/pretrained_CNN/"
model_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker) + "_" + str(total_testacc)[0:-9]
print("Model saved to: ")
print(model_path + model_name)

#export bash log file:
#~ #file_ = open("output"+".txt", "w")
#~ #process = subprocess.Popen(['ls','-l'], stdout=subprocess.PIPE)
#~ #output = proc.stdout.read()
#subprocess.Popen("ls", stdout=file_, shell=True)
#ipAddress = '192.168.1.92'
#subprocess.Popen(["host", ipAddress], stdout=file_, shell=True)

#tensorflow method:-----------------------------------------------------
#save_path = saver.save(sess, model_path + model_name+".ckpt" )
#print("Model saved to file: %s" % save_path)

#pickle method:---------------------------------------------------------
W1 = np.array(sess.run(W1))
W2 = np.array(sess.run(W2))
W3 = np.array(sess.run(W3))
W4 = np.array(sess.run(W4))
W5 = np.array(sess.run(W5))
W6 = np.array(sess.run(W6))

b1 = np.array(sess.run(b1))
b2 = np.array(sess.run(b2))
b3 = np.array(sess.run(b3))
b4 = np.array(sess.run(b4))
b5 = np.array(sess.run(b5))
b6 = np.array(sess.run(b6))

#create export lists:
modelweights = [W1, W2, W3, W4, W5, W6, b1, b2, b3, b4,b5, b6 ];
modellayout = [ classnum, framenum, acttype];
trainingparams = [randomint, epochs, learnrate,batsize_train, batsize_test]
history = [cost_history, train_acc_history, test_acc_history]

#export trained model:
f = open(model_path + model_name, "wb")
pickle.dump(modelweights, f,protocol=2)
pickle.dump(modellayout, f,protocol=2)
pickle.dump(trainingparams, f,protocol=2)
pickle.dump(history, f,protocol=2)
f.close()

print('=============================================')
print("export finished")
print('=============================================')
print(" "+"\n")

#print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: " +str(modlayer)+" hidden layer "+str(modtype))
print("*******************************************")
print("3conv + 2fc layer, trained with Adam")
print("activation function: "+str(acttype))
print("-------------------------------------------")
print("data name: Bavarian Speech Corpus")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(t1_3-t1_1)[0:-7])
print("overall training accuracy %.3f"%np.mean(train_acc_history, axis=0))
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(t2_3-t2_1)[0:-7])
print("overall test accuracy %.3f"%total_testacc)
print("*******************************************")


########################################################################
#plot settings:
########################################################################

#=======================================================================
#plot training
#=======================================================================

#plot training loss function
plt.figure(1, figsize=(8,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='b')
plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batsize_train))
plt.ylabel('loss')

#plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_acc_history)),train_acc_history, color='g')
plt.axis([0,len(cost_history),0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches, batch size:'+ str(batsize_train))
plt.ylabel('accuracy percentage')


#=======================================================================
#plot testing
#=======================================================================

#plot validation loss function
plt.figure(2, figsize=(8,8))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(crossval_history)),crossval_history, color='b')
plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

#plot test accuracy function
plt.subplot(212)
plt.plot(range(len(test_acc_history)),test_acc_history, color='m')
plt.axis([0,len(test_acc_history),0,1])
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batsize_test))
plt.ylabel('accuracy percentage')



#plt.ion()
plt.show()
#plt.hold(False)
