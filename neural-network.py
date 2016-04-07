#Parthiv Mohan

import numpy as np
import random
import scipy.io as sio
import pickle

def preprocess(trainFeatures, trainLabels, testFeatures):
	trainFeatures = trainFeatures/256.0
	#trainFeatures = np.vstack(trainFeatures, np.ones((60000,1)))
	testFeatures = testFeatures/256.0
	#testFeatures = np.vstack(testFeatures, np.ones((60000,1)))
	newTrainLabels = []
	for label in trainLabels:
		newLabel = np.zeros(10)
		newLabel[label] = 1
		newTrainLabels.append(newLabel)
	# print "preprocess"
	return trainFeatures, np.array(newTrainLabels), testFeatures

def load(trainPath, testPath):
	train = sio.loadmat(trainPath) #training_data and training_labels
	trainFeatures = train["train_images"]
	trainFeatures = np.reshape(trainFeatures,(784,60000)).T
	trainLabels = np.ravel(train["train_labels"])
	testFeatures = np.reshape(sio.loadmat(testPath)["test_images"],(784,10000)).T #training_data and training_labels
	trainFeatures, trainLabels, testFeatures = preprocess(trainFeatures, trainLabels, testFeatures)
	return trainFeatures,trainLabels,testFeatures

def splitValidationTrain(trainFeatures,trainLabels):
	train = zip(trainFeatures,trainLabels)
	random.shuffle(train)
	val = train[:(len(train)*1)/5]
	valFeatures = [i for i,j in val]
	valLabels = [j for i,j in val]
	train = train[(len(train)*1)/5:]
	trainFeatures = [i for i,j in train]
	trainLabels = [j for i,j in train]
	# print "splitValidationTrain"
	return np.array(valFeatures),np.array(valLabels),np.array(trainFeatures),np.array(trainLabels)

class Neural_Network(object):
	def __init__(self, N_IN = 784, N_OUT = 10, N_HID = 200, EPSILON = 0.5):
		#haven't added bias
		self.inputLayerSize = N_IN
		self.hiddenLayerSize = N_HID
		self.outputLayerSize = N_OUT

		#Weights (Parameters)
		#self.W1 = np.random.randn(N_IN + 1, N_HID + 1)*EPSILON
		self.W1 = np.random.randn(N_IN, N_HID)*EPSILON
		#self.W2 = np.random.randn(N_HID + 1, N_OUT)*EPSILON
		self.W2 = np.random.randn(N_HID, N_OUT)*EPSILON
		# print "W2", self.W2
		# print "__init__"

	# def trainNeuralNetwork(trainFeatures, trainLabels):
	# 	W1 = np.random.random((N_IN, N_HID))*EPSILON
	# 	W2 = np.random.random((N_HID, N_OUT))*EPSILON


	# def predictNeuralNetwork(weights, images):
	# 	return

	def forward(self, x):
		self.s2 = np.dot(x, self.W1)
		self.x2 = self.sigmoid(self.s2)
		self.s3 = np.dot(self.x2, self.W2)
		yHat = self.sigmoid(self.s3)
		# print "forward"
		return yHat

	def sigmoid(self, z):
		# print "sigmoid"
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		# print "sigmoidPrime"
		return np.exp(-z)/((1+np.exp(-z))**2)

	def tanhPrime(self, z):
		return 1 - np.tan(z)**2

	def costFunction(self, x, y):
		self.yHat = self.forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		# print "costFunction"
		return J

	def costFunctionPrime(self, X, y):
		i = random.randint(0, 47999)
		x_i = X[i]
		y_i = y[i]
		y_iHat = self.forward(x_i)
		
		#MSE
		# print "-(y_i-y_iHat)", (-(y_i-y_iHat)).shape
		# print "self.sigmoidPrime(self.s3)", self.sigmoidPrime(self.s3).shape
		delta3 = np.multiply(-(y_i-y_iHat),self.sigmoidPrime(self.s3))
		# print "delta3", delta3.shape
		dJdW2 = np.outer(self.x2, delta3)
		# print "dJdW2", dJdW2.shape

		#cross entropy
		# dJdW2 = np.outer(self.sigmoid(), self.x2)

		# print "np.dot(delta3, self.W2.T)", np.dot(delta3, self.W2.T).shape
		# print "self.sigmoidPrime(self.s2)", self.sigmoidPrime(self.s2).shape
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.s2)
		# print "delta2", delta2.shape
		dJdW1 = np.outer(x_i, delta2)
		# print "dJdW1", dJdW1.shape

		# print "dJdW1.shape", dJdW1.shape
		# print "dJdW2.shape", dJdW2.shape
		return dJdW1, dJdW2

	def update(self, trainFeatures, trainLabels, ETA):
		dJdW1,dJdW2 = self.costFunctionPrime(trainFeatures, trainLabels)
		self.W1 = self.W1 - ETA*dJdW1
		self.W2 = self.W2 - ETA*dJdW2
		# print "update"

def validate():
	trainFeatures,trainLabels,testFeatures = load("digit-dataset/train.mat", "digit-dataset/test.mat")
	valFeatures,valLabels,trainFeatures,trainLabels = splitValidationTrain(trainFeatures,trainLabels)
	# print "train"
	train = Neural_Network()
	# print "validate"
	validate = Neural_Network()
	for j in range(1000000):
		if j % 1000 == 0:
			validate.W1 = train.W1
			validate.W2 = train.W2
			predValLabels = validate.forward(valFeatures)
			# print "valFeatures", valFeatures
			# print "predValLabels", predValLabels
			# print "predValLabels.shape", predValLabels.shape
			labels = np.argmax(predValLabels, axis = 1)
			# print "indices", indices
			# print "indices.shape", indices.shape
			predValLabels = []
			for label in labels:
				newLabel = np.zeros(10)
				newLabel[label] = 1
				predValLabels.append(newLabel)
			predValLabels = np.array(predValLabels)
			# predValLabels = np.zeros(12000,10)[indices] = 1
			# print "predValLabels", predValLabels
			# print "valLabels", valLabels
			correct = sum([1 for i in range(len(valLabels)) if (valLabels[i]==predValLabels[i]).all()])/float(len(valLabels))	
			print "i", j, "correct", correct
		train.update(trainFeatures, trainLabels, 0.05)
		# print train.W1
		# print train.W2

	# with open('nn.pkl', 'wb') as output:
	#     pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)

	#     company2 = Company('spam', 42)
	#     pickle.dump(company2, output, pickle.HIGHEST_PROTOCOL)

def trainvalidate():
	trainFeatures,trainLabels,testFeatures = load("digit-dataset/train.mat", "digit-dataset/test.mat")
	valFeatures,valLabels,trainFeatures,trainLabels = splitValidationTrain(trainFeatures,trainLabels)
	# print "train"
	train = Neural_Network()
	# print "validate"
	validate = Neural_Network()
	for j in range(1000000):
		if j % 10000 == 0:
			validate.W1 = train.W1
			validate.W2 = train.W2
			predValLabels = validate.forward(valFeatures)
			# print "valFeatures", valFeatures
			# print "predValLabels", predValLabels
			# print "predValLabels.shape", predValLabels.shape
			labels = np.argmax(predValLabels, axis = 1)
			# print "indices", indices
			# print "indices.shape", indices.shape
			predValLabels = []
			for label in labels:
				newLabel = np.zeros(10)
				newLabel[label] = 1
				predValLabels.append(newLabel)
			predValLabels = np.array(predValLabels)
			# predValLabels = np.zeros(12000,10)[indices] = 1
			# print "predValLabels", predValLabels
			# print "valLabels", valLabels
			correct = sum([1 for i in range(len(valLabels)) if (valLabels[i]==predValLabels[i]).all()])/float(len(valLabels))	
			print "validate", j, correct
			predTrainLabels = train.forward(trainFeatures)
			labels = np.argmax(predTrainLabels, axis = 1)
			predTrainLabels = []
			for label in labels:
				newLabel = np.zeros(10)
				newLabel[label] = 1
				predTrainLabels.append(newLabel)
			predTrainLabels = np.array(predTrainLabels)
			# predValLabels = np.zeros(12000,10)[indices] = 1
			# print "predValLabels", predValLabels
			# print "valLabels", valLabels
			correct = sum([1 for i in range(len(trainLabels)) if (trainLabels[i]==predTrainLabels[i]).all()])/float(len(trainLabels))	
			print "train", j, correct
		train.update(trainFeatures, trainLabels, 0.05)
		# print train.W1
		# print train.W2

	# with open('nn.pkl', 'wb') as output:
	#     pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)

	#     company2 = Company('spam', 42)
	#     pickle.dump(company2, output, pickle.HIGHEST_PROTOCOL)

def test():
	trainFeatures,trainLabels,testFeatures = load("digit-dataset/train.mat", "digit-dataset/test.mat")
	valFeatures,valLabels,trainFeatures,trainLabels = splitValidationTrain(trainFeatures,trainLabels)
	# print "train"
	train = Neural_Network()
	# print "validate"
	test = Neural_Network()
	for j in range(600000):
		train.update(trainFeatures, trainLabels, 0.05)
	test.W1 = train.W1
	test.W2 = train.W2
	predTestLabels = test.forward(testFeatures)
	labels = np.argmax(predTestLabels, axis = 1)
	testLabels = open("test_labels.csv", "w")
	testLabels.write("Id,Category\n")
	for i,label in enumerate(labels):
		testLabels.write(str(i+1) + "," + str(label) + "\n")
	testLabels.close()

test()