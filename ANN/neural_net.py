'''
My first attempt to build a neural network...."
-Aditya Soni

'''
#Import Statements

import numpy as np  # for fast calculations
import matplotlib.pyplot as plt # for plotiing
import scipy.special # for sigmoid function
from sklearn.metrics import confusion_matrix

k = list()
k_ =list()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

class NeuralNetworks:

	# initialising nn
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate ):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		#weights
		self.wih = np.random.normal(0.0 , pow(self.hnodes , -0.5),(self.hnodes, self.inodes))
		self.who = np.random.normal(0.0 , pow(self.onodes , -0.5),(self.onodes , self.hnodes))
		self.activation_function = lambda x: scipy.special.expit(x)
		pass


#train the ANN
#the subtle part....
# it is quite similar to the query function
	def train(self, input_list, target_list):
		
		#converting input to 2d array
		inputs = np.array(input_list , ndmin = 2).T 
		targets = np.array(target_list , ndmin =2).T

		#calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih , inputs)

		#calculating o/p from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculating signals into final layer
		final_inputs = np.dot(self.who , hidden_outputs)

		# calculating final o/s value
		final_outputs = self.activation_function(final_inputs)

		#error is target - actual value
		output_errors = targets - final_outputs

		#applying backpropagation logic now (state of art of ANN in ML)
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)

		#updating the weights for the link between hidden and output layers
		# the formula we apply is eta*y(1-y)*o/p
		self.who += self.lr*np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
		
		self.wih += self.lr*np.dot((hidden_errors * hidden_outputs *(1 - hidden_outputs)), np.transpose(inputs))

		pass


	def query(self, input_list):
		
		#converting input to 2d array
		inputs = np.array(input_list , ndmin = 2).T 

		#calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih , inputs)
		
		#calculating o/p from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculating signals into final layer
		final_inputs = np.dot(self.who , hidden_outputs)

		# calculating final o/s value
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

input_nodes = 784 		#28*28
hidden_nodes = 50
output_nodes = 10
learning_rate = 0.1

#creating an instance of the class.....
n = NeuralNetworks(input_nodes , hidden_nodes , output_nodes , learning_rate)
