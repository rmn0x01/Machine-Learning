from math import exp
from random import seed
from random import random
import pandas as pd
import math
'''
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs, weight_hidden, weight_output):
	network = list()
	hidden_layer = [{'weights':weight_hidden} for i in range(n_hidden)]
	#print(hidden_layer)
	network.append(hidden_layer)
	output_layer = [{'weights':weight_output} for i in range(n_outputs)]
	#print(output_layer)
	network.append(output_layer)
	return network
'''
 
# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
 
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])   #wtf????
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		#inputs = row[:-1]
		inputs = row[:-2]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']
 
# Train a network for a fixed number of epochs
def train(network, training, l_rate, n_epoch, n_outputs):
	kumpul_error = []
	kumpul_akurasi = []
	for epoch in range(n_epoch):
		sum_error = 0
		sum_correct = 0
		for row in training:
			outputs = forward_propagate(network, row)
			#print(outputs)
			#expected = [0.0 for i in range(n_outputs)]
			#expected[int(row[-1])] = 1
			expected = [0,0]
			expected[0] = row[-2]
			expected[1] = row[-1]
			sum_error += sum([0.5*(math.pow((expected[i]-outputs[i]),2)) for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

			for i in range(len(expected)):
				if outputs[i] > 0.5:
					outputs[i] = 1
				else:
					outputs[i] = 0 
			if(expected == outputs):
				sum_correct+=1
			print(outputs)
			print(expected)
		kumpul_error.append(sum_error/len(training))
		kumpul_akurasi.append(sum_correct/len(training))
		print(str(epoch)+'===')
		#print(expected)
		#print(outputs)
		#print(sum_error/len(training))
		#print(sum_correct/len(training))
 
# Test training backprop algorithm
df = pd.read_csv('iris.csv')
df = df.rename(index=str,columns={"sepal_length":"x1","sepal_width":"x2","petal_length":"x3","petal_width":"x4"})
df['y0']=0
df['y1']=0
#Drop kolom species untuk mempermudah proses berikutnya
df.loc[df['species'] == 'versicolor','y1'] = 1
df.loc[df['species'] == 'virginica','y0'] = 1
df = df.drop(columns=['species'])



validasi = (pd.concat([df.iloc[:10], df.iloc[50:60], df.iloc[100:110]])).head(30).values.tolist()
training = (pd.concat([df.iloc[10:50], df.iloc[60:100], df.iloc[110:150]])).head(120).values.tolist()
#print(training)
theta_bias_hidden = [0.6,0.6,0.6,0.6,0.5]
theta_bias_output = [0.6,0.6,0.5]

skema = list()
n_inputs = 4
n_outputs = 2
hidden_layer = [{'weights' : theta_bias_hidden} for i in range(2)] #inisialisasi weight/theta per hidden neuron (pada kasus ini = 2)
skema.append(hidden_layer)
output_layer = [{'weights' : theta_bias_output} for i in range(2)] #inisialisasi weight per output (2)
skema.append(output_layer)
learning_rate = 0.8
n_epoch = 10
#n_inputs = len(training[0]) - 1
#n_outputs = len(set([row[-1] for row in training]))
#network = initialize_network(n_inputs, 2, n_outputs,theta_bias_hidden, theta_bias_output )
train(skema, training, learning_rate, n_epoch, n_outputs)

#for layer in skema:
#	print(layer)
