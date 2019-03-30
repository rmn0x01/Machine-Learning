import math
from random import seed
from random import random
import pandas as pd 
import matplotlib.pyplot as plt 

#Fungsi Aktivasi + Sigmoid
def activate(theta, inputs):
	activation = theta[-1]
	for i in range(len(theta)-1):
		activation += theta[i] * inputs[i]
	return 1.0 / (1.0 + math.exp(-activation))
 
 
 #Fungsi Feed Forward
def forward(skema, row):
	current = row
	for layer in skema:
		after = []
		for node in layer:
			node['output'] = activate(node['theta'],current)
			after.append(node['output'])
		current = after
	return current

 #Fungsi Backpropagation
def backward(skema, target):
	#Loop mundur dari layer output ke layer hidden
	for i in range(len(skema)-1,-1,-1):
		layer = skema[i]
		errors = list()
		#Hitung error untuk hidden layer
		if i != len(skema)-1:
			for j in range(len(layer)):
				error = 0.0
				for node in skema[i + 1]:
					error += (node['theta'][j] * node['d'])
				errors.append(error)
		#Hitung error untuk output layer
		else:
			for j in range(len(layer)):
				node = layer[j]
				errors.append(target[j] - node['output'])
		#Hitung derivatif (dtheta dan dbias)
		for j in range(len(layer)):
			node = layer[j]
			node['d'] = errors[j] * (node['output'] * (1.0 - node['output']))
 
#Fungsi Update Theta dan Bias
def update_weights(skema, row, learning_rate):
	for i in range(len(skema)):
		inputs = row[:-1]		#row[:-1] = bias
		if i != 0:
			inputs = [node['output'] for node in skema[i - 1]]
		for node in skema[i]:
			#Bagian update theta dan bias
			for j in range(len(inputs)):
				node['theta'][j] += learning_rate * node['d'] * inputs[j]
			node['theta'][-1] += learning_rate * node['d']
 
#Fungsi Training Data, mengembalikan rata-rata error dan akurasi
def process_train(skema, training, learning_rate, n_outputs):
	returned = []
	error_total = 0
	correct_total = 0
	for row in training:
		outputs = forward(skema, row)
		target = [0 for i in range(n_outputs)]
		target[int(row[-1])] = 1
		for i in range(len(target)):
			error_total += math.pow((target[i]-outputs[i]),2)*0.5
		for i in range(len(target)):
			if outputs[i] > 0.5:
				outputs[i] = 1
			else:
				outputs[i] = 0
		if (outputs == target):
			correct_total+=1
		backward(skema, target)
		update_weights(skema, row, learning_rate)
	returned.append(error_total/len(training))
	returned.append(correct_total/len(training))
	return returned

#Fungsi Validasi Data
def process_validate(skema, training, n_outputs):
	returned = []
	error_total = 0
	correct_total = 0
	for row in training:
		outputs = forward(skema, row)
		target = [0 for i in range(n_outputs)]
		target[int(row[-1])] = 1
		for i in range(len(target)):
			error_total += math.pow((target[i]-outputs[i]),2)*0.5
		for i in range(len(target)):
			if outputs[i] > 0.5:
				outputs[i] = 1
			else:
				outputs[i] = 0
		if (outputs == target):
			correct_total+=1
		backward(skema, target)
	returned.append(error_total/len(training))
	returned.append(correct_total/len(training))
	return returned

#Baca dokumen, penyesuaian format
df = pd.read_csv('iris.csv')
df = df.rename(index=str,columns={"sepal_length":"x1","sepal_width":"x2","petal_length":"x3","petal_width":"x4"})
df['y0']=0
df.loc[df['species'] == 'versicolor','y0'] = 1
df.loc[df['species'] == 'virginica','y0'] = 2
df = df.drop(columns=['species'])
#Split data, 30 validasi 120 training
validasi = (pd.concat([df.iloc[:10], df.iloc[50:60], df.iloc[100:110]])).head(30).values.tolist()
training = (pd.concat([df.iloc[10:50], df.iloc[60:100], df.iloc[110:150]])).head(120).values.tolist()
#Inisialisasi variabel awal
seed(2)
n_inputs = 4
n_outputs = 3
n_hidden = 6 
n_epoch = 300
learning_rate = 0.1
#Inisialisasi skema yang akan digunakan
skema = list()
hidden_layer = [{'theta':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
skema.append(hidden_layer)
output_layer = [{'theta':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
skema.append(output_layer)
#Menyimpan data error,akurasi dari training dan validasi untuk plot ke grafik
list_avg_avg_error=[]
list_avg_accuracy=[]
list_avg_avg_error_v=[]
list_avg_accuracy_v=[]

for epoch in range(n_epoch):
	tmp_error = 0.0         #menyimpan error training
	tmp_accuracy = 0.0
	tmp_error_v = 0.0       #v = validasi
	tmp_accuracy_v = 0.0
	print(epoch)
	t = process_train(skema, training, learning_rate, n_outputs)
	tmp_error+=t[0]
	tmp_accuracy+=t[1]
	v = process_validate(skema,validasi,n_outputs)
	tmp_error_v = v[0]
	tmp_accuracy_v = v[1]
	list_avg_avg_error.append(tmp_error)
	list_avg_accuracy.append(tmp_accuracy)
	list_avg_avg_error_v.append(tmp_error_v)
	list_avg_accuracy_v.append(tmp_accuracy_v)

def cetak_grafik_loss_function(list_avg_avg_error,list_avg_avg_error_v):
    plt.plot(list_avg_avg_error,label='Training')
    plt.plot(list_avg_avg_error_v,label='Validation')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def cetak_grafik_accuracy(list_avg_accuracy,list_avg_accuracy_v):
    plt.plot(list_avg_accuracy,label='Training')
    plt.plot(list_avg_accuracy_v,label='Validation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#uncomment salah satu untuk tampilkan grafik

cetak_grafik_loss_function(list_avg_avg_error,list_avg_avg_error_v)
#cetak_grafik_accuracy(list_avg_accuracy,list_avg_accuracy_v)
