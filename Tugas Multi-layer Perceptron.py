import pandas as pd
import math
from random import seed
from random import random
import matplotlib.pyplot as plt

index = ['x1','x2','x3','x4','types']
df = pd.read_csv('iris.data',names=index)

iris = df.head(150).values.tolist()

seed(9)
n_inputs = 4
n_outputs = 3
n_hidden = 6
n_epoch = 250
learning_rate = 0.8
scheme = list()

for i in iris:
    if(i[4]=='Iris-setosa'):
        i[4]=0
    elif(i[4]=='Iris-versicolor'):
        i[4]=1
    else:
        i[4]=2

valid_data = iris[:10]+iris[50:60]+iris[100:110]
train_data = iris[10:50]+iris[60:100]+iris[110:150]

hidden_layer = [{'theta':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
scheme.append(hidden_layer)
output_layer = [{'theta':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
scheme.append(output_layer)

def forward(scheme, row):
    current = row
    for layer in scheme:
        new = []
        for node in layer:
            node['output'] = activate(node['theta'],current)
            new.append(node['output'])
        current = new
    return current

def backward(scheme, target):
    for i in range(len(scheme)-1,-1,-1):
        layer = scheme[i]
        errors = list()
        if i != len(scheme)-1:
            for j in range(len(layer)):
                error = 0.0
                for node in scheme[i + 1]:
                    error += (node['theta'][j] * node['d'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                node = layer[j]
                errors.append(target[j] - node['output'])
        for j in range(len(layer)):
            node = layer[j]
            node['d'] = errors[j] * (node['output'] * (1.0 - node['output']))

def activate(theta, inputs):
    res = theta[-1]
    for i in range(len(theta)-1):
        res += theta[i] * inputs[i]
    return 1.0 / (1.0 + math.exp(-res))

def update_theta(scheme, row, learning_rate):
    for i in range(len(scheme)):
        inputs = row[:-1]		#row[:-1] = bias
        if i != 0:
            inputs = [node['output'] for node in scheme[i - 1]]
        for node in scheme[i]:
            for j in range(len(inputs)):
                node['theta'][j] += learning_rate * node['d'] * inputs[j]
            node['theta'][-1] += learning_rate * node['d']

def train(scheme, training, learning_rate, n_outputs):
    returned = []
    error_total = 0
    correct_total = 0
    for row in training:
        outputs = forward(scheme, row)
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
        backward(scheme, target)
        update_theta(scheme, row, learning_rate)
    returned.append(error_total/len(training))
    returned.append(correct_total/len(training))
    return returned

def validate(scheme, training, n_outputs):
    returned = []
    error_total = 0
    correct_total = 0
    for row in training:
        outputs = forward(scheme, row)
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
        backward(scheme, target)
    returned.append(error_total/len(training))
    returned.append(correct_total/len(training))
    return returned

list_avg_avg_error=[]
list_avg_accuracy=[]
list_avg_avg_error_v=[]
list_avg_accuracy_v=[]

log_avg_avg_error=[]
log_avg_accuracy=[]
log_avg_avg_error_v=[]
log_avg_accuracy_v=[]

for epoch in range(n_epoch):
    tmp_error = 0.0
    tmp_accuracy = 0.0
    tmp_error_v = 0.0
    tmp_accuracy_v = 0.0
    
    t = train(scheme, train_data, learning_rate, n_outputs)
    tmp_error+=t[0]
    tmp_accuracy+=t[1]
    
    v = validate(scheme,valid_data,n_outputs)
    tmp_error_v = v[0]
    tmp_accuracy_v = v[1]

    list_avg_avg_error.append(tmp_error)
    list_avg_accuracy.append(tmp_accuracy)
    list_avg_avg_error_v.append(tmp_error_v)
    list_avg_accuracy_v.append(tmp_accuracy_v)
    
    log_avg_avg_error.append(math.log(tmp_error))
    if tmp_accuracy!=0:
        log_avg_accuracy.append(math.log(tmp_accuracy))
    log_avg_avg_error_v.append(math.log(tmp_error_v))
    if tmp_accuracy_v!=0:
        log_avg_accuracy_v.append(math.log(tmp_accuracy_v))

plt.figure(1, figsize=(7,7))
plt.plot(list_avg_avg_error,'-r',label='Training')
plt.plot(list_avg_avg_error_v,'-b',label='Validation')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.legend(loc='upper right', fontsize=16)

plt.figure(2, figsize=(7,7))
plt.plot(log_avg_avg_error,'-r',label='Training')
plt.plot(log_avg_avg_error_v,'-b',label='Validation')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Log Error', fontsize=16)
plt.legend(loc='upper right', fontsize=16)

plt.figure(3, figsize=(7,7))
plt.plot(list_avg_accuracy,'-r',label='Training')
plt.plot(list_avg_accuracy_v,'-b',label='Validation')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(loc='lower right', fontsize=16)

plt.figure(4, figsize=(7,7))
plt.plot(log_avg_accuracy,'-r',label='Training')
plt.plot(log_avg_accuracy_v,'-b',label='Validation')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Log Accuracy', fontsize=16)
plt.legend(loc='lower right', fontsize=16)
plt.show()
