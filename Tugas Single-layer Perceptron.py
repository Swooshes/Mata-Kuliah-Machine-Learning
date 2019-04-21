import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

index = ['x1','x2','x3','x4','types']
df = pd.read_csv('iris.data',names=index)

df.head(100)

iris = df.head(100).values.tolist()

for i in iris:
	if(i[4]=='Iris-setosa'):
		i.append(0)
	else:
		i.append(1)

print(*iris[:][:5], sep="\n")
print(*iris[:][50:55], sep="\n")

bin1 = iris[0:10]+iris[50:60]
bin2 = iris[10:20]+iris[60:70]
bin3 = iris[20:30]+iris[70:80]
bin4 = iris[30:40]+iris[80:90]
bin5 = iris[40:50]+iris[90:100]

#set validation
valid1 = bin1[:]
valid2 = bin2[:]
valid3 = bin3[:]
valid4 = bin4[:]
valid5 = bin5[:]
datavalid = [valid1,valid2,valid3,valid4,valid5]

#set data training
datatr1 = bin2[:]+bin3[:]+bin4[:]+bin5[:]
datatr2 = bin1[:]+bin3[:]+bin4[:]+bin5[:]
datatr3 = bin1[:]+bin2[:]+bin4[:]+bin5[:]
datatr4 = bin1[:]+bin2[:]+bin3[:]+bin5[:]
datatr5 = bin1[:]+bin2[:]+bin3[:]+bin4[:]
datatrain = [datatr1,datatr2,datatr3,datatr4,datatr5]

#assign theta and bias
theta = [0.4,0.4,0.4,0.4]
bias = 0.4
thetalist = [theta[:] for i in range(5)]
biaslist = [bias for i in range(5)]
dt = [0.4,0.4,0.4,0.4]
db = 0

#result function
def result(x,j):
  res = 0.0
  for i in range(4):
    global thetalist
    res += (x[i]*thetalist[j][i])
  global bias
  return res + bias

restest = result(datatr2[0],0)
restest

def actv(res):
  return 1/(1+math.exp(-res))

actest = actv(result(datatr2[0],0))
actest

def predict(act_v):
  if(act_v>0.5):
    return 1
  else:
    return 0

predtest = predict(actest)
predtest

def error(y,act):
  return math.pow((y-act),2)

errtest = error(datatr2[0][5],actest)
errtest

def dtUpdate(x,y,act):
  global dt
  for i in range(4):
    dt[i] = 2*(act-y)*(1-act)*act*x[i]

dtUpdate(datatr2[0],datatr2[0][5],actv(restest))
print(*dt, sep = ",  ")

def dbUpdate(y,act):
  global db
  db = 2*(act-y)*(1-act)*act

dbUpdate(datatr2[0][5],actv(restest))
db

def tUpdate(lrate,j):
  global thetalist
  for i in range(4):
    thetalist[j][i] = thetalist[j][i]-(lrate*dt[i])

def bUpdate(lrate,j):
  global biaslist
  biaslist[j] = biaslist[j]-(lrate*db)

acc_datatr_final = []
acc_valid_final = []

err_datatr_final = []
err_valid_final = []

for i in range(300):
  sum_err_datatr = 0
  sum_err_valid = 0 
  sum_acc_datatr = 0
  sum_acc_valid = 0 
  
  for j in range(5):
    sun = 0
    sun2 = 0
    tptn = 0
    tptn2 = 0
    
    for k in range(80):
      actr = actv(result(datatrain[j][k],j))
      predr = predict(actr)
      if(predr==datatrain[j][k][5]):
        tptn = tptn+1
      sun = sun + error(datatrain[j][k][5],actr)
      dtUpdate(datatrain[j][k][0:4],datatrain[j][k][5],actr)
      dbUpdate(datatrain[j][k][5],actr)
      tUpdate(0.1,j)
      bUpdate(0.1,j)
    
    sum_err_datatr = sum_err_datatr+sun/80 
    sum_acc_datatr = sum_acc_datatr+(tptn/80)*100
    
    for k in range(20):
      actval = actv(result(datavalid[j][k],j))
      predv = predict(actval)
      if(predv==datavalid[j][k][5]):
        tptn2 = tptn2 + 1
      sun2 = sun2+error(datavalid[j][k][5],actval)
    
    sum_err_valid = sum_err_valid+(sun2/20)
    sum_acc_valid = sum_acc_valid+(tptn2/20)*100
  
  err_datatr_final.append(sum_err_datatr/5)
  err_valid_final.append(sum_err_valid/5)
  acc_datatr_final.append(sum_acc_datatr/5)
  acc_valid_final.append(sum_acc_valid/5)

plt.figure(1, figsize=(6,6))
plt.plot(acc_datatr_final,'r-', label='training', linewidth=2)
plt.plot(acc_valid_final,'b-', label='validasi', linewidth=2)
plt.xlabel('epoch', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.legend(loc='lower right', fontsize=18)

plt.figure(2, figsize=(6,6))
plt.plot(err_datatr_final,'r-', label='training', linewidth=2)
plt.plot(err_valid_final,'b-', label='validasi', linewidth=2)
plt.xlabel('epoch', fontsize=16)
plt.ylabel('error', fontsize=16)
plt.legend(loc='upper right', fontsize=18)

plt.show()
