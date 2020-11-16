#!usr/bin/env python
# -*- coding: utf-8 -*-

import scipy, math, MLUtility

from scikits.learn import svm

# Training function for learning the multiple one class SVM model
def train(data,label):
  mocsvm_model = []
  maxdata = scipy.zeros((len(scipy.unique(label)),data.shape[1]))
  mindata = scipy.zeros((len(scipy.unique(label)),data.shape[1]))
  cnt = 0
  for i in scipy.unique(label):
    # for every class learn a one class svm model
    # Learn an approximate value for gamma
    tempdata = data[(label==i),:]
    tempdata, maxdata[cnt,:], mindata[cnt,:] = MLUtility.normalizedata(tempdata.copy())
    dist = scipy.zeros((tempdata.shape[0],tempdata.shape[0]))
    for j in range(tempdata.shape[0]):
        for k in range(tempdata.shape[0]):
            dist[j,k] = sum((tempdata[j,:]-tempdata[k,:])**2)
    g = dist.max().max()
    model = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=g*8,C=10)
    model.fit(tempdata)
    # append it to the list of models
    mocsvm_model.append(model)
    cnt = cnt + 1
  return mocsvm_model,maxdata,mindata

# Determining the output of the classifier
def test(data,mocsvm_model,maxdata,mindata):
  probas = scipy.zeros((data.shape[0],len(mocsvm_model)))
  for i in range(len(mocsvm_model)):
    tempdata,tm,tn = MLUtility.normalizedata(data.copy(),maxdata[i,:],mindata[i,:])
    probas[:,i] = MLUtility.computeprobability(mocsvm_model[i].predict_margin(tempdata)).transpose()
    # stretch the probability values to range in [0,1]
    maxp = probas[:,i].max(0)
    minp = probas[:,i].min(0)
    probas[:,i] = (probas[:,i]-minp)/(maxp-minp)
  return probas

# Determining the output of the classifier
def testmin(data,mocsvm_model):
  probas = scipy.zeros((data.shape[0],len(mocsvm_model)))
  op = scipy.zeros((data.shape[0]))-1
  for i in range(len(mocsvm_model)):
    probas[:,i] = mocsvm_model[i].predict(data)
  for i in range(data.shape[0]):
    for j in range(probas.shape[1]):
        if probas[i,j] == 1:
            op[i]=1
  return op

# Determining the output of the classifier
def testdst(data,mocsvm_model,maxdata,mindata):
  probas = scipy.zeros((data.shape[0],len(mocsvm_model)))
  op = scipy.zeros((data.shape[0]))-1
  for i in range(len(mocsvm_model)):
    tempdata,tm,tn = MLUtility.normalizedata(data.copy(),maxdata[i,:],mindata[i,:])
    probas[:,i] = MLUtility.computeprobability(mocsvm_model[i].predict_margin(tempdata)).transpose()
    # stretch the probability values to range in [0,1]
    maxp = probas[:,i].max(0)
    minp = probas[:,i].min(0)
    probas[:,i] = (probas[:,i]-minp)/(maxp-minp)
  for i in range(data.shape[0]):
    a = probas[i,:]
    b = 1 - probas[i,:]
    denominator = 0
    denominator = denominator + scipy.multiply.reduce(a)
    for outer in range(len(a)-3):
        denominator = denominator + scipy.multiply.reduce(a[:-(len(a)-outer)+1])*b[outer+2]*scipy.multiply.reduce(a[outer+3:])
    denominator = denominator + scipy.multiply.reduce(a[:-1])*b[len(a)-1]
    denominator = denominator + b[0]*scipy.multiply.reduce(a[1:])
    numerator = 1
    for outer in range(probas.shape[1]):
        numerator = numerator*b[outer]
    op[i]=1 - (numerator/(1-denominator))
  return op