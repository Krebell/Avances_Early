#!/usr/bin/python
# -*- coding: utf8 -*-
import sys
import operator
import os

import pandas as pd
import numpy as np
import pickle
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import nltk
from nltk.corpus import stopwords
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import CuDNNLSTM,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from theano.tensor import basic as tensor
import argparse
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import re
import functools
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def as_keras_metric(method):    
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


def partially_linear(true_dist, coding_dist):
        loss = 0
        TIME = 1#Es probable que este cambie
        N_C = 21
        batch = 32
	print tf.shape(true_dist)
	print tf.shape(coding_dist)
        for t in range (TIME):
		print t
                term1 = true_dist[:,t] * tf.log(coding_dist[:,t]+0.0000001)
                term2 = (1-true_dist[:,t]) * tf.log(1-coding_dist[:,t]+0.0000001)
                loss = loss + np.double(1)/N_C * tf.add(term1,term2*np.double(t)/TIME)

        return -loss/batch

#Primero se necesita obtener todo el train, tanto positivo como negativo
textos=[""]*486
labels=[0]*486
pos=0
neg=0
for i in range(10):
	cont=0
	for j in range(2):
		path=""		
		if(j+1)==1:
			path="./train-completo/pos"+str(i+1)+".txt"
		if(j+1)==2:
			path="./train-completo/neg"+str(i+1)+".txt"
		arch=open(path)
		for linea in arch.readlines():
			if(j+1)==1:
				labels[cont]=1
				pos+=1
			else:
				labels[cont]=0
				neg+=1
			line=linea.strip('\n').lower().split('\t');						
			#linea=' '.join([word for word in linea.split() if word not in (stopwords.words('english'))])
			#linea=re.sub(r'[^\w]', ' ', linea)
			textos[cont]=textos[cont]+" "+line[0]
			#print path,line[1]
			cont+=1
		#print cont
###Una vez que se tiene cargado los textos, se procede a determinar el texto mas largo
maxl=3000
minl=1000000
"""for t in textos:
	tama=len(t.strip().split())
	if maxl<tama:
		maxl=tama
	if minl>tama:
		minl=tama
#print maxl
#print minl
#print maxl/486
"""	
labels=np.array(labels)	
corpus=[nltk.word_tokenize(sent.decode('utf-8')) for sent in textos]
model = Word2Vec(min_count=2, window=5, size=100, sample=1e-4, negative=5, workers=7, sg=1)
model.build_vocab(corpus)
model.train(corpus,total_examples=model.corpus_count,epochs=10)
preweigth=model.wv.syn0
vocabulario,embedding=preweigth.shape


def word2index(word):
	return model.wv.vocab[word].index

def idx2word(idx):
	return model.wv.index2word[idx]


print "Tamaños de los embedding:",preweigth.shape
#Ahora se necesita crear y cargar los datos del train
mentrena=np.zeros([len(textos),maxl],dtype=np.int32)

i=0
for doc in textos:
	j=0
	linea=doc.strip().split()
	for word in linea:
		if j<maxl:
			try:
				mentrena[i,j]=word2index(word)
			except:
				if j<maxl:
					mentrena[i,j]=0
		j+=1
	i+=1


print "Positivos:",pos
print "Negativos:",neg
print "Shape mentrena:",mentrena.shape
print "Shape labels:",labels.shape
#mentrena=np.expand_dims(mentrena,axis=0)
#Ahora se procede a crear la red lstm

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

lstm=Sequential()
lstm.add(Embedding(input_dim=vocabulario,output_dim=embedding,weights=[preweigth],trainable=False,input_length=maxl,mask_zero=True))
lstm.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, input_shape=(maxl,),return_sequences=False))#El numero de secuencias regresadas es igual al numero de unidades que se tengan
lstm.add(Dense(1,activation='sigmoid'))
#lstm.add(Dense(1,activation='sigmoid'))
lstm.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
print (lstm.summary())
lstm.fit(mentrena,labels,epochs=30)

#Una vez que se tiene el modelo creado se procede a crear los datos de test de forma secuencial-incremental segun los chunks y las predicciones

#Primero se cargan los textos 
chunk1=[""]*401
chunk2=[""]*401
chunk3=[""]*401
chunk4=[""]*401
chunk5=[""]*401
chunk6=[""]*401
chunk7=[""]*401
chunk8=[""]*401
chunk9=[""]*401
chunk10=[""]*401
post1=[0]*401
post2=[0]*401
post3=[0]*401
post4=[0]*401
post5=[0]*401
post6=[0]*401
post7=[0]*401
post8=[0]*401
post9=[0]*401
post10=[0]*401

nega=np.zeros((349,),dtype=int)
posi=np.ones((52,),dtype=int)
clas=np.concatenate((posi,nega),axis=0)
textos=[]
for i in range(10):
	cont=0
	for j in range(2):
		path=""		
		if(j+1)==1:
			path="./test-completo/pos"+str(i+1)+".txt"
		if(j+1)==2:
			path="./test-completo/neg"+str(i+1)+".txt"
		arch=open(path)
		for linea in arch.readlines():			
			line=linea.strip('\n').lower().split('\t');			
			#linea=' '.join([word for word in linea.split() if word not in (stopwords.words('english'))])
			#linea=re.sub(r'[^\w]', ' ', linea)
			if (i+1)==1:				
				chunk1[cont]=chunk1[cont]+" "+line[0]
				post1[cont]=int(line[1])
			if (i+1)==2:				
				chunk2[cont]=chunk2[cont]+" "+line[0]
				post2[cont]=int(line[1])
			if (i+1)==3:				
				chunk3[cont]=chunk3[cont]+" "+line[0]
				post3[cont]=int(line[1])
			if (i+1)==4:				
				chunk4[cont]=chunk4[cont]+" "+line[0]
				post4[cont]=int(line[1])
			if (i+1)==5:				
				chunk5[cont]=chunk5[cont]+" "+line[0]
				post5[cont]=int(line[1])
			if (i+1)==6:				
				chunk6[cont]=chunk6[cont]+" "+line[0]
				post6[cont]=int(line[1])
			if (i+1)==7:				
				chunk7[cont]=chunk7[cont]+" "+line[0]
				post7[cont]=int(line[1])
			if (i+1)==8:				
				chunk8[cont]=chunk8[cont]+" "+line[0]
				post8[cont]=int(line[1])
			if (i+1)==9:				
				chunk9[cont]=chunk9[cont]+" "+line[0]
				post9[cont]=int(line[1])
			if (i+1)==10:				
				chunk10[cont]=chunk10[cont]+" "+line[0]
				post10[cont]=int(line[1])
			cont+=1

#La matriz de evaluacion es incremental conforme a los chunks leidos, por tanto se crea 10 veces en un ciclo for
for m in range(10):
	test=np.zeros([len(chunk1),maxl],dtype=np.int32)
	posttotal=np.zeros((401,),dtype=int)#aqui se guarda el total de post usados para despues calcular ERDE
	i=0
	for doc in range(len(chunk1)):#todos los chunks tienen el mismo tamaño
		j=0
		linea=" "
		if (m+1)==1:
			linea=chunk1[doc].strip().split()
			posttotal[doc]=post1[doc]
		if (m+1)==2:
			linea=" ".join([chunk1[doc],chunk2[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]
		if (m+1)==3:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]
		if (m+1)==4:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]
		if (m+1)==5:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]
		if (m+1)==6:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc],chunk6[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]+post6[doc]
		if (m+1)==7:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc],chunk6[doc],chunk7[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]+post6[doc]+post7[doc]
		if (m+1)==8:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc],chunk6[doc],chunk7[doc],chunk8[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]+post6[doc]+post7[doc]+post8[doc]
		if (m+1)==9:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc],chunk6[doc],chunk7[doc],chunk8[doc],chunk9[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]+post6[doc]+post7[doc]+post8[doc]+post9[doc]
		if (m+1)==10:
			linea=" ".join([chunk1[doc],chunk2[doc],chunk3[doc],chunk4[doc],chunk5[doc],chunk6[doc],chunk7[doc],chunk8[doc],chunk9[doc],chunk10[doc]]).strip().split()
			posttotal[doc]=post1[doc]+post2[doc]+post3[doc]+post4[doc]+post5[doc]+post6[doc]+post7[doc]+post8[doc]+post9[doc]+post10[doc]

		for word in linea:
			if j<maxl:
				try:
					test[i,j]=word2index(word)
				except:
					if j<maxl:
						test[i,j]=0
			j+=1
		i+=1
	print "Shape test:",test.shape
	#Una vez que se tiene la matriz de test dependiendo del chunk que se este leyendo, se procede a hacer las predicciones
	#Recordar que se necesita guardar el punto en el que la prediccion fue hecha para los que son verdaderos con cierto % de confianza
	finalpost=np.zeros((401,),dtype=int)#aqui se guardara el numero de chunks usados para la prediccion	
	pproba=lstm.predict(test)		
	pclases=lstm.predict_classes(test)	
	for p in range(len(pclases)):
		if pclases[p]==1 and pproba[p]>0.5:
			finalpost[p]=posttotal[p]

#Ahora se calcula el erde
erde5=np.zeros((len(clas),),dtype=float)
erde50=np.zeros((len(clas),),dtype=float)
for i in range(len(clas)):	
	if(pclases[i] == 1 and clas[i] == 0):
		erde5[i] = float(len(posi))/len(clas)
		erde50[i] = float(len(posi))/len(clas)
	elif(pclases[i] == 0 and clas[i] == 1):
		erde5[i] = 1.0
		erde50[i] = 1.0
	elif(pclases[i] == 1 and clas[i] == 1):
		erde5[i] = 1.0 - (1.0/(1.0+np.exp(finalpost[i]-5)))##Se necesita cambiar por el numero correcto de post del usuario, por mientras se deja el promedio
		erde50[i] = 1.0 - (1.0/(1.0+np.exp(finalpost[i]-50)))##Se necesita cambiar por el numero correcto de post del usuario, por mientras se deja el promedio				
	elif(pclases[i] == 0 and clas[i] == 0):
		erde5[i] = 0.0
		erde50[i] = 0.0

print "Erde 5:",erde5.mean()*100
print "Erde 50:",erde50.mean()*100
#precision = float(cont)/cont2
#recall = float(cont)/cont3
#F1 = 2 * (precision * recall) / (precision + recall)

#print "Precision:",precision
#print "Recall:",recall
#print "F1:",F1
for i in range(len(pclases)):
	print pclases[i],clas[i]

print accuracy_score(clas,pclases),precision_score(clas,pclases),recall_score(clas,pclases),f1_score(clas,pclases,average='macro'),f1_score(clas,pclases,average='micro')
























