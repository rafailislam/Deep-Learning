# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:52:37 2020

@author: rit1115
"""

import tensorflow as tf
import tensorflow.keras as ks
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
def load_data(problem_no):
    ''' this function returns dataset for traing and testing
    '''
    
    if(problem_no==1):
        
        trainDf = pd.read_csv('Exam02_data/Problem 1/p1_train.csv')
        testDf = pd.read_csv('Exam02_data/Problem 1/p1_test.csv')
    else:
        trainDf = pd.read_csv('Exam02_data/Problem2/p2_train.csv')
        testDf = pd.read_csv('Exam02_data/Problem2/p2_test.csv')
    
    # dataframe to numpy arrays
    train = trainDf.to_numpy()
    test = testDf.to_numpy()
   
    # split features and labels
    X_train,y_train = train[:,:-1].astype("float32"),train[:,-1].astype("float32")
    X_test, y_test = test[:,:-1].astype("float32"),test[:,-1].astype("float32")
    
    return X_train,y_train,X_test, y_test
def build_model(input_space):
    ''' this function return a compiled model
    '''
    # add layers
    x = ks.layers.Input(shape=(input_space,))
    hl = ks.layers.Dense(128,activation='relu')(x)
    hl = ks.layers.Dense(256,activation='relu')(hl)
    #hl = ks.layers.Dense(128,activation='relu')(hl)
    output = ks.layers.Dense(1,activation='softmax')(hl)
    
    model = ks.Model(inputs=x, outputs=output)
    
   
    
    return model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()
@tf.function
def train_step(model,inputs, labels):
    with tf.GradientTape() as tape:
    
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs)
        tf.print(predictions)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model,inputs, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs)
    print("test",predictions.shape)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
    test_accuracy(labels, predictions)

def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # set problem_no = 1 for problem-01, 2 for problem-02
    X_train,y_train,X_test, y_test = load_data(problem_no=1)
    model = build_model(X_train.shape[1])
    print(model.summary())
    # change verbose = 2 or 1 to show progress
    epochs = 200
    batch_size = 64
    acc_hist=[]
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        
        for i in range(len(X_train)):
            x = tf.expand_dims(X_train[i],axis=0)
            #print(x.shape)
            print(i)
            train_step(model,x,y_train[i])
        for j in range(len(X_test)):
            test_step(model,X_test,y_test)
        acc_hist.append(test_accuracy*100)
    plt.plot([i for i in range(epochs)],acc_hist)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs. Accuracy")
    plt.show()
main()
