"""
Author: Rafail Islam
CSC790: Deep Learning
Fall 2020, Assignment #2
"""

import random 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import csv

def activate(sum):
    """ this function return either 1 or -1 based on the sign of sum
    """
    return sum>0 and 1 or -1

def misclassified(x,y,w1,w2,b, label):
    """ This function calculate sum from given inputs and returns True if
    the sign of the predicted output mathches with actual ouput otherwise returns
    False
    """
    sum = x*w1 + y*w2 + b
    
    if(activate(sum) == label):
        return False
    else:
        return True
    
    
def read_data(name):
    """ This function read data for a given file, and returns data as list
    """
    inputs=[]
    with open(name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
           
            row[0] = float(row[0])
            row[1] = float(row[1])
            row[2] = int(row[2])
            
            inputs.append(row)
        
    return inputs

def draw_canvas(input, weight1, weight2, bias):
    """ This function draws points based on their class and indicates if the point is missclassified by the perceptron.
    """
    for i in range(len(input)):
        
        #check if the point is misclassified 
        flag = misclassified(x = input[i][0],y= input[i][1], w1 = weight1,w2 = weight2, b = bias, label =input[i][2] )
        
        if(input[i][2] == -1):
            if(flag):
                plt.scatter(input[i][0],input[i][1],facecolors='r', edgecolors='g')
            else:
                plt.scatter(input[i][0],input[i][1],facecolors='none', edgecolors='g')
        else:
            if(flag):
                plt.scatter(input[i][0],input[i][1],color = 'b', facecolors='r', edgecolors='b')
            else:
                plt.scatter(input[i][0],input[i][1],color = 'b', facecolors='none', edgecolors='b')
        

def train(input_train, weight1,weight2,bias,n):
    """ This train function train the perceptron untill it can classified all points and
    return updated weights, bias, and required epoch
    """
    plt.ion()
    
    epoch=0
    while(True):
        epoch+=1
        
        # Training Perceptron 
        miss_Classified =0
        for index in range(len(input_train)):
            
            sum = bias + weight1 * input_train[index][0]+weight2 * input_train[index][1] 
            
            # evaluating output of the perceptron
            output = activate(sum)

            
            #updating weight and bias
            if output != input_train[index][2]:
                weight1 += n * input_train[index][2] * input_train[index][0]
                weight2 += n * input_train[index][2] * input_train[index][1]
                bias += n * input_train[index][2]*bias
                miss_Classified+=1
        
        # getting points using the weight and bias
        xx = np.linspace(-10,10,10)
        yy = ((-weight1*xx - bias)/float(weight2))
        
        # printing points on the canvus based on their class
        draw_canvas(input_train,weight1,weight2,bias)
        
        # drawwing classification line by learning perceptron 
        plt.plot(xx,yy,'-y')
        
        plt.draw()    
        plt.pause(0.1) # change to make fast or slow animation
        plt.clf()
        print("Epoch %3d Misclassification =%d"%(epoch+1,miss_Classified))
        if (miss_Classified==0):
            break;
    return weight1, weight2, bias, epoch+1

def test(input,weight1,weight2,bias):
    """this function test the perceptron accuracy with different test data
    """
    miss_Classified =0
    n = len(input)
    for index in range(n):
        
        sum = bias + weight1 * input[index][0]+ weight2 * input[index][1] 
        
        # evaluating output of the perceptron
        output = activate(sum)

        if output != input[index][2]:
            miss_Classified+=1
    xx = np.linspace(-10,10,10)
    yy = ((-weight1*xx - bias)/float(weight2))
    
    # printing points on the canvus based on their class
    draw_canvas(input,weight1,weight2,bias)
    
    # drawwing classification line by learning perceptron 
    plt.plot(xx,yy,'-y')
    
    plt.draw()    
    plt.pause(0.1) # change to make fast or slow animation
    #plt.clf()
    accuracy = ( (n-miss_Classified)/ n )*100
    return accuracy

def main():
    
    
    # Load input training data 
    input_train = read_data("train.csv")
    input_test = read_data("test.csv")
    
    learning_rate = [0.1,0.6,0.005,0.01,0.001]
    
    epoch_hist = []
    accuracy_hist = []
    
    # do train and test with different learning rate
    for n in learning_rate:
        random.seed(0)
        # initializing bias and with
        bias = random.uniform(-1,1.0)
        weight1 = random.uniform(-1,1.0)
        weight2 = random.uniform(-1,1.0)
        
        # train perceptron 
        weight1, weight2, bias, epoch = train(input_train,weight1,weight2,bias,n)
        epoch_hist.append(epoch)
        
        # test accuracy
        accuracy = test(input_test,weight1,weight2,bias)
        print("Test is ",accuracy," % accurate")
        accuracy_hist.append(accuracy)
    
    #showing epoch and accuracy for different learnig rate
    df = pd.DataFrame({
            "Learning Rate" : learning_rate,
            "Epoch" : epoch_hist,
            "Accuracy": accuracy_hist
            })
    
    ax = df.plot.bar(x="Learning Rate",y=["Epoch","Accuracy"])
    
    ax.set_xlabel('Learnign Rate')
    ax.set_ylabel('Epoch & Accuracy')
    plt.show()
    input ( "press [enter] to exit" )
main()