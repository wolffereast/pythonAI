'''
Created on March 03, 2018

@author: Stephen Wolff
'''

from random import random
from math import fabs
from decimal import *

class Neuron():
    def __init__(self, numInputs=0, inputWeights=[], activationFunction=None):
        if 1 > numInputs:
            print 'Invalid number of inputs. Number of inputs must be a positive integer.'
            return None
        self.numInputs = numInputs
        if 'step' == activationFunction:
            self.activationFunction = self.stepHandler
        else:
            self.activationFunction = self.signHandler
        
        if numInputs == len(inputWeights):
            self.inputWeights = inputWeights
            return
        
        self.randomizeWeights()
            
        
        
    def getWeight(self, inputNum=0):
        if 0 > inputNum or inputNum >= self.numInputs:
            print 'Cannot get the weight of the non existent input num: %i' % inputNum
            return False
        
        return self.inputWeights[inputNum]
    
    def getWeights(self):
        return self.inputWeights
    
    def getNumInputs(self):
        return self.numInputs
    
    def setWeight(self, inputNum, weight):
        if 0 > inputNum or inputNum >= self.numInputs:
            print 'Cannot set the weight of the non existent input num: %i' % inputNum
            return False
        
        self.inputWeights[inputNum] = weight
        
    def randomizeWeights(self, minimum=-.5, maximum=.5, sigFigs=1):
        if minimum >= maximum:
            print 'The minimum weight must be less than the maximum weight. {0} !< {1}'.format(minimum, maximum)
            return False
        
        rangeNum = maximum - minimum
        absMinimum = fabs(minimum)
        
        self.inputWeights = [(round(random(),sigFigs) * rangeNum - absMinimum) for _ in range(self.numInputs)]
    
    '''
    Default activation function for a neuron. Used if no activation function
    is passed during initialization.
    '''    
    def signHandler(self, inputs=[], delta=0):
        # Ensure the number of inputs is equal to the number of weights
        if len(inputs) != self.numInputs:
            print 'The number of inputs must equal the number of weights'
            return False 
        
        total = 0
        for i in range(0, self.numInputs):
            total += self.inputWeights[i] * inputs[i]
        
        return 1 if total - delta >= 0 else -1
    
    # The Perceptron requires a step activation function.
    def stepHandler(self, inputs=[], delta=0):
        # Ensure the number of inputs is equal to the number of weights
        if len(inputs) != self.numInputs:
            print 'The number of inputs must equal the number of weights'
            return False 
        
        total = 0
        for i in range(0, self.numInputs):
            total += self.inputWeights[i] * inputs[i]
        
        return 1 if total - delta >= 0 else 0
