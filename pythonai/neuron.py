'''
Created on March 03, 2018

@author: Stephen Wolff
'''

from random import randrange
from math import fabs
from decimal import *
import logging

class Neuron():
    def __init__(self, numInputs=0, inputWeights=[], activationFunction=None):
        # Set the basic logger config.
        logging.basicConfig()
        
        if 1 > numInputs:
            logger = logging.getLogger(__name__)
            logger.error('Invalid number of inputs. Number of inputs must be a positive integer.')
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
            logger = logging.getLogger(__name__)
            logger.error('Cannot get the weight of the non existent input num: {0}'.format(inputNum))
            return False
        
        return self.inputWeights[inputNum]
    
    def getWeights(self):
        return self.inputWeights
    
    def getNumInputs(self):
        return self.numInputs
    
    def setWeight(self, inputNum, weight):
        if 0 > inputNum or inputNum >= self.numInputs:
            logger = logging.getLogger(__name__)
            logger.error('Cannot set the weight of the non existent input num: {0}'.format(inputNum))
            return False
        
        self.inputWeights[inputNum] = weight
        
    def setWeights(self, weights):
        try:
            if len(weights) != self.numInputs:
                logger = logging.getLogger(__name__)
                logger.error('Invalid number of weights passed to setWeights: {0}. Expected {1}'.format(len(weights), self.numInputs))
                return False
        except TypeError: # catch when for loop fails
            logger = logging.getLogger(__name__)
            logger.error('Invalid weights passed to setWeights. Expected an array, received {0}'.format(weights))
            return False
        
        self.inputWeights = weights
        
    def randomizeWeights(self, minimum=-50, maximum=50):
        if minimum >= maximum:
            logger = logging.getLogger(__name__)
            logger.error('The minimum weight must be less than the maximum weight. {0} !< {1}'.format(minimum, maximum))
            return False
        
        self.inputWeights = [randrange(minimum, maximum) for _ in range(self.numInputs)]
    
    '''
    Default activation function for a neuron. Used if no activation function
    is passed during initialization.
    '''    
    def signHandler(self, inputs=[], delta=0):
        # Ensure the number of inputs is equal to the number of weights
        if len(inputs) != self.numInputs:
            logger = logging.getLogger(__name__)
            logger.error('The number of inputs must equal the number of weights')
            return False 
        
        total = 0
        for i in range(0, self.numInputs):
            total += self.inputWeights[i] * inputs[i]
        
        return 1 if total - delta >= 0 else -1
    
    def stepHandler(self, inputs=[], delta=0):
        # Ensure the number of inputs is equal to the number of weights
        if len(inputs) != self.numInputs:
            logger = logging.getLogger(__name__)
            logger.error('The number of inputs must equal the number of weights')
            return False 
        
        total = 0
        for i in range(0, self.numInputs):
            total += self.inputWeights[i] * inputs[i]
        
        return 1 if total - delta >= 0 else 0
