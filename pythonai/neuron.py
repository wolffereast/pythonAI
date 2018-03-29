'''
Created on March 03, 2018

@author: Stephen Wolff
'''

from random import randrange, uniform
from math import fabs, exp
from decimal import *
import logging

class Neuron():
    '''
    The input weights each weight one of the inputs. Those inputs are aggregated
    then run through the activationFunction. The results of the
    activationFunction are then optionally weighted by the returnWeight and
    returnWeightModifier.
    The returnWeightModifier defaults to 0, which negates the returnWeight.
    Setting both the return weight and return weight modifier to a non zero
    number will alter the output of the activated neuron.
    '''
    def __init__(self, numInputs=0, inputWeights=[], activationFunction=None, returnWeight=0, returnWeightModifier=0):
        # Set the basic logger config.
        logging.basicConfig()
        
        if 1 > numInputs:
            logger = logging.getLogger(__name__)
            logger.error('Invalid number of inputs. Number of inputs must be a positive integer.')
            return None
        
        self.numInputs = numInputs
        if 'step' == activationFunction:
            self.activate = self.stepHandler
        elif 'sigmoid' == activationFunction:
            self.activate = self.sigmoidHandler
        else:
            self.activate = self.signHandler
            
        self.returnWeight = returnWeight
        self.returnWeightModifier = returnWeightModifier
        
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
    
    def getReturnWeight(self):
        return self.returnWeight
    
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
        
    def setReturnWeight(self, weight):
        self.returnWeight = weight
        
    def randomizeWeights(self, minimum=-.5, maximum=.5):
        if minimum >= maximum:
            logger = logging.getLogger(__name__)
            logger.error('The minimum weight must be less than the maximum weight. {0} !< {1}'.format(minimum, maximum))
            return False
        
        self.inputWeights = [uniform(minimum, maximum) for _ in range(self.numInputs)]
    
    # Handler error handling and shared functionality.
    def handlerHelper(self, inputs=[], delta=0):
        # Ensure the number of inputs is equal to the number of weights
        if len(inputs) != self.numInputs:
            logger = logging.getLogger(__name__)
            logger.error('The number of inputs must equal the number of weights')
            return False 
        
        total = 0
        for i in range(0, self.numInputs):
            total += self.inputWeights[i] * inputs[i]
            
        return total
    
    '''
    Default activation function for a neuron. Used if no activation function
    is passed during initialization.
    '''    
    def signHandler(self, inputs=[], delta=0):
        total = self.handlerHelper(inputs, delta)
        if (False == total):
            return False
        retVal = 1 if total - delta >= 0 else -1
        return retVal + (self.returnWeight * self.returnWeightModifier)
    
    def stepHandler(self, inputs=[], delta=0):
        total = self.handlerHelper(inputs, delta)
        if (False == total):
            return False
        retVal = 1 if total - delta >= 0 else 0
        return retVal + (self.returnWeight * self.returnWeightModifier)
    
    def sigmoidHandler(self, inputs=[], delta=0):
        total = self.handlerHelper(inputs, delta)
        if (False == total):
            return False
        retVal = 1 / (1 + exp(-1 * total))
        return retVal + (self.returnWeight * self.returnWeightModifier)
        
        
