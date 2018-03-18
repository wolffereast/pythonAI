'''
Created on March 03, 2018

@author: Stephen Wolff
'''

import neuron
import logging

class Perceptron():
    # alpha is the learning rate. Delta is the threshold.
    def __init__(self, alpha=.1, delta=.2):
        # Set the basic logger config.
        logging.basicConfig()

        self.alpha = alpha
        self.delta = delta
        # create a neuron that takes 2 inputs, has random starting weights from
        # -.5 to .5, and uses the default sign activation function.
        self.neuron = neuron.Neuron(2, [], 'step')
        
    def activate(self, inputs):
        if len(inputs) != 2:
            logger = logging.getLogger(__name__)
            logger.error('There must be 2 inputs to activate the perceptron.')
            return False
        
        return self.neuron.activate(inputs, self.delta)
    
    '''
    training values is a multidimensional array with the first index being a
    tuple of inputs and the second being the expected output. The Perceptron
    will train until it reaches the expected value or it hits maxIterations.
    train really just error checks then sends the inputs on to the train tester.
    '''
    def train(self, trainingValues, maxIterations=100):
        print 'beginning training with weights: {0}'.format(self.neuron.getWeights())
        print 'and training values: {0}'.format(trainingValues)
        for trainingValue in trainingValues:
            # Catch the type error, this will verify that the row is iterable. 
            try:
                if len(trainingValue[0]) != 2:
                    logger = logging.getLogger(__name__)
                    logger.error('Invalid number of inputs in the training value {0}'.format(trainingValue))
                    return False
            except TypeError: # catch when for loop fails
                logger = logging.getLogger(__name__)
                logger.error('invalid training value: the first element in the array must have a length of 2. Offending row: {0}'.format(trainingValue))
                return False
        return self.trainHelper(trainingValues, maxIterations, 0)
        
    # Validation has already occurred in the train function.
    def trainHelper(self, trainingValues, maxIterations, iteration):
        if iteration == maxIterations:
            logger = logging.getLogger(__name__)
            logger.info('failed after {0} iterations'.format(iteration))
            
            retVal = self.getTrainingOutputs(trainingValues)
            retVal.append('failed after {0} iterations'.format(iteration))
            return retVal

        failed = False
        for trainingValue in trainingValues:
            output = self.activate(trainingValue[0])
            # Update the weights if necessary.
            if output != trainingValue[1]:
                failed = True
                print 'iteration {0} failed with output {1}'.format(iteration, output)
                print 'training inputs: {0}'.format(trainingValue)
                print 'weights: {0}'.format(self.neuron.getWeights())
                self.updateWeights(trainingValue[0], trainingValue[1] - output)
                print 'weights after update: {0}'.format(self.neuron.getWeights())
    
        # If we failed, iterate.
        if failed:
            return self.trainHelper(trainingValues, maxIterations, iteration + 1)
        
        logger = logging.getLogger(__name__)
        logger.info('succeeded after {0} iterations'.format(iteration))
        
        retVal = self.getTrainingOutputs(trainingValues)
        retVal.append('succeeded after {0} iterations'.format(iteration))
        return retVal

    def updateWeights(self, inputs, error):
        for i in range(0, len(inputs)):
            self.neuron.setWeight(i, self.neuron.getWeight(i) + (self.alpha * inputs[i] * error))
            
    def getWeights(self):
        return self.neuron.getWeights()
    
    def getTrainingOutputs(self, trainingValues):
        retVal = []
        for trainingValue in trainingValues:
            trainingValue.append(self.activate(trainingValue[0]))
            retVal.append(trainingValue)
        
        return retVal
