'''
Created on March 07, 2018

@author: Stephen Wolff
'''

import unittest
import neuron

from mock import patch

class TestNeuron(unittest.TestCase):
    # The patch call sets the mock to an instance of the supplied class.
    # Ensure we catch a neuron instantiated with no inputs.
    @patch('logging.Logger.error')
    def test_neuron_no_inputs(self, mock):
        neuron1 = neuron.Neuron(0)
        mock.assert_called_with('Invalid number of inputs. Number of inputs must be a positive integer.')
    
    # Test passing a set of weights to the initialization.
    def test_initial_weight_set(self):
        testWeights = [.3]
        testWeights.append(-.7)
        neuron1 = neuron.Neuron(2, testWeights)
        self.assertEqual(neuron1.getWeights(), testWeights)
        
    # Test set weights with an argument that is not an array.
    @patch('logging.Logger.error')
    def test_set_weights_with_incorrect_type(self, mock):
        neuron1 = neuron.Neuron(3)
        neuron1.setWeights(None)
        mock.assert_called_with('Invalid weights passed to setWeights. Expected an array, received None')
    
    # Test set weights with an argument that is the wrong length.
    @patch('logging.Logger.error')
    def test_set_weights_with_incorrect_num_weights(self, mock):
        neuron1 = neuron.Neuron(3)
        neuron1.setWeights([.5])
        mock.assert_called_with('Invalid number of weights passed to setWeights: 1. Expected 3')
        
    # Test set weights with a correct arg.
    def test_set_weights(self):
        neuron1 = neuron.Neuron(3)
        weights = [.5]
        weights.append(-.2)
        weights.append(.3)
        neuron1.setWeights(weights)
        self.assertEqual(neuron1.getWeights(), weights)
        
    # Test set weight with an invalid index.
    @patch('logging.Logger.error')
    def test_set_weight_bad_index(self, mock):
        neuron1 = neuron.Neuron(2)
        neuron1.setWeight(-1, [.5])
        mock.assert_called_with('Cannot set the weight of the non existent input num: -1')
        neuron1.setWeight(3, [.5])
        mock.assert_called_with('Cannot set the weight of the non existent input num: 3')
        