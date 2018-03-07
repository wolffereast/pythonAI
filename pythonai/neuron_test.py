import unittest
import neuron

from mock import patch

class TestNeuron(unittest.TestCase):
    # The patch call sets the mock to an instance of the supplied class.
    @patch('logging.Logger.error')
    def test_neuron_no_inputs(self, mock):
        neuron1 = neuron.Neuron(0)
        mock.assert_called_with('Invalid number of inputs. Number of inputs must be a positive integer.')
    