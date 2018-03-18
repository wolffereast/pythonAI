'''
Created on March 07, 2018

@author: Stephen Wolff
'''

import unittest
import perceptron
import re

from mock import patch

class TestPerceptron(unittest.TestCase):
    @patch('logging.Logger.info')
    def test_perceptron_and(self, mock):
        andTest = []

        andTestInnerItem = [0] * 2
        andTestItem = [andTestInnerItem]
        andTestItem.append(0)
        andTest.append(andTestItem)
        
        andTestInnerItem = [0]
        andTestInnerItem.append(1)
        andTestItem = [andTestInnerItem]
        andTestItem.append(0)
        andTest.append(andTestItem)
        
        andTestInnerItem = [1]
        andTestInnerItem.append(0)
        andTestItem = [andTestInnerItem]
        andTestItem.append(0)
        andTest.append(andTestItem)
        
        andTestInnerItem = [1] * 2
        andTestItem = [andTestInnerItem]
        andTestItem.append(1)
        andTest.append(andTestItem)
        
        perceptron1 = perceptron.Perceptron(.1, .2)
        perceptron1.train(andTest)
        
        self.assertNotEqual(None, mock.call_args)
        
        matches = re.match('succeeded after \d+ iterations', mock.call_args[0][0])
        self.assertNotEqual(re.match('succeeded after \d+ iterations', mock.call_args[0][0]), None)
    
    @patch('logging.Logger.info')
    def test_perceptron_or(self, mock):
        andTest = []

        andTestInnerItem = [0] * 2
        andTestItem = [andTestInnerItem]
        andTestItem.append(0)
        andTest.append(andTestItem)
        
        andTestInnerItem = [0]
        andTestInnerItem.append(1)
        andTestItem = [andTestInnerItem]
        andTestItem.append(1)
        andTest.append(andTestItem)
        
        andTestInnerItem = [1]
        andTestInnerItem.append(0)
        andTestItem = [andTestInnerItem]
        andTestItem.append(1)
        andTest.append(andTestItem)
        
        andTestInnerItem = [1] * 2
        andTestItem = [andTestInnerItem]
        andTestItem.append(1)
        andTest.append(andTestItem)
        
        perceptron1 = perceptron.Perceptron(.1, .2)
        perceptron1.train(andTest)
        
        self.assertNotEqual(None, mock.call_args)
        
        matches = re.match('succeeded after \d+ iterations', mock.call_args[0][0])
        self.assertNotEqual(re.match('succeeded after \d+ iterations', mock.call_args[0][0]), None)
 