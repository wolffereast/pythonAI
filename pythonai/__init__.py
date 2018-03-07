import neuron
import perceptron

neuron1 = neuron.Neuron(2)
print
print neuron1.getWeight(0)
print neuron1.getWeight(1)

args = [1] * 2
print neuron1.activationFunction(args)
print '...'
print '...'

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

perceptron1 = perceptron.Perceptron(2)
retVal = perceptron1.train(andTest)
print retVal
