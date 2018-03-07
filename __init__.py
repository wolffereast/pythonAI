import neuron

neuron1 = neuron.Neuron(2)
print
print neuron1.getWeight(0)
print neuron1.getWeight(1)

args = [1] * 2
print neuron1.activationFunction(args)
