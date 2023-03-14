#simple nueral network tool
#https://www.youtube.com/watch?v=aircAruvnKk

import math

class node:
    def setValue(self, value, input = False):
        value = float(value)
        self.unfixedValue = value

        if input:
            self.value = value
        else:
            self.value = self.activation(value)
        print(f'ID: {self.id}, New Value: {self.value}')

    def getValue(self):
        return self.value
    
    def setWeights(self, weights):
        self.weights = weights
    
    def setWeight(self, index, weight):
        self.weights[index] = weight

    def addWeight(self, weight):
        self.weights.append(weight)

    def clearWeights(self):
        self.setWeights([])

    def setBias(self, bias):
        self.bias = bias

    def setInputs(self, inputs):
        self.inputs = inputs

    def process(self):
        n = 0
        for i in range(0, len(self.inputs)):
            print(f'ID: {self.id}, Input: {self.inputs[i].id}, Input Value: {self.inputs[i].getValue()}, Weight: {self.weights[i]}')
            n += self.inputs[i].getValue() * self.weights[i]
        print(f'ID: {self.id}, Input Sum: {n}, Bias: {self.bias}, Sum + Bias: {n + self.bias}')
        n += self.bias
        self.unfixedValue = n
        self.setValue(n)

    def sigmoid(value):
        s = 0 - value
        s = math.pow(math.e, s)
        s += 1
        s = 1/s
        return s

    def hyperbolicTangent(value):
        return math.tanh(value)
    
    def rectifiedLinearUnits(value):
        if value < 0:
            value = 0
        return value
    
    def positiveNegative(value):
        if value <= 0:
            value = 0
        else:
            value = 1
        return value
    
    def __init__(self, bias, weights = [], inputs = [], id = None, label = None, activation = positiveNegative):
        self.id = id
        self.bias = bias

        #weights are always coming from previous layer
        self.weights = weights
        self.inputs = inputs

        #value should always be between 0 and 1
        self.value = 0
        self.unfixedValue = 0

        self.label = label
        self.activation = activation

class brain:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.layers = [self.inputs, self.outputs]
        self.nodes = []

    def addNode(self, layer, bias = 0, weightsIn = None, weightsOut = None, inputs = [], label = None, activation = node.positiveNegative):
        #0 is always input
        #len(self.layers) - 1 is always output
        #layer must already exist
        id = f'L{layer}-I{len(self.layers[layer])}'

        if weightsIn == None:
            if layer > 0:
                weightsIn = self.getWeightsList(layer - 1)
        
        if inputs == []:
            if layer > 0:
                inputs = self.layers[layer - 1]

        n = node(bias, weightsIn, inputs, id = id, label = label, activation = activation)
        self.nodes.append(n)
        self.layers[layer].append(n)

        if weightsOut == None:
            if layer < len(self.layers) - 1:
                weightsOut = self.getWeightsList(layer + 1)

        #now need to add weights for that node to next layer
        if layer < len(self.layers) - 1:
            for i in range(len(self.layers[layer + 1])):
                self.layers[layer + 1][i].addWeight(weightsOut[i])
                self.layers[layer + 1][i].setInputs(inputs)

        return n
    
    def addInput(self, bias = 0, weightsOut = None, label = None):
        n = self.addNode(0, bias, weightsOut = weightsOut, label = label)
        return n

    def addOutput(self, bias = 0, weightsIn = None, inputs = None, label = None, activation = node.positiveNegative):
        if inputs == None:
            inputs = self.layers[len(self.layers) - 2]
        n = self.addNode(len(self.layers) - 1, bias, weightsIn = weightsIn, inputs = inputs, label = label, activation = activation)
        return n
    
    def addLayer(self, index = None, bias = 0, weightsIn = None, weightsOut = None):
        if index == None:
            index = len(self.layers) - 1

        #clear out weights in from next layer
        for i in self.layers[index]:
            i.clearWeights()

        #first add the layer to our layer list
        self.layers.insert(index, [])
        
        return self.layers[index]
    
    def setInput(self, input, value):
        input.setValue(value, True)

    def setInputByIndex(self, index, value):
        brain.setInput(self.inputs[index], value)

    def getOutput(self, output):
        return output.getValue()
    
    def getOutputByIndex(self, index):
        return self.getOutput(self.outputs[index])

    def process(self):
        for l in range(1, len(self.layers)):
            for n in self.layers[l]:
                n.process()

    def getWeightsList(self, layer):
        weights = [1 for i in self.layers[layer]]
        return weights