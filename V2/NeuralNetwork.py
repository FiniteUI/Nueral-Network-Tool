#simple nueral network tool
#https://www.youtube.com/watch?v=aircAruvnKk

import pickle
import os
import operator
import NueralNetwork_ActivationFunctions

class node:
    #node will just store inputs, and weights, and it's own value
    def __init__(self, activation, label = None, id = None):
        #list of nodes and weights that input to this node
        self.inputs = []
        self.value = 0
        self.rawValue = 0
        self.activation = activation

        #optional
        self.label = label
        self.id = id

    def setValue(self, value):
        self.rawValue = value
        self.value = self.activation(value)

    def getValue(self):
        return self.value
    
    def getRawValue(self):
        return self.rawValue

    def addInput(self, node, weight):
        input = [node, weight]
        self.inputs.append(input)

    #need to add a way to remove inputs

    #need to add way to modify weight

    def process(self):
        value = 0
        for i in self.inputs:
            #value = input value * weight
            value += i[0].getValue() * i[1]
        self.setValue(value)

class brain:
    def __init__(self):
        #list of layers, nodes in layers
        self.layers = [[], []]
        self.links = []
    
    def addNode(self, layer, activation, label = None):
        if layer < len(self.layers):
            id = f'L{layer}-I{len(self.layers[layer])}'
            n = node(activation, label = label, id = id)
            self.layers[layer].append(n)
            return n
        else:
            return None

    def addInput(self, label = None):
        n = self.addNode(0, activation=NueralNetwork_ActivationFunctions.direct, label = label)
        return n

    def addOutput(self, activation, label = None):
        n = self.addNode(len(self.layers) - 1, activation=activation, label = label)
        return n

    def setInput(self, input, value):
        input.setValue(value)

    def setInputByIndex(self, index, value):
        input = self.getNode(0, index)
        input.setValue(value)

    def getOutput(self, output):
        return output.getValue()
    
    def getOutputByIndex(self, index):
        output = self.getNode(len(self.layers) - 1, index)
        return output.getValue()

    def addNodeBetween(self):
        pass

    def getNode(self, layer, index):
        return self.layers[layer][index]

    def addLink(self, input, output, weight):
        output.addInput(input, weight)
        self.links.append([input, output, weight])

    def updateLink(self):
        pass

    def removeLink(self):
        pass

    def removeNode(self):
        pass

    def addLayer(self, layer = None):
        if layer == None:
            layer = len(self.layers) - 1
        self.layers.insert(layer, [])

    def removeLayer(self):
        pass

    def process(self):
        for i in range(1, len(self.layers)):
            for j in self.layers[i]:
                j.process()
                #print(f'ID: {j.id}, Value: {j.getValue()}')
        
        return self.layers[len(self.layers) - 1]


