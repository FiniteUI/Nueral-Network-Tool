#simple nueral network tool

import pickle
import os
import operator
import NueralNetwork_ActivationFunctions

class node:
    count = 0
    #node will just store inputs, and weights, and it's own value
    def __init__(self, activation, label = None):
        #list of nodes and weights that input to this node
        self.inputs = []
        self.value = 0
        self.rawValue = 0
        self.activation = activation

        #optional
        self.label = label
        self.id = node.count
        node.count += 1

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
        return len(self.inputs) - 1

    def removeInput(self, input):
        self.inputs.remove(input)

    def setInputWeightByIndex(self, index, weight):
        self.inputs[index][1] = weight

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
            #id = f'L{layer}-I{len(self.layers[layer])}'
            n = node(activation, label = label)
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

    def replaceLinkWithNode(self, index, activation, label = None):
        #add node in place of link
        link = self.links[index]
        inputLayer = self.getLayerOfNode(link[0])
        outputLayer = self.getLayerOfNode(link[1])

        if outputLayer - inputLayer == 1:
            #they are right next to eachother, we need to add a layer
            layer = outputLayer
            self.addLayer(layer)
        else:
            layer = outputLayer + inputLayer // 2

        #now add the node
        newNode = self.addNode(layer, activation, label = label)

        #delete link
        weightIn = self.links[index][2]
        input = self.links[index][0]
        output = self.links[index][1]
        self.removeLinkByIndex(index)

        #add new links in and out of node
        self.addLink(input, newNode, weightIn)
        self.addLink(newNode, output, 1)

        return newNode

    def getLayerOfNode(self, node):
        layer = None
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if self.layers[i][j] == node:
                    layer = i
                    break
        return layer

    def getNode(self, layer, index):
        return self.layers[layer][index]

    def addLink(self, input, output, weight):
        #also need to check they aren't on the same layer
        #also need to check input is before output
        #also need to add check for cycle

        #CHECK FOR DUPLICATES
        if [input, weight] not in output.inputs:
            index = output.addInput(input, weight)
            self.links.append([input, output, weight, index])
            return len(self.links) - 1
        return None

    def updateLinkByIndex(self, index, weight):
        self.links[index][1].setInputWeightByIndex(self.links[index][3], weight)

    def removeLinkByIndex(self, index):
        link = self.links[index]
        link[1].removeInput([link[0], link[2]])
        del self.links[index]

    def removeNode(self, node):
        #delete links if there are any
        for i in range(len(self.links) - 1, -1, -1):
            l = self.links[i]
            if l[0] == node or l[1] == node:
                #if this node is an input in a link, delete it from the input list in the output node
                if l[0] == node:
                    l[1].removeInput([l[0], l[2]])
                del self.links[i]
    
        #remove node from node list
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i]) - 1, -1, -1):
                if self.layers[i][j] == node:
                    del self.layers[i][j]

        #delete the node
        del node

    def removeNodeByIndex(self, layer, index):
        self.removeNode(self.layers[layer][index])

    def addLayer(self, layer = None):
        if layer == None:
            layer = len(self.layers) - 1
        self.layers.insert(layer, [])
        return layer

    def removeLayer(self, layer):
        #delete nodes in layer
        for i in range(len(self.layers[layer]) - 1, -1, -1):
            self.removeNode(self.layers[layer][i])

        #delete layer
        del self.layers[layer]

    def process(self):
        for i in range(1, len(self.layers)):
            for j in self.layers[i]:
                j.process()
                #print(f'ID: {j.id}, Value: {j.getValue()}')
        
        return self.layers[len(self.layers) - 1]


