#simple nueral network tool

import pickle
import os
import operator
import NeuralNetwork_ActivationFunctions

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

    def setActivationFunction(self, function):
        self.activation = function

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
        #print(f'Attempting to remove input {input} from node {self}...')
        #self.inputs.remove(input)
        for i in self.inputs:
            if i[0] == input[0]:
                del(i)

    def setInputWeightByIndex(self, index, weight):
        self.inputs[index][1] = weight

    def setInputWeight(self, input, weight):
        for i in self.inputs:
            if i[0] == input:
                i[1] = weight
                break

    def process(self):
        value = 0
        for i in self.inputs:
            #value = input value * weight
            value += i[0].getValue() * i[1]
        self.setValue(value)

class brain:
    id = 0

    def __init__(self, label = None):
        #list of layers, nodes in layers
        self.layers = [[], []]
        self.links = []
        self.id = brain.id
        brain.id += 1
        self.label = label

    def setNodeActivationFunction(self, node, function):
        node.setActivationFunction(function)
    
    def addNode(self, layer, activation, label = None):
        if layer < len(self.layers):
            n = node(activation, label = label)
            self.layers[layer].append(n)
            return n
        else:
            return None

    def addInput(self, label = None):
        n = self.addNode(0, activation=NeuralNetwork_ActivationFunctions.direct, label = label)
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
    
    def getOutputs(self):
        return self.layers[len(self.layers) - 1]

    def getSortedOutputs(self):
        outputs = self.getOutputs()
        outputs = [[x, outputs[x].label, outputs[x].value] for x in range(len(outputs))]
        outputs = sorted(outputs, key = operator.itemgetter(2), reverse = True)
        return outputs

    def getOutputByIndex(self, index):
        output = self.getNode(len(self.layers) - 1, index)
        return output.getValue()

    def replaceLinkWithNode(self, index, activation, label = None):
        #print(f"Attempting replaceLinkWithNode for network {self.label}, index {index}...")

        #add node in place of link
        link = self.links[index]
        inputLayer = self.getLayerOfNode(link[0])
        outputLayer = self.getLayerOfNode(link[1])

        if outputLayer - inputLayer == 1:
            #they are right next to eachother, we need to add a layer
            layer = outputLayer
            self.addLayer(layer)
        else:
            layer = (outputLayer + inputLayer) // 2

        #now add the node
        newNode = self.addNode(layer, activation, label = label)

        #delete link
        weightIn = self.links[index][2]
        input = self.links[index][0]
        output = self.links[index][1]
        self.removeLinkByIndex(index)

        #add new links in and out of node
        linkIn = self.addLink(input, newNode, weightIn)
        linkOut = self.addLink(newNode, output, 1)

        if linkIn == None or linkOut == None:
            print(f"Error adding link for replaceLinkWithNode for network {self.label}, index {index}. Link in: {linkIn}, Link Out: {linkOut}, Input: {input}, Output: {output}, Weight In: {weightIn}")

        return newNode

    def getLayerOfNode(self, node):
        #print(f"Getting layer of node {node}...")
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                if self.layers[i][j] == node:
                    #print(f"Node {node} found on layer {i}")
                    return i
        return None

    def getNode(self, layer, index):
        return self.layers[layer][index]

    def getNodeOutputs(self, layer, index):
        outputs = []
        node = self.getNode(layer, index)
        for i in self.links:
            if i[0] == node:
                outputs.append(i[1])
        
        return outputs

    def addLink(self, input, output, weight):
        inputLayer = self.getLayerOfNode(input)
        outputLayer = self.getLayerOfNode(output)

        #for some reason this happens sometimes... will have to look into it
        if inputLayer == None or outputLayer == None:
            print(f'Failed to add link on network {self.label} for input {input}, output {output} because input {inputLayer} or outputlayer {outputLayer} was none.')
            return None

        #check they aren't on the same layer, check input is before output
        if inputLayer >= outputLayer:
            print(f'Failed to add link on network {self.label} for input {input}, output {output} because input layer was greater than or equal to output layer.')
            return None
        
        #make sure input isn't output layer
        #technically the above should catch this but maybe not, something seems to be causing this
        if inputLayer == len(self.layers) - 1 or outputLayer == 0:
            print(f'Failed to add link on network {self.label} for input {input}, output {output} because input was output layer or output was input layer.')
            return None
        
        #make sure that link doesn't already exist
        links = [[i[0], i[1]] for i in self.links]
        if [input, output] in links:
            print(f'Failed to add link on network {self.label} for input {input}, output {output} because link already exists.')
            return None
        
        #CHECK FOR DUPLICATES
        if [input, weight] not in output.inputs:
            index = output.addInput(input, weight)
            #self.links.append([input, output, weight, index])
            self.links.append([input, output, weight])
            return len(self.links) - 1
        return None

    def updateLinkByIndex(self, index, weight):
        #self.links[index][1].setInputWeightByIndex(self.links[index][3], weight)
        self.links[index][1].setInputWeight(self.links[index][0], weight)
        self.links[index][2] = weight

    def getLinkWeightByIndex(self, index):
        return self.links[index][2]

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

    def getLayer(self, index):
        return self.layers[index]

    def process(self):
        for i in range(1, len(self.layers)):
            for j in self.layers[i]:
                j.process()
                #print(f'ID: {j.id}, Value: {j.getValue()}')
        
        return self.layers[len(self.layers) - 1]
    
    def saveStructure(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        file = os.path.join(path, name) + '.ns'
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def loadStructure(file):
        with open(file, 'rb') as f:
            b = pickle.load(f)
        return b

    def getMaxLinkCount(self):
        count = 0
        for i in range(len(self.layers)):
            for j in range(i + 1, len(self.layers)):
                count += len(self.layers[i]) * len(self.layers[j])
        return count
    
    def getLinkCount(self):
        return len(self.links)

    def getLayerCount(self):
        return len(self.layers)
    
    def checkDuplicateLink(self, input, output):
        link = [input, output]
        links = [[i[0], i[1]] for i in self.links]
        if link in links:
            return True
        else:
            return False

    def getOpenLinks(self):
        links = []
        #for every layer
        for i in range(len(self.layers)):
            #for every next layer
            for j in range(i + 1, len(self.layers)):
                #create a link for each
                for k in self.layers[i]:
                    for l in self.layers[j]:
                        if self.checkDuplicateLink(k, l) == False:
                            links.append([k, l])

        return links
