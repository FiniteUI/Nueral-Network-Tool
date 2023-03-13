#simple nueral network tool
#https://www.youtube.com/watch?v=aircAruvnKk


#oh so technically ALL nuerons in a layer are connected to all others
#it's the weights that change

import math

class nueron:
    #for now, simple nuerons that hold a 0-1 value
    def __init__(self, bias = 0):
        self.value = 0
        #self.layer = layer
        self.bias = bias
        self.inputs = []
        self.outputs = []
    
    def set(self, value):
        value = float(value)
        value -= self.bias
        self.value = nueron.sigmoid(value)

    def getValue(self):
        return self.value
    
    def readInputs(self):
        sum = 0
        for i in self.inputs:
            sum += i.propogate()
        
        self.set(sum)

    def addInput(self, input):
        self.inputs.append(input)

    def addOutput(self, output):
        self.outputs.append(output)

    def removeOutput(self, output):
        self.outputs.remove(output)

    def removeInput(self, input):
        self.inputs.remove(input)

    def sigmoid(value):
        #for now using sigmoid activation function
        s = 0 - value
        s = math.pow(math.e, s)
        s += 1
        s = 1/s
        return s

class link:
    #connect two nuerons
    def __init__(self, input, output, weight = 1):
        self.input = input
        self.input.addOutput(self)

        self.output = output
        self.output.addInput(self)

        self.weight = weight

    def propogate(self):
        output = self.input.getValue() * self.weight
        return output
    
    def removeLink(self):
        self.output.removeInput(self)
        self.input.removeOutput(self)

class brain:
    def __init__(self):
        #there will always be base inputs and outputs, no matter what
        self.inputs = []
        self.outputs = []

        self.layers = None
        self.nodes = []
        self.links = []

    #inputs -----------------------------------------------------
    def addInput(self, bias = 0):
        input = self.addNode(-1, bias)
        self.inputs.append(input)
        return input
    
    def removeInput(self, input):
        #delete from input list
        self.inputs.remove(input)

        #delete node
        self.removeNode(input)

    def removeInputByIndex(self, index):
        self.removeInput(self.inputs[index])

    def setInput(self, input, value):
        input.set(value)

    def setInputByIndex(self, index, value):
        self.setInput(self.inputs[index], value)

    #outputs ------------------------------------------------------
    def addOutput(self, bias = 0):
        output = self.addNode(-2, bias)
        self.outputs.append(output)
        return output
    
    def removeOutput(self, output):
        #delete from output list
        self.outputs.remove(output)
        self.removeNode(output)

    def removeOutputbyIndex(self, index):
        self.removeInput(self.outputs[index])

    def getOutput(output):
        return output.getValue()
    
    def getOutputByIndex(self, index):
        return self.getOutput(self.outputs[index])
    
    #nodes --------------------------------------------------------
    def removeNode(self, node):
        #delete from node list
        self.nodes.remove(node)

        #remove links
        for i in node.inputs:
            i.removeLink()
            del i
        
        for i in node.outputs:
            i.removeLink()
            del i

        #remove from layers
        for i in self.layers:
            if node in i:
                i.remove(node)
        
        del node
    
    def addNode(self, layer, bias = 0):
        node = nueron(bias)
        self.nodes.append(node)

        #input or output, -1 and -2 respectively
        if layer >= 0: 
            #check if layer exists
            if layer >= len(self.layers):
                self.addLayer()
            self.layers[layer].append(node)

        return node
    
    def getNodeValue(node):
        return node.getValue()

    #layers ----------------------------------------------------------    
    def addLayer(self):
        if self.layers == None:
            self.layers = []
        self.layers.append([])

    def insertLayer(self, before):
        self.layers.insert(before, [])

    #links -----------------------------------------------------------
    def addLink(self, input, output, weight = 1):
        l = link(input, output, weight)
        self.links.append(l)
        return l

    def removeLink(self, l):
        l.removeLink()
        self.links.remove(l)
        del l

    #functioning -----------------------------------------------------------
    def process(self):
        #process all layers
        for l in self.layers:
            for n in l:
                n.readInputs()

        #now process outputs
        for o in self.outputs:
            o.readInputs()
        
