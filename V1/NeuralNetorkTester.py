import NeuralNetwork
import NeuralNetworkVisualizer
import os
import time
import datetime

#https://www.youtube.com/watch?v=JeVDjExBf7Y
#https://youtu.be/dPWYUELwIdM
#https://www.youtube.com/watch?v=ILsA4nyG7I0

#create a simple nueral network for testing
b = NeuralNetwork.brain()
nodes = []

#we will try to recognize a 2x2 pixel pattern
#add inputs
#upper left pixel
nodes.append([])
nodes[len(nodes) - 1].append(b.addInput(label='Upper Left'))
#upper right
nodes[len(nodes) - 1].append(b.addInput(label='Upper Right'))
#lower right
nodes[len(nodes) - 1].append(b.addInput(label='Lower Right'))
#lower left
nodes[len(nodes) - 1].append(b.addInput(label='Lower Left'))

bias = 0

#now add layers and links
nodes.append([])
b.addLayer()
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [1, 0, 0, 1], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [0, 1, 1, 0], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [1, 0, 0, -1], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [0, 1, -1, 0], activation = NeuralNetwork.node.hyperbolicTangent))

nodes.append([])
b.addLayer()
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [1, 1, 0, 0], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [-1, 1, 0, 0], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [0, 0, 1, -1], activation = NeuralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [0, 0, 1, 1], activation = NeuralNetwork.node.hyperbolicTangent))

#add outputs
nodes.append([])
#empty
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [1, 0, 0, 0], label = "Empty", activation = NeuralNetwork.node.rectifiedLinearUnits))
#full
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [-1, 0, 0, 0], label = "Full", activation = NeuralNetwork.node.rectifiedLinearUnits))
#left vertical
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 1, 0, 0], label = "Left Vertical", activation = NeuralNetwork.node.rectifiedLinearUnits))
#rightvertical
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, -1, 0, 0], label = "Right Vertical", activation = NeuralNetwork.node.rectifiedLinearUnits))
##up diagonal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 1, 0], label = "Up Diagonal", activation = NeuralNetwork.node.rectifiedLinearUnits))
#down diagonal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, -1, 0], label = "Down Diagonal", activation = NeuralNetwork.node.rectifiedLinearUnits))
#lower horizontal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 0, 1], label = "Lower Horizontal", activation = NeuralNetwork.node.rectifiedLinearUnits))
#upper horizontal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 0, -1], label = "Upper Horizontal", activation = NeuralNetwork.node.rectifiedLinearUnits))

#path = os.path.realpath(os.path.dirname(__file__))
#path = os.path.join(path, 'Testing', 'NueralNetworkTester', 'Data')
#name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#b.saveStructure(path, name)

#now set input
b.setInput(nodes[0][0], 1)
b.setInput(nodes[0][1], 1)
b.setInput(nodes[0][2], -1)
b.setInput(nodes[0][3], -1)

#b = NueralNetwork.brain.loadStructure('E:/#A - PROGRAMMING PROJECTS/Nueral Networks/Testing/NueralNetworkTester/Data/20230315231446.ns')
b.setInputByIndex(0, 1)
b.setInputByIndex(1, 1)
b.setInputByIndex(2, -1)
b.setInputByIndex(3, -1)

v = NeuralNetworkVisualizer.visualizer(b, 600, 1000, node_size = 50, line_scale = 4)
v.drawNetwork(showWeights = False, showBias = False, showUnweightedConnections = True)
v.waitUntilClick()

#now process
b.process()
v.drawNetwork(showWeights = False, showBias = False)
v.waitUntilClick()



