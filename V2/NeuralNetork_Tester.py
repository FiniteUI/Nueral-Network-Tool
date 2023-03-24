import NeuralNetwork
import NeuralNetwork_Visualizer
import os
import time
import datetime
import NueralNetwork_ActivationFunctions

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

#now add layers and links
nodes.append([])
b.addLayer()
nodes[len(nodes) - 1].append(b.addNode(1, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[0][0], nodes[len(nodes) - 1][0], 1)
b.addLink(nodes[0][3], nodes[len(nodes) - 1][0], 1)
nodes[len(nodes) - 1].append(b.addNode(1, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[0][1], nodes[len(nodes) - 1][1], 1)
b.addLink(nodes[0][2], nodes[len(nodes) - 1][1], 1)
nodes[len(nodes) - 1].append(b.addNode(1, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[0][0], nodes[len(nodes) - 1][2], 1)
b.addLink(nodes[0][3], nodes[len(nodes) - 1][2], -1)
nodes[len(nodes) - 1].append(b.addNode(1, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[0][1], nodes[len(nodes) - 1][3], 1)
b.addLink(nodes[0][2], nodes[len(nodes) - 1][3], -1)

nodes.append([])
b.addLayer()
nodes[len(nodes) - 1].append(b.addNode(2, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[1][0], nodes[len(nodes) - 1][0], 1)
b.addLink(nodes[1][1], nodes[len(nodes) - 1][0], 1)
nodes[len(nodes) - 1].append(b.addNode(2, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[1][0], nodes[len(nodes) - 1][1], -1)
b.addLink(nodes[1][1], nodes[len(nodes) - 1][1], 1)
nodes[len(nodes) - 1].append(b.addNode(2, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[1][2], nodes[len(nodes) - 1][2], 1)
b.addLink(nodes[1][3], nodes[len(nodes) - 1][2], -1)
nodes[len(nodes) - 1].append(b.addNode(2, activation = NueralNetwork_ActivationFunctions.hyperbolicTangent))
b.addLink(nodes[1][2], nodes[len(nodes) - 1][3], 1)
b.addLink(nodes[1][3], nodes[len(nodes) - 1][3], 1)

#add outputs
nodes.append([])

#empty
nodes[len(nodes) - 1].append(b.addOutput(label = "Empty", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][0], nodes[len(nodes) - 1][0], 1)
#full
nodes[len(nodes) - 1].append(b.addOutput(label = "Full", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][0], nodes[len(nodes) - 1][1], -1)
#left vertical
nodes[len(nodes) - 1].append(b.addOutput(label = "Left Vertical", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][1], nodes[len(nodes) - 1][2], 1)
#rightvertical
nodes[len(nodes) - 1].append(b.addOutput(label = "Right Vertical", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][1], nodes[len(nodes) - 1][3], -1)
##up diagonal
nodes[len(nodes) - 1].append(b.addOutput(label = "Up Diagonal", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][2], nodes[len(nodes) - 1][4], 1)
#down diagonal
nodes[len(nodes) - 1].append(b.addOutput(label = "Down Diagonal", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][2], nodes[len(nodes) - 1][5], -1)
#lower horizontal
nodes[len(nodes) - 1].append(b.addOutput(label = "Lower Horizontal", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][3], nodes[len(nodes) - 1][6], 1)
#upper horizontal
nodes[len(nodes) - 1].append(b.addOutput(label = "Upper Horizontal", activation = NueralNetwork_ActivationFunctions.rectifiedLinearUnits))
b.addLink(nodes[2][3], nodes[len(nodes) - 1][7], -1)

#path = os.path.realpath(os.path.dirname(__file__))
#path = os.path.join(path, 'Testing', 'NueralNetworkTester', 'Data')
#name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#b.saveStructure(path, name)

#now set input
#b = NueralNetwork.brain.loadStructure('E:/#A - PROGRAMMING PROJECTS/Nueral Networks/Testing/NueralNetworkTester/Data/20230315231446.ns')
b.setInputByIndex(0, 1)
b.setInputByIndex(1, -1)
b.setInputByIndex(2, -1)
b.setInputByIndex(3, 1)

v = NeuralNetwork_Visualizer.visualizer(b, 600, 1000, node_size = 50, line_scale = 4)
v.drawNetwork(showWeights = True)
v.waitUntilClick()

#test modifications
#b.removeNode(testNode)
#b.removeLinkByIndex(5)
#b.updateLinkByIndex(5, 1.3)
#b.removeLayer(2)
#b.replaceLinkWithNode(5, NueralNetwork_ActivationFunctions.hyperbolicTangent)
#v.drawNetwork(showWeights = True)
#v.waitUntilClick()

#now process
b.process()
v.drawNetwork(showWeights = True)
v.waitUntilClick()

for i in nodes[len(nodes) - 1]:
    print(f'Label: {i.label}, Value: {i.getValue()}')



