import NueralNetwork
import NueralNetworkVisualizer

#https://www.youtube.com/watch?v=JeVDjExBf7Y
#https://youtu.be/dPWYUELwIdM
#https://www.youtube.com/watch?v=ILsA4nyG7I0

#create a simple nueral network for testing
b = NueralNetwork.brain()
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
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [1, 0, 0, 1], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [0, 1, 1, 0], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [1, 0, 0, -1], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(1, bias = bias, weightsIn = [0, 1, -1, 0], activation = NueralNetwork.node.hyperbolicTangent))

nodes.append([])
b.addLayer()
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [1, 1, 0, 0], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [-1, 1, 0, 0], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [0, 0, 1, -1], activation = NueralNetwork.node.hyperbolicTangent))
nodes[len(nodes) - 1].append(b.addNode(2, bias = bias, weightsIn = [0, 0, 1, 1], activation = NueralNetwork.node.hyperbolicTangent))

#add outputs
nodes.append([])
#empty
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [1, 0, 0, 0], label = "Empty", activation = NueralNetwork.node.rectifiedLinearUnits))
#full
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [-1, 0, 0, 0], label = "Full", activation = NueralNetwork.node.rectifiedLinearUnits))
#left vertical
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 1, 0, 0], label = "Left Vertical", activation = NueralNetwork.node.rectifiedLinearUnits))
#rightvertical
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, -1, 0, 0], label = "Right Vertical", activation = NueralNetwork.node.rectifiedLinearUnits))
##up diagonal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 1, 0], label = "Up Diagonal", activation = NueralNetwork.node.rectifiedLinearUnits))
#down diagonal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, -1, 0], label = "Down Diagonal", activation = NueralNetwork.node.rectifiedLinearUnits))
#lower horizontal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 0, 1], label = "Lower Horizontal", activation = NueralNetwork.node.rectifiedLinearUnits))
#upper horizontal
nodes[len(nodes) - 1].append(b.addOutput(bias = bias, weightsIn = [0, 0, 0, -1], label = "Upper Horizontal", activation = NueralNetwork.node.rectifiedLinearUnits))

#now set input
b.setInput(nodes[0][0], 1)
b.setInput(nodes[0][1], 1)
b.setInput(nodes[0][2], -1)
b.setInput(nodes[0][3], -1)

v = NueralNetworkVisualizer.visualizer(b, 600, 1000, node_size = 50, line_scale = 4)
v.drawNetwork()
v.waitUntilClick()

#now process
b.process()
v.drawNetwork()
v.waitUntilClick()



