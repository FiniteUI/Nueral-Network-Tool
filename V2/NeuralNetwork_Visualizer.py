import tkinter
import time

class visualizer:
    def __init__(self, n, length, width, border=20, node_size = 50, line_scale = 2):
        self.nueralNetwork = n
        self.length = length
        self.width = width
        self.border = border
        self.node_size = node_size
        self.line_scale = line_scale

        self.initializeWindow()

    def initializeWindow(self):
        window = tkinter.Tk()
        self.window = window

        window.title('Nueral Network Visualizer')
        window.geometry(f'{self.width}x{self.length}')

        canvas = tkinter.Canvas(height = self.length, width = self.width, bg = "white")
        self.canvas = canvas
        canvas.pack()

        window.update()

    def close(self):
        self.window.destroy()
        self.window = None
        self.canvas = None
        
    def hideWindow(self):
        self.window.withdraw()

    def showWindow(self):
        self.window.deiconify()

    def drawNetwork(self, showWeights = True, showValues = True):
        #add something to dynamically choose node size based of canvas size maybe
        if self.window == None:
            self.initializeWindow()
        else:
            self.canvas.delete('all')

        columns = len(self.nueralNetwork.layers)
        columnDistance = (self.width - (self.border * 2)) / columns
        
        #first draw nodes
        nodes = {}
        for l in range(columns):
            rows = len(self.nueralNetwork.layers[l])
            rowDistance = (self.length - (self.border * 2)) / rows

            for n in range(rows):
                x = l * columnDistance + (columnDistance / 2)
                y = n * rowDistance + (rowDistance / 2)
                
                #if l % 2 == 0:
                #    y -= 50

                #maybe change this to minimize distance/crossover
                x1 = x + self.node_size
                y1 = y + self.node_size
                node = self.nueralNetwork.layers[l][n]
                nodeImage = self.canvas.create_oval(x, y, x1, y1, fill = '#1BC2A0')
                #nodes[f'{node.id}'] = nodeImage

                #nodes[node.id] = nodeImage
                nodes[node] = nodeImage

                #write value, bias
                text = ''
                if showValues:
                    text += f'v: {round(self.nueralNetwork.layers[l][n].getValue(), 2)}'

                if text != '':
                    textX = (x + x1) / 2
                    textY = (y + y1) / 2
                    self.canvas.create_text(textX, textY, text = text)

                #if node has a label, label it
                if self.nueralNetwork.layers[l][n].label != None:
                    textX = (x + x1) / 2
                    textY = y - (self.node_size / 5)
                    self.canvas.create_text(textX, textY, text = self.nueralNetwork.layers[l][n].label)

                #now draw inputs to this node
                for i in range(len(self.nueralNetwork.layers[l][n].inputs)):
                    #origin will be previous node
                    if l != 0:
                        #first check weight
                        weight = self.nueralNetwork.layers[l][n].inputs[i][1]
                        width = abs(weight) * self.line_scale

                        if weight < 0:
                            fill = 'red'
                        elif weight > 0:
                            fill = 'blue'
                        else:
                            fill = 'gray'

                        previousNode = self.nueralNetwork.layers[l][n].inputs[i][0]
                        #origin = self.canvas.coords(nodes[previousNode.id])
                        origin = self.canvas.coords(nodes[previousNode])

                        #calculate center
                        centerX = (origin[0] + origin[2]) / 2
                        centerY = (origin[1] + origin[3]) / 2

                        #calculate center of destination
                        destCenterX = (x + x1) / 2
                        destCenterY = (y + y1) / 2

                        #print(f'Drawing link from node {previousNode}, {previousNode.label}, X={centerX}Y={centerY}to node {self.nueralNetwork.layers[l][n]}, {self.nueralNetwork.layers[l][n].label}, X1={destCenterX}Y1={destCenterY} with weight {weight} ...')

                        #now create line
                        line = self.canvas.create_line(centerX, centerY, destCenterX, destCenterY, width = width, fill = fill)
                        self.canvas.tag_lower(line)

                        #now write weight
                        if showWeights:
                            weightX = centerX + (destCenterX - centerX) / 5 
                            weightY = centerY + (destCenterY - centerY) / 5 
                            self.canvas.create_text(weightX, weightY, text = weight, font = ('Times', '16'))

        #links are lines
        self.window.update()

    def _click(self, event):
        self.close()

    def waitUntilClick(self):
        self.canvas.bind('<Button-1>', self._click)
        self.window.mainloop()

    def waitForSeconds(self, seconds):
        time.sleep(seconds)
        self.close()

        