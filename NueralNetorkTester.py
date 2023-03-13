import NueralNetwork

#https://www.youtube.com/watch?v=JeVDjExBf7Y
#https://youtu.be/dPWYUELwIdM
#https://www.youtube.com/watch?v=ILsA4nyG7I0

#create a simple nueral network for testing
b = NueralNetwork.brain()

#we will try to recognize a 2x2 pixel pattern
#add inputs
#upper left pixel
ul = b.addInput()
#upper right
ur = b.addInput()
#lower left
ll = b.addInput()
#lower right
lr = b.addInput()

#add outputs
#empty
e = b.addOutput()
#full
f = b.addOutput()
#left vertical
lv = b.addOutput()
#rightvertical
rv = b.addOutput()
##up diagonal
ud = b.addOutput()
#down diagonal
dd = b.addOutput()
#lower horizontal
lh = b.addOutput()
#upper horizontal
uh = b.addOutput()

#now add layers and links
b.addLayer()
n1 = b.addNode(0)
n2 = b.addNode(0)
n3 = b.addNode(0)
n4 = b.addNode(0)
b.addLink(ul, n1)
b.addLink(ul, n3)
b.addLink(ur, n2)
b.addLink(ur, n4)
b.addLink(ll, n1)
b.addLink(ll, n3)
b.addLink(lr, n2)
b.addLink(lr, n4)

b.addLayer()
l2n1 = b.addNode(1)
l2n2 = b.addNode(1)
l2n3 = b.addNode(1)
l2n4 = b.addNode(1)
b.addLink(n1, l2n1)
b.addLink(n1, l2n2)
b.addLink(n2, l2n1)
b.addLink(n2, l2n2)
b.addLink(n3, l2n3)
b.addLink(n3, l2n4)
b.addLink(n4, l2n3)
b.addLink(n4, l2n4)

#now add links to outputs
b.addLink(l2n1, e)
b.addLink(l2n1, f)
b.addLink(l2n2, lv)
b.addLink(l2n2, rv)
b.addLink(l2n3, ud)
b.addLink(l2n3, dd)
b.addLink(l2n4, uh)
b.addLink(l2n4, lh)

#now set input
b.setInput(ul, 1)
b.setInput(ur, 1)

#now process
b.process()

print(f'e: {e.getValue()}')
print(f'f: {f.getValue()}')
print(f'lv: {lv.getValue()}')
print(f'rv: {rv.getValue()}')
print(f'ud: {ud.getValue()}')
print(f'dd: {dd.getValue()}')
print(f'lh: {lh.getValue()}')
print(f'uh: {uh.getValue()}')



