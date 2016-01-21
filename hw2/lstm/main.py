from lstm_net import *

lstm_net = LSTM_net([3,1],3)

X = np.asarray(
    np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]), 'float32'
)
Y_H = np.asarray(
    np.array( [[0],[0],[0],[0],[0],[7],[0],[0],[6] ] ),'float32'
)

Mask= np.asarray(
    np.array( [[1],[1],[1],[1],[1],[1],[1],[1],[1]]), 'float32'
)

for i in xrange(1000):
    c = lstm_net.train(X,Y_H,Mask,0.001)
    print "Cost : {0}".format(c)

Y = lstm_net.test(X,Mask)

print Y
