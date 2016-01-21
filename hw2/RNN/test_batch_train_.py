#from RNN_class import RNN_net
#from RNN_class_GPU import RNN_net
from RNN_class_GPU_NAG import RNN_net
import numpy as np



LAYERS = [3] + [100] + [10]  
nn = RNN_net(LAYERS, batch_size=3 , momentum_type = "NAG" , act_type = "ReLU" , cost_type = "CE") 
print "start "
#  datas begin
a1 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
a2 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
a3 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
c = np.dstack((a1,a2,a3))

d = np.asarray (np.array( (c[0].T , c[1].T ,c[2].T,c[3].T,c[4].T,c[5].T,c[6].T,c[7].T,c[8].T )  )  , 'float32' )

print d[0].shape
print d.shape
'''
y1 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
y2 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
y3 = np.asarray( np.array( [[1,0,0],[3,1,0],[2,0,0],[4,1,0],[2,0,0],[1,0,1],[3,-1,0],[6,1,0],[1,0,1]]  ) , 'float32' )
y1 = np.asarray ( np.array( [[0,0],[0,0],[0,0],[0,0],[0,0],[7,7],[0,0],[0,0],[6,6] ] ) , 'float32' )
y2 = np.asarray ( np.array( [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ] ) , 'float32' )
y3 = np.asarray ( np.array( [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0] ] ) , 'float32' )
'''
y1 = np.asarray ( np.array( [[0,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0]
                             ]) , 'float32' )
y2 = np.asarray ( np.array( [[0,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0]
                             ]) , 'float32' )
y3 = np.asarray ( np.array( [[0,1,0,0,0,0,0,0,0,0],
                             [0,0,0,0,1,0,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0]
                             ]) , 'float32' )
y = np.dstack( (y1,y2,y3) )

yy = np.asarray ( np.array( (y[0].T , y[1].T ,y[2].T,y[3].T,y[4].T,y[5].T,y[6].T,y[7].T,y[8].T )  )  , 'float32' )
# datas end

print yy[0].shape
print yy.shape

mask1 = np.asarray ( np.array( [[1],[1],[1],[1],[1],[1],[1],[1],[1]]  ) , 'float32' )
mask2 = np.asarray ( np.array( [[1],[1],[1],[1],[1],[1],[1],[1],[1]]  ) , 'float32' )
mask3 = np.asarray ( np.array( [[1],[1],[1],[1],[1],[0],[0],[0],[0]]  ) , 'float32' )
mask  = np.dstack((mask1,mask2,mask3))
mmask = np.asarray ( np.array((mask[0].T , mask[1].T ,mask[2].T,mask[3].T,mask[4].T,mask[5].T,mask[6].T,mask[7].T,mask[8].T )  )  , 'float32' )

print mmask[0].shape
print mmask.shape

print "================="
print d.shape
print yy.shape
print mmask.shape
for i in xrange(1300):
    cost = nn.train(d,yy,mmask , np.cast['float32'](0.0015) , np.cast['float32'](0.9) , np.cast['float32'](5) , np.cast['float32'](0.9) )
    print "the " ,i ,"th  Cost : {0}".format(cost)
print "output = " , nn.test(d, mmask)
