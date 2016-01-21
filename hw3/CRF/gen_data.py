import numpy as np


def gen_data ():
    
    length = 5 #np.random.randint(5,6)  
    x_seq  = np.random.uniform( 0 ,1 , size=( length , 48 )   )
    y_seq = np.argmax ( x_seq , axis=1  )
    #y_seq = abs(y_seq)%48
    #print "this is y "
    #print y_seq

    #x_seq = np.array( [ [-2,-8,-9] , [-2,3,0] ] )
    #y_seq = np.array( [    9  ,  1  ] )

    return x_seq , y_seq  


gen_data() 
