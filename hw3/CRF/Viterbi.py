import numpy as np
import sys
from loss import loss
from  math import exp


class Viterbi:
    def __init__ (self , x , w , y_class , y_hat , print_flag):
        self.w = w 
        self.y_class = y_class 
        self.x = np.asarray(x) 
        self.SEQ_LENGTH = len(x)
        self.PHONE_Length = len(x[0])
        self.w_o = w[ : y_class * self.PHONE_Length ] 
        self.w_t = w[ y_class * self.PHONE_Length : ]
        self.w_o = self.w_o.reshape( y_class , self.PHONE_Length  )
        self.w_t = self.w_t.reshape( y_class+1 , y_class+1 ) 
        self.Prob = np.zeros( ( self.SEQ_LENGTH + 2 , y_class ))
        self.Path = np.zeros( ( self.SEQ_LENGTH + 1 , y_class ))
        self.y_tilde = np.zeros(self.SEQ_LENGTH)
        
        self.sigma_t = np.zeros(self.w_t.shape)
        self.sigma_o = np.zeros(self.w_o.shape)
        self.prob_sum = 0.1 
        self.y_hat   = y_hat 
        self.print_flag = print_flag

        self.CRF_Prob = np.zeros(self.Prob.shape)
        self.CRF_Psi  = np.zeros( (self.SEQ_LENGTH+2 , y_class , len(w) ))

    def compute_max_P ( self ,layers  , y_possible  , mode = "None" ):
        #print self.Prob[layers]
        if mode == "Start" :
            Prob =                        np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t[self.y_class][y_possible] #/ self.SEQ_LENGTH
        elif mode == "End" :
            Prob =  self.Prob[layers] + self.w_t.T[self.y_class][:-1]
        else:
            #print self.w_t.shape
            #print self.w_t.T[y_possible][:-1].shape
            #print self.Prob[layers].shape
            Prob =  self.Prob[layers] + np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t.T[y_possible][:-1]#/self.SEQ_LENGTH
        '''
        if mode == "Start" : 
            max_Prob =                        np.dot(self.w_o[y_possible] , self.x[layers]/self.SEQ_LENGTH) + 10*self.w_t[self.y_class][y_possible]/self.SEQ_LENGTH
            #max_Prob =                        np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t[self.y_class][y_possible]
        elif mode == "End" : 
            max_Prob = self.Prob[layers][0] +                                                 10*self.w_t[0][self.y_class] / self.SEQ_LENGTH
            #max_Prob = self.Prob[layers][0] +                                                 self.w_t[0][self.y_class] 
        else:
            max_Prob = self.Prob[layers][0] + np.dot(self.w_o[y_possible] , self.x[layers]/self.SEQ_LENGTH) + 1*self.w_t[0][y_possible] / self.SEQ_LENGTH
            #max_Prob = self.Prob[layers][0] + np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t[0][y_possible] 
        y_path = 0 
        for y_last in range(1,self.y_class):
            if mode == "Start" : 
                Prob =                        np.dot(self.w_o[y_possible] , self.x[layers]/self.SEQ_LENGTH) + 10*self.w_t[self.y_class][y_possible] / self.SEQ_LENGTH
                #Prob =                        np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t[self.y_class][y_possible] 
            elif mode == "End" : 
                Prob = self.Prob[layers][y_last] +                                               10*  self.w_t[y_last][self.y_class] / self.SEQ_LENGTH
                #Prob = self.Prob[layers][y_last] +                                                 self.w_t[y_last][self.y_class] 
            else:
                Prob = self.Prob[layers][y_last] + np.dot(self.w_o[y_possible] , self.x[layers]/self.SEQ_LENGTH) + 1*self.w_t[y_last][y_possible]/self.SEQ_LENGTH
                #Prob = self.Prob[layers][y_last] + np.dot(self.w_o[y_possible] , self.x[layers]) + self.w_t[y_last][y_possible]
        
            if Prob > max_Prob:
                max_Prob = Prob 
                y_path = y_last 
        '''
        self.Path[layers][y_possible] = np.argmax(Prob) 
        self.Prob[layers+1][y_possible] = np.max(Prob)
        #self.Path[layers][y_possible] = y_path 
        #self.Prob[layers+1][y_possible] = max_Prob

    def gen_path ( self , length ):
        if length == 0: 
            return 
        elif length == self.SEQ_LENGTH:
            self.y_tilde = [ self.Path[ length ][ 0 ] ]
            #print self.y_tilde
        else:
            self.y_tilde += [ self.Path[ length ][self.y_tilde[-1]]  ]
            #print self.y_tilde
        self.gen_path( length-1 ) 

    def main_Viterbi (self):
        # start part 
        for y_possible in range(self.y_class):
            self.compute_max_P(  0 , y_possible , "Start")
        # layers 
        for x_idx in range(1,self.SEQ_LENGTH):
            for y_possible in range(self.y_class):
                self.compute_max_P(  x_idx , y_possible , "None" )
        # end 
        self.compute_max_P( self.SEQ_LENGTH  , 0  , "End" )

        #generate path 
        self.gen_path( self.SEQ_LENGTH )
        
        #print self.y_tilde[::-1]
        
        return self.y_tilde[::-1]

    def update_Viterbi (self):
        # start part 
        for y_possible in range(self.y_class):
            self.compute_sig_PPsi(  0 , y_possible , "Start")
        # layers 
        for x_idx in range(1,self.SEQ_LENGTH):
            print  x_idx 
            #print " layers " , x_idx 
            for y_possible in range(self.y_class):
                self.compute_sig_PPsi(  x_idx , y_possible , "None" )
        # end 
        self.compute_sig_PPsi( self.SEQ_LENGTH , 0  , "End" )

    def compute_sig_PPsi ( self ,layers  , y_possible  , mode = "None" ):
        #print self.Prob[layers]
        update_Psi_o = np.zeros(self.w_o.shape)
        update_Psi_t = np.zeros(self.w_t.shape)
        update_Psi   = np.zeros(self.w.shape)
        CRF_Prob = 0
        if mode == "Start" :
            log_current_Prob = ( np.dot(self.w_o[y_possible] , self.x[layers]/(self.SEQ_LENGTH) ) + self.w_t[self.y_class][y_possible]/self.SEQ_LENGTH )  
            update_Psi_o[y_possible]               = (log_current_Prob) + np.log(self.x[layers])- np.log(self.SEQ_LENGTH) 
            update_Psi_t[self.y_class][y_possible] = (log_current_Prob) - np.log(self.SEQ_LENGTH)
            CRF_Prob = log_current_Prob
        elif mode == "End" :
            log_current_Prob           = ( self.w_t.T[self.y_class][:-1] )  
            update_Psi                 = log_current_Prob + self.CRF_Psi[layers][:] 
            update_Psi_t.T[self.y_class] = (log_current_Prob) - (self.SEQ_LENGTH) 
            CRF_Prob                   =  log_current_Prob + (self.CRF_Prob[layers][:]) 
        else:
            log_current_Prob = ( np.dot(self.w_o[y_possible] , self.x[layers]/(self.SEQ_LENGTH) ) + self.w_t.T[y_possible][:-1])  
            update_Psi                     = log_current_Prob + self.CRF_Psi[layers][:] 
            update_Psi_t.T[y_possible][:]  = (log_current_Prob) - np.log(self.SEQ_LENGTH) 
            update_Psi_o[y_possible]       = log_current_Prob + np.log(self.x[layers]) - np.log(self.SEQ_LENGTH) 
            CRF_Prob                         = log_current_Prob + self.CRF_Prob[layers][:]  
        '''
        for y_last in range(0,self.y_class):
            if mode == "Start" :
                log_current_Prob = ( np.dot(self.w_o[y_possible] , self.x[layers]/(self.SEQ_LENGTH) ) + self.w_t[self.y_class][y_possible]/self.SEQ_LENGTH )  
                update_Psi_o[y_possible]               = (log_current_Prob) + np.log(self.x[layers])- np.log(self.SEQ_LENGTH) 
                update_Psi_t[self.y_class][y_possible] = (log_current_Prob) - np.log(self.SEQ_LENGTH)

                CRF_Prob = log_current_Prob
            elif mode == "End" :
                log_current_Prob = ( self.w_t[y_last][self.y_class]/self.SEQ_LENGTH )  
                update_Psi                       = np.logaddexp ( update_Psi ,  log_current_Prob + self.CRF_Psi[layers][y_last] )
                update_Psi_t[y_last][y_possible] = np.logaddexp( update_Psi_t[y_last][y_possible] , (log_current_Prob) - (self.SEQ_LENGTH) )

                CRF_Prob                         = np.logaddexp( CRF_Prob   ,  log_current_Prob + (self.CRF_Prob[layers][y_last]) )
            else:
                log_current_Prob = ( np.dot(self.w_o[y_possible] , self.x[layers]/(self.SEQ_LENGTH) ) + self.w_t[y_last][y_possible]/self.SEQ_LENGTH )  
                update_Psi                       = np.logaddexp ( update_Psi ,  log_current_Prob + self.CRF_Psi[layers][y_last]) 
                update_Psi_t[y_last][y_possible] = np.logaddexp ( update_Psi_t[y_last][y_possible] , (log_current_Prob) - np.log(self.SEQ_LENGTH) )
                update_Psi_o[y_possible]         = np.logaddexp ( update_Psi_o[y_possible]  ,  log_current_Prob + np.log(self.x[layers]) - np.log(self.SEQ_LENGTH) )

                CRF_Prob                         = np.logaddexp( CRF_Prob , log_current_Prob + self.CRF_Prob[layers][y_last] ) 
        '''
        update_Psi = np.hstack( ( np.hstack(update_Psi_o) , np.hstack(update_Psi_t)    ) )
        if mode != "Start":
            a = update_Psi[0]
            b = CRF_Prob[0]
            for idx in range(1 , self.y_class):
                a = np.logaddexp( a , update_Psi[idx])
                b = np.logaddexp( b , CRD_Prob[idx])
        else:
            a = update_Psi
            b = CRF_Prob
            
        self.CRF_Psi[layers+1][y_possible]  =  a #update_Psi
        #print "update_Psi = " , update_Psi
        self.CRF_Prob[layers+1][y_possible] = b #CRF_Prob
        #print "update_Prob  " , layers+1 , ", ", y_possible , " = " , CRF_Prob #self.CRF_Prob[layers+1][y_possible]
