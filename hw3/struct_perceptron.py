import pdb

import numpy as np


class StructPerceptron:
    def __init__(self, x_len, y_len):
        phi_len = y_len*(y_len+2) + y_len*x_len
        self.w = np.zeros( phi_len )
        self.phi_len = phi_len
        self.x_len = x_len
        self.y_len = y_len

    def idx_n_xy(self, y):
        return self.y_len*(self.y_len+2) + y*self.x_len

    def phi(self, x, y):
        phi = np.zeros((self.phi_len))
        #start
        i=0
        phi[ self.idx_n_xy(y[i]):self.idx_n_xy(y[i])+self.x_len ] += x[i]
        pos_start = self.y_len*self.y_len
        phi[ pos_start+y[i] ] += 1/len(x)
        # loop
        for i in xrange(1,len(x)-1):
          phi[ self.idx_n_xy(y[i]):self.idx_n_xy(y[i])+self.x_len ] += x[i]
          phi[ y[i-1]*self.y_len + y[i]] += 1/len(x)
        # end
        i=len(x)-1
        phi[ self.idx_n_xy(y[i]):self.idx_n_xy(y[i])+self.x_len ] += x[i]
        pos_end = self.y_len*(self.y_len+1)
        phi[ pos_end+y[i] ] += 1/len(x)

        return phi

    def get_best(self, x):
        prev_phi = np.zeros( (self.y_len, self.phi_len) )
        backtrace = np.zeros( (len(x), self.y_len),dtype=int )
        # start
        i=0
        for y_l in xrange(self.y_len):
          #idx_n_xy = self.y_len*(self.y_len+2) + y_l*self.x_len
          #prev_phi[y_l, idx_n_xy:idx_n_xy + self.x_len] += x[i]
          prev_phi[y_l, self.idx_n_xy(y_l):self.idx_n_xy(y_l) + self.x_len] += x[i]
          pos_start = self.y_len*self.y_len
          prev_phi[y_l, pos_start+y_l] += 1/len(x)
        # loop
        for i in xrange(1,len(x)-1):
          phi = np.empty_like(prev_phi)
          np.copyto(phi, prev_phi)
          for y_l in xrange(self.y_len):        
            max_value = -float('inf')
            for prev_y_l in xrange(self.y_len):
              phi_i = np.empty_like(prev_phi[prev_y_l])
              np.copyto(phi_i, prev_phi[prev_y_l])
              #idx_n_xy = self.y_len*(self.y_len+2) + y_l*self.x_len
              #phi_i[ idx_n_xy:idx_n_xy+self.x_len ] += x[i]
              phi_i[ self.idx_n_xy(y_l):self.idx_n_xy(y_l)+self.x_len ] += x[i]
              phi_i[ prev_y_l*self.y_len + y_l] += 1/len(x)

              value = np.dot(self.w, phi_i)
              if(max_value < value):
                max_value = value
                backtrace[i, y_l] = prev_y_l
                phi[y_l,:] = phi_i
          np.copyto(prev_phi, phi)
        # end
        i=len(x)-1
        phi = np.empty_like(prev_phi)
        np.copyto(phi, prev_phi)
        for y_l in xrange(self.y_len):
          max_value = -float('inf')
          for prev_y_l in xrange(self.y_len):
            phi_i = np.empty_like(prev_phi[prev_y_l])
            np.copyto(phi_i, prev_phi[prev_y_l])
            #idx_n_xy = self.y_len*(self.y_len+2) + y_l*self.x_len
            #phi_i[ idx_n_xy:idx_n_xy+self.x_len ] += x[i]
            phi_i[ self.idx_n_xy(y_l):self.idx_n_xy(y_l)+self.x_len ] += x[i]
            phi_i[ prev_y_l*self.y_len + y_l] += 1/len(x)
            pos_end = self.y_len*(self.y_len+1)
            phi_i[ pos_end + y_l] += 1/len(x)

            value = np.dot(self.w, phi_i)
            if(max_value < value):
              max_value = value
              backtrace[i, y_l] = prev_y_l
              phi[y_l,:] = phi_i
        # compare all phi.
        best_value = np.argmax(np.dot(phi,self.w))
        best_phi = phi[best_value]
        best_y = [ best_value ]
        for b in reversed( xrange(1,len(x))):
            prev_y_l = backtrace[b, best_y[0]]
            best_y.insert(0, prev_y_l)

        return (best_y, best_phi)

    def train(self, x, y_hat):
        y_tilde, phi_tilde = self.get_best(x)

        pruned_y_tilde = []
        pruned_y_hat = []
        prev_tilde, prev_hat = None, None
        for l in xrange(len(y_tilde)):
            if prev_tilde != y_tilde[l]:
                pruned_y_tilde.append( y_tilde[l] )
            if prev_hat != y_hat[l]:
                pruned_y_hat.append( y_hat[l] )
            prev_tilde = y_tilde[l]
            prev_hat   = y_hat[l]

        if pruned_y_tilde != pruned_y_hat:
          phi_hat = self.phi(x,y_hat)
          self.w += phi_hat - phi_tilde
          return True
          '''
          if np.sum(phi_hat != phi_tilde) > 0:
            if y_tilde == y_hat.tolist():
              print "y_tilde == y_hat"
              pdb.set_trace()
            self.w += phi_hat - phi_tilde
            return True
          '''
        else:
          return False

