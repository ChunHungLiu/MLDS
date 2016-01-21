#include "struct_perceptron.h"


StructPerceptron::StructPerceptron(int x_d, int y_d){
  x_dim = x_d;
  y_dim = y_d;
  phi_dim = y_dim*(y_dim+2) + y_dim*x_dim;
  w = new double[phi_dim];
  phi_tilde = new double[phi_dim];
  y_length = 0;
}

StructPerceptron::~StructPerceptron(){
  delete[] w;
  delete[] phi_tilde;
  delete[] y_tilde;
}

double* StructPerceptron::get_phi(double** x, int x_len, int* y, int y_len){
  y_tilde = new int[y_len];
  double* phi = new double[phi_dim];

  // start
  int i=0;
  int idx;
  for(idx=0; idx<x_dim; ++idx)
    phi[ idx_n_xy(y[i])+idx ] += x[i][idx];
  int pos_start = y_dim * y_dim;
  phi[ pos_start+y[i] ] += 1/y_len;
    
  // loop
  for(i=1; i<x_len-1; ++i){
    for(idx=0; idx<x_dim; ++idx)
      phi[ idx_n_xy(y[i])+idx ] += x[i][idx];
    phi[ y[i-1] * y_dim + y[i]] += 1/y_len;
  }
  
  // end
  i = x_len - 1;
  for(idx=0; idx<x_dim; ++idx)
      phi[ idx_n_xy(y[i])+idx ] += x[i][idx];
  int pos_end = y_dim * (y_dim+1);
  phi[ pos_end+y[i] ] += 1/y_len;

  return phi;
}

void StructPerceptron::cal_best(double** x, int x_len){
  y_tilde = new int[x_len];
  y_length = x_len;

  double** prev_phi = new double*[y_dim];
  for(int i=0; i<y_dim; ++i)
    prev_phi[i] = new double[phi_dim];

  int** backtrace = new int*[x_len];
  for(int i=0; i<x_len; ++i)
    backtrace[i] = new int[y_dim];
  
  // start
  int i=0;
  int idx;
  int y_l;
  int prev_y_l;
  for(y_l=0; y_l<y_dim; ++y_l){
    for(idx=0; idx<x_dim; ++idx)
      prev_phi[y_l][ idx_n_xy(y_l)+idx ] += x[i][idx];
    int pos_start = y_dim * y_dim;
    prev_phi[y_l][pos_start+y_l] += 1/x_len;
  }
  
  double** phi = new double*[y_dim];
  for(idx=0; idx<phi_dim; ++idx)
    phi[idx] = new double[phi_dim];

  // loop
  for(i=1; i<x_len-1; ++i){
    // copy phi from prev_phi  
    for(idx=0; idx<y_dim; ++idx)
      for(int idx2=0; idx2<phi_dim; ++idx2)
        phi[idx][idx2] = prev_phi[idx][idx2];

    // viterbi
    for(y_l=0; y_l<y_dim; ++y_l){
      double max_value;
      (*((long long*)&max_value))= ~(1LL<<52);

      for(prev_y_l=0; prev_y_l<y_dim; ++prev_y_l){
        // copy phi_i
        double* phi_i = new double[phi_dim];
        for(idx=0; idx<phi_dim; ++idx)
          phi_i[idx] = prev_phi[prev_y_l][idx];
        // modify phi_i
        for(idx=0; idx<x_dim; ++idx)
          phi_i[ idx_n_xy(y_l)+idx ] += x[i][idx];
        phi_i[ prev_y_l * y_dim + y_l] += 1/x_len;

        double value = dot(w, phi_i);
        if(max_value < value){
          max_value = value;
          backtrace[i][y_l] = prev_y_l;
          for(idx=0; idx<phi_dim; ++idx)
            phi[y_l][idx] = phi_i[idx];
        }
        delete[] phi_i;
      }
    }
    // copy phi to prev_phi
    for(idx=0; idx<y_dim; ++idx)
      for(int idx2=0; idx2<phi_dim; ++idx2)
        prev_phi[idx][idx2] = phi[idx][idx2];
  }
 
  // end
  i = x_len - 1;
  // copy phi from prev_phi  
  for(idx=0; idx<y_dim; ++idx)
    for(int idx2=0; idx2<phi_dim; ++idx2)
      phi[idx][idx2] = prev_phi[idx][idx2];

  for(y_l=0; y_l<y_dim; ++y_l){
    double max_value;
    (*((long long*)&max_value))= ~(1LL<<52);
    for(prev_y_l=0; prev_y_l<y_dim; ++prev_y_l){
      // copy phi_i
      double* phi_i = new double[phi_dim];
      for(idx=0; idx<phi_dim; ++idx)
        phi_i[idx] = prev_phi[prev_y_l][idx];
      // modify phi_i
      for(idx=0; idx<x_dim; ++idx)
        phi_i[ idx_n_xy(y_l)+idx ] += x[i][idx];
      phi_i[ prev_y_l * y_dim + y_l] += 1/x_len;
      int pos_end = y_dim * (y_dim+1);
      phi_i[ pos_end + y_l] += 1/x_len;

      double value = dot(w, phi_i);
      if(max_value < value){
        max_value = value;
        backtrace[i][y_l] = prev_y_l;
        for(idx=0; idx<phi_dim; ++idx)
          phi[y_l][idx] = phi_i[idx];
      }
      delete[] phi_i;
    }
  } 

  // compare all phi.
  double max_f;
  int max_id;
  (*((long long*)&max_f))= ~(1LL<<52);
  for(idx=0; idx<y_dim; ++idx){
    double f = dot(w, phi[idx]);
    if(max_f < f){
      max_f = f;
      max_id = idx;
    }
  }

  for(idx=0; idx<phi_dim; ++idx)
    phi_tilde[idx] = phi[max_id][idx];

  y_tilde[x_len-1] = max_id;
  for(idx=x_len-2; idx>=0; --idx){
    prev_y_l = backtrace[idx+1][y_tilde[idx+1]];
    y_tilde[idx] = prev_y_l;
  }

  for(int i=0; i<y_dim; ++i)
    delete[] prev_phi[i];
  delete[] prev_phi;

  /*
  for(int i=0; i<x_len; ++i)
    delete[] backtrace[i];
  delete[] backtrace;
  

  for(idx=0; idx<phi_dim; ++idx)
    delete[] phi[idx];
  delete[] phi;
  */

  return;
}

bool StructPerceptron::train(double** x, int x_len, int* y_hat, int y_len){
  cal_best(x, x_len);
  
  bool mistake = false;

  int* pruned_tilde = new int[y_len];
  int* pruned_hat = new int[y_len];
  int pruned_tilde_len = 0;
  int pruned_hat_len = 0;
  int prev_tilde = y_len;
  int prev_hat   = y_len;

  for(int idx=0; idx<y_len; ++idx){
    if(prev_tilde != y_tilde[idx]){
      pruned_tilde[pruned_tilde_len] = y_tilde[idx];
      pruned_tilde_len++;
    }
    if(prev_hat != y_hat[idx]){
      pruned_hat[pruned_hat_len] = y_hat[idx];
      pruned_hat_len++;
    }

    prev_tilde = y_tilde[idx];
    prev_hat   = y_hat[idx];
  }

  if(pruned_tilde_len == pruned_hat_len){
    for(int idx=0; idx<pruned_hat_len; ++idx){
      if(pruned_tilde[idx] != pruned_hat[idx]){
        mistake = true;
        break;
      }
    }  
  } else { mistake = true; }
  
  if(mistake){
    double* phi_hat = get_phi(x, x_len, y_hat, y_len);
    for(int idx=0; idx<phi_dim; ++idx)
      w[idx] += phi_hat[idx] - phi_tilde[idx];

    delete[] phi_hat;
    return true;
  } 
  return false;
}