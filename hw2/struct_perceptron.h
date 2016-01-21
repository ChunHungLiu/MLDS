#ifndef _STRUCT_PERCEPTRON_H
#define _STRUCT_PERCEPTRON_H

#include <iostream>

using namespace std;

class StructPerceptron {
  public:
    StructPerceptron(int x_d, int y_d);
    ~StructPerceptron();

    double* get_phi(double** x, int x_len, int* y, int y_len);
    void cal_best(double** x, int x_len);
    bool train(double** x, int x_len, int* y_hat, int y_len);

    void print_w(){ 
      for(int i=0; i<phi_dim; ++i)
        cout<<w[i]<<" "; 
    }
    void set_w(int pos, double val){ w[pos] = val; }
    void print_result(){
      //cout<<"y_tilde : ";
      for(int i=0; i<y_length; ++i)
        cout<<y_tilde[i]<<" ";
      //cout<<"\n";
    }

  private:
    int idx_n_xy(int y){ return y_dim*(y_dim+2) + y*x_dim; }
    double dot(double* x, double* y){
      double ans = 0;
      for(int i=0; i<phi_dim; ++i)
        ans += x[i] * y[i];
      return ans;
    }

    int phi_dim;
    int x_dim;
    int y_dim;
    int y_length;
    int* y_tilde;
    double* w;
    double* phi_tilde;
};

#endif