#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string.h>

#include "struct_perceptron.h"

using namespace std;

int main(int argc, char* argv[]){
  /* argv
   * 1 ~ 2                              : x_dim, y_dim
   * 3 ~ 2+phi_dim                      : w
   * 3+phi_dim                          : x_len
   * 4+phi_dim ~ 3+phi_dim+x_len*x_dim  : x
   * 4+phi_dim+x_len*x_dim ~ 
   * 3+phi_dim+x_len*x_dim + y_len      : y_hat
   */
  
  int x_dim;
  int y_dim;
  int phi_dim;
  int x_len;
  int y_len;

  ifstream in_file;
  in_file.open(argv[1]);

  //in_file.seekg (0, in_file.end);
  //int length = in_file.tellg() / ;
  //in_file.seekg (0, in_file.beg);
  int length = 65536;

  char* str = new char[length];

  // x_dim 
  in_file.getline(str, length);
  x_dim = atoi(str);
  //cout<<"x_dim : "<<x_dim<<'\n';
  
  //y_dim
  in_file.getline(str, length);
  y_dim = atoi(str);
  //cout<<"y_dim : "<<y_dim<<'\n';

  StructPerceptron sp = StructPerceptron(x_dim, y_dim);

  //x_dim=0;y_dim=0;

  // w
  phi_dim = y_dim*(y_dim+2) + y_dim*x_dim;
  int i;
  for(i=0; i<phi_dim-1; ++i){
    in_file.get(str, length, ' ');
    sp.set_w(i, atof(str));
    //cout<<atof(str);
    in_file.ignore(1);
  }
  i = phi_dim-1;
  in_file.getline(str, length);
  sp.set_w(i, atof(str));
  //cout<<atof(str)<<'\n';

  // x_len
  in_file.getline(str, length);
  x_len = atoi(str);
  //cout<<"x_len : "<<x_len<<'\n';
  
  // x
  double** x = new double*[x_len];
  for(int i=0; i<x_len; ++i)
    x[i] = new double[x_dim];

  for(int i=0; i<x_len; ++i){
    for(int j=0; j<x_dim; ++j){
      if(i==x_len-1 && j==x_dim-1){
        in_file.get(str, length, '\n');
        
      }else{
        in_file.get(str, length, ' ');
      }
      //if(i==0 && j==0)
      //  cout<<"first x "<<atof(str)<<'\n';
      x[i][j] = atof(str);
      in_file.ignore(1);
      //cout<<"i = "<<i<<" j = "<<j<<'\n';
    }
  }

  //cout<<"Reading y";
  // y
  y_len = x_len;
  int* y_hat = new int[y_len];
  for(int i=0; i<y_len-1; ++i){
    in_file.get(str, length, ' ');
    //if(i==6)
    //  cout<<"y_hat :"<<atoi(str)<<'\n';
    y_hat[i] = atoi(str);
    in_file.ignore(1);
  }
  in_file.getline(str, length);
  y_hat[y_len-1] = atoi(str);
  in_file.close();

  if (sp.train(x, x_len, y_hat, y_len)){
    cout<<"1 ";
    sp.print_w();
  } else 
    cout<<"0";
  
  return 0;
  /*
  double new_w[14] = {-0.35667494, -1.2039728 , -0.91629073,
                   -0.51082562, -0.69314718, -0.69314718, 
                   -0.69314718, -0.69314718, -1.60943791, 
                   -0.91629073, -0.91629073, -0.69314718, 
                   -0.91629073, -2.30258509};
  for(int i=0;i<14;++i)
    sp.set_w(i,new_w[i]);

  double** x = new double*[3];
  for(int i=0; i<3; ++i)
    x[i] = new double[3];
  x[0][0] = 0;
  x[0][1] = 0;
  x[0][2] = 1;
  x[1][0] = 1;
  x[1][1] = 0;
  x[1][2] = 0;
  x[2][0] = 1;
  x[2][1] = 0;
  x[2][2] = 0;

  sp.cal_best(x, 3);
  sp.print_result();

  return 0;
  */
}