#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string.h>

#include "struct_perceptron.h"

using namespace std;

int main(int argc, char* argv[]){
  /* argv
   * 1                              : x_dim
   * 2 ~ 1+x_dim                    : x_len
   * 2+x_dim ~ 1+x_dim+x_len*x_dim  : x
   */
  
  int x_dim;
  int y_dim;
  int x_len;
  int phi_dim;

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
  in_file.getline(str, length);
  y_dim = atoi(str);
  //cout<<"x_dim : "<<x_dim<<'\n';
  

  StructPerceptron sp = StructPerceptron(x_dim, y_dim);

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
  in_file.close();

  sp.cal_best(x, x_len);
  sp.print_result();
  
  return 0;
}