#include <iostream>
#include "tools.h"
#include <assert.h>
#include <math.h>
#include <cstddef>

using std::cin;
using std::cout;
using std::vector;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  assert (estimations.size() != 0);
  assert (estimations.size() == ground_truth.size());

  VectorXd sum(4);
  //cout << estimations.begin() << std::endl;
      VectorXd rmse(4);
rmse << 0,0,0,0;
for(unsigned int i=0; i < estimations.size(); ++i){
        
        VectorXd residual = estimations[i] - ground_truth[i];
        
        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
}
return rmse;
/*  for (size_t i1 = estimations.begin(), size_t i2 = ground_truth.begin();
       i1 != estimations.end() && i2 != ground_truth.end*();
       ++i1, ++i2) {
    cout << estimations[i1] - ground_truth[i2] << std::endl;
//    VectorXd diff = estimations[i1] - ground_truth[i2];
//    diff = pow(diff, 2);
//    sum += 

  }
  return sqrt(sum);*/
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{


}
