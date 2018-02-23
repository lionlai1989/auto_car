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
  assert (estimations.size() != 0 && "The estimations vector should not be 0.");
  assert (estimations.size() == ground_truth.size() && "estimations and ground_truth size should be the same.");

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    
    diff = diff.array() * diff.array();
    rmse += diff;
  }
  
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);
  MatrixXd Hj(3,4);

  // Pre-compute a set of terms to avoid repeated calculation
  double c1 = px*px+py*py;
  double c2 = sqrt(c1);
  double c3 = (c1*c2);

  // Check division by zero
  if (c1 < 0.0000001) {
    c1 = 0.0001;
  }
  if (c2 < 0.0000001) {
    c2 = 0.0001;
  }
  if (c3 < 0.0000001) {
    c3 = 0.0001;
  }
  // Compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
       -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  return Hj;
}
