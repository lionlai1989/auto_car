#include "kalman_filter.h"
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init() {
}

void KalmanFilter::Predict() {
  /*
   * predict the state
   */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /*
   * update the state by using Kalman Filter equations
   */

  VectorXd y = z - H_ * x_;

  MatrixXd Ht = H_.transpose();  
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
   * update the state by using Extended Kalman Filter equations
   */
  double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
//  double theta = atan(x_(1) / x_(0));
  double theta = atan2(x_(1), x_(0));
  double rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
  VectorXd h = VectorXd(3); // h(x_)
  h << rho, theta, rho_dot;
  
  VectorXd y = z - h;

  MatrixXd Ht = H_.transpose();  
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
