#include "kalman_filter.h"
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

void NormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

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
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  P_ -= K * H_ * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
   * update the state by using Extended Kalman Filter equations
   */
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
  double rho = sqrt(px * px + py * py);

  if (py == 0 && px == 0) {
    py = 0.0001;
    px = 0.0001;
  }
  double theta = atan2(py, px);
  if (rho < 0.0000001) {
    rho = 0.0001;
  }
  double rho_dot = (px * vx + py * vy) / rho;

  VectorXd h_x_ = VectorXd(3);
  h_x_ << rho, theta, rho_dot;
  
  VectorXd y = z - h_x_;
  /* Tips from course:
   * In C++, atan2() returns values between -pi and pi. When calculating phi 
   * in y = z - h(x) for radar measurements, the resulting angle phi in the 
   * y vector should be adjusted so that it is between -pi and pi. The Kalman
   *  filter is expecting small angle values between the range -pi and pi. 
   * HINT: when working in radians, you can add 2*pi or subtract 2*pi until 
   * the angle is within the desired range.
   */
  NormalizeAngle(y(1));

  MatrixXd Ht = H_.transpose();  
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  x_ = x_ + (K * y);
  P_ -= K * H_ * P_;
}
