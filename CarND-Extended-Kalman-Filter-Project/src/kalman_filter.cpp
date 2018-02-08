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

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &Q_in, MatrixXd &H_in, MatrixXd &R_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;
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
  cout << "KalmanFilter::Update()" << endl;
  cout << "x_: " << x_ << endl;
  cout << "H_: " << H_ << endl;
  VectorXd z_pred = H_ * x_;
  cout << "z_pred: " << z_pred << endl;
  
  cout << "z: " << z << endl;
  VectorXd y = z - z_pred;
  cout << "y: " << y << endl;
  MatrixXd Ht = H_.transpose();
  cout << "Ht: " << Ht << endl;
  
  cout << "H_: " << H_ << endl;
  cout << "P: " << P_ << endl;
  cout << "R_: " << R_ << endl;
  MatrixXd S = H_ * P_ * Ht + R_;
  cout << "S: " << S << endl;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  cout << "K: " << K << endl;
  
  //new estimate
  cout << "y: " << y << endl;
  cout << "x_: " << x_ << endl;
  x_ = x_ + (K * y);
  cout << "x_: " << x_ << endl;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  cout << "P_: " << P_ << endl;
  cout << "I: " << I << endl;
  cout << "K: " << K << endl;
  cout << "H_: " << H_ << endl;
  P_ = (I - K * H_) * P_;
  cout << "P_: " << P_ << endl;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /*
   * update the state by using Extended Kalman Filter equations
   */
}
