#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"

class KalmanFilter {
public:
  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // covariance matrix of process noise
  Eigen::MatrixXd Q_;

  // transformation measurement matrix
  Eigen::MatrixXd H_;

  // covariance matrix of measurement noise
  Eigen::MatrixXd R_;

  KalmanFilter();

  virtual ~KalmanFilter();

  /*
   * Prediction Predicts the state and the state covariance
   * using the process model
   */
  void Predict();

  /*
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  /*
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   */
  void UpdateEKF(const Eigen::VectorXd &z);

};

#endif /* KALMAN_FILTER_H_ */