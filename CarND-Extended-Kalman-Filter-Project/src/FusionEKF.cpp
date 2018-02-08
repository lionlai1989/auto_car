#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <vector>

using std::cin;
using std::cout;
using std::vector;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 0.09, 0, 0, 0,
              0, 0.0009, 0, 0;

  //measurement jacobian matrix - 
  Hj_ = MatrixXd(3, 4);
  Hj_ << 0.09, 0, 0, 0,
         0, 0.0009, 0, 0,
         0, 0, 0.09, 0;

  MatrixXd H = MatrixXd(4, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd Q = MatrixXd(4,4);
  Q << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  VectorXd x = VectorXd(4);
  x << 0, 0, 0, 0;

  ekf_.Init(x, P, F, Q, H, R_laser_);
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) 
{
  /* Initialization */
  if (!is_initialized_) {
    double px, py;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /*
       * Convert radar from polar to cartesian coordinates.
       */
      cout << "Radar Init ..." << endl;

      double rho = measurement_pack.raw_measurements_[0];
      double theta = measurement_pack.raw_measurements_[1];
      double rho_dot = measurement_pack.raw_measurements_[2];

      px = rho*cos(theta);
      py = rho*sin(theta);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      cout << "Laser Init ..." << endl;

      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    } else {
      cout << "Sensor type is not radar nor laser. EXIT !!!" << endl; 
    }
    /*
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ << px, py, 0, 0;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  cout << "1" << endl;
  /* Prediction */

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  ekf_.Predict();

  /* Update */

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.Update(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
