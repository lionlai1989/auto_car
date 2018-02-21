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
  cout << "Init FusionEKF ..." << endl;
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // laser's covariance matrix of measurement noise
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // radar's covariance matrix of measurement noise
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // laser's transformation measurement matrix
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // transformation measurement jacobian matrix
  Hj_ = MatrixXd(3, 4);
//  Hj_ << 0.09, 0, 0, 0,
//         0, 0.0009, 0, 0,
//         0, 0, 0.09, 0;

  VectorXd x = VectorXd(4);
  x << 0, 0, 0, 0;

  // state covariance matrix
  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1000, 0,
       0, 0, 0, 1000;

  // state transition matrix
  MatrixXd F = MatrixXd(4, 4);
//  F << 1, 0, 0, 0,
//       0, 1, 0, 0,
//       0, 0, 1, 0,
//       0, 0, 0, 1;

  // covariance matrix of process noise
  MatrixXd Q = MatrixXd(4,4);
//  Q << 1, 0, 0, 0,
//       0, 1, 0, 0,
//       0, 0, 1, 0,
//       0, 0, 0, 1;

  MatrixXd H = MatrixXd(4, 4);
//  H << 1, 0, 0, 0,
//       0, 1, 0, 0,
//       0, 0, 1, 0,
//       0, 0, 0, 1;

  ekf_.Init(x, P);
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

      px = rho * cos(theta);
      py = rho * sin(theta);
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
    // Handle small px, py
    if (fabs(px) < 0.0001) {
        px = 0.1;
        cout << "init px too small" << endl;
    }
    if (fabs(py) < 0.0001) {
        py = 0.1;
        cout << "init py too small" << endl;
    }

    // first measurement. Assume velocity is 0.
    ekf_.x_ << px, py, 0, 0;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /* Prediction
   *
   * Update the state transition matrix F according to the new elapsed time.
   * - Time is measured in seconds.
   * Update the covariance matrix of process noise .
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  double dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt = dt / 1000000.0; // convert micro second to second.
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
  // Noise covariance matrix computation  
  // Noise values from the task
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  // Precompute some usefull values to speed up calculations of Q
  float dt_2 = dt * dt; //dt^2
  float dt_3 = dt_2 * dt; //dt^3
  float dt_4 = dt_3 * dt; //dt^4
  float dt_4_4 = dt_4 / 4; //dt^4/4
  float dt_3_2 = dt_3 / 2; //dt^3/2
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
             0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
             dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
             0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;
  ekf_.Predict();

  /* Update
   *
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // Use Jacobian instead of H
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  } else {
    cout << "ERROR !!!" << endl;
  }

  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
