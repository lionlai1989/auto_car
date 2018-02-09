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
  H_laser_ << 0.09, 0, 0, 0,
              0, 0.0009, 0, 0;

  // transformation measurement jacobian matrix
  Hj_ = MatrixXd(3, 4);
  Hj_ << 0.09, 0, 0, 0,
         0, 0.0009, 0, 0,
         0, 0, 0.09, 0;

  MatrixXd H = MatrixXd(4, 4);
  H << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  // state transition matrix
  MatrixXd F = MatrixXd(4, 4);
  F << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  // state covariance matrix
  MatrixXd P = MatrixXd(4, 4);
  P << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  // covariance matrix of process noise
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
    // Handle small px, py
    if(fabs(px) < 0.0001){
        px = 0.1;
        cout << "init px too small" << endl;
    }
    if(fabs(py) < 0.0001){
        py = 0.1;
        cout << "init py too small" << endl;
    }

    // first measurement
    ekf_.x_ << px, py, 0, 0;
    cout << "x_: " << ekf_.x_ << endl;


    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  cout << "1" << endl;
  /* Prediction
   *
   * Update the state transition matrix F according to the new elapsed time.
   * - Time is measured in seconds.
   * Update the covariance matrix of process noise .
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  ekf_.Predict();

  /* Update
   *
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
      cout << "Radar update" << endl;
      cout << "before assign" << endl;
      ekf_.hx_ = VectorXd(3);
      cout << "after assign" << endl;
      double px = ekf_.x_[0];
      cout << "gan";
      double py = ekf_.x_[1];
      double vx = ekf_.x_[2];
      double vy = ekf_.x_[3];

      float rho;
      float phi;
      float rhodot;
      cout << "a1";
      if(fabs(px) < 0.0001 or fabs(py) < 0.0001){
        cout << "a2";
        if(fabs(px) < 0.0001){
          px = 0.0001;
          cout << "px too small" << endl;
        }

        if(fabs(py) < 0.0001){
          py = 0.0001;
          cout << "py too small" << endl;
        }
        cout << "a3";
        rho = sqrt(px*px + py*py);
        phi = 0;
        rhodot = 0;
  
      } else {
        cout << "a4";
        rho = sqrt(px*px + py*py);
        phi = atan2(py,px); //  arc tangent of y/x, in the interval [-pi,+pi] radians.
        rhodot = (px*vx + py*vy) /rho;
      }      
      cout << "a5";
      ekf_.hx_ << rho, phi, rhodot;
      cout << "before jac";
      // set H_ to Hj when updating with a radar measurement
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      cout << "after jac";
      // don't update measurement if we can't compute the Jacobian
      if (Hj_.isZero(0)){
        cout << "Hj is zero" << endl;
        return;
      }
      
      ekf_.H_ = Hj_;

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
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
