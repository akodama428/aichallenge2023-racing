// Copyright 2018-2021 The Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mpc_lateral_controller/vehicle_model/vehicle_model_bicycle_dynamics.hpp"

#include <algorithm>

namespace autoware::motion::control::mpc_lateral_controller
{
DynamicsBicycleModel::DynamicsBicycleModel(
  const double wheelbase, const double mass_fl, const double mass_fr, const double mass_rl,
  const double mass_rr, const double cf, const double cr, const double steer_tau)
: VehicleModelInterface(/* dim_x */ 5, /* dim_u */ 1, /* dim_y */ 2, wheelbase)
{
  const double mass_front = mass_fl + mass_fr;
  const double mass_rear = mass_rl + mass_rr;

  m_mass = mass_front + mass_rear;
  m_lf = m_wheelbase * (1.0 - mass_front / m_mass);
  m_lr = m_wheelbase * (1.0 - mass_rear / m_mass);
  m_iz = m_lf * m_lf * mass_front + m_lr * m_lr * mass_rear;
  m_cf = cf;
  m_cr = cr;
  m_steer_tau = steer_tau;
}

void DynamicsBicycleModel::calculateDiscreteMatrix(
  Eigen::MatrixXd & a_d, Eigen::MatrixXd & b_d, Eigen::MatrixXd & c_d, Eigen::MatrixXd & w_d,
  const double dt)
{
  /*
   * x[k+1] = a_d*x[k] + b_d*u + w_d
   */

  const double vel = std::max(m_velocity, 1.0);

  a_d = Eigen::MatrixXd::Zero(m_dim_x, m_dim_x);
  a_d(0, 1) = 1.0;
  a_d(1, 1) = -(m_cf + m_cr) / (m_mass * vel);
  a_d(1, 2) = (m_cf + m_cr) / m_mass;
  a_d(1, 3) = (m_lr * m_cr - m_lf * m_cf) / (m_mass * vel);
  a_d(1, 4) = m_cf / m_mass;
  a_d(2, 3) = 1.0;
  a_d(3, 1) = (m_lr * m_cr - m_lf * m_cf) / (m_iz * vel);
  a_d(3, 2) = (m_lf * m_cf - m_lr * m_cr) / m_iz;
  a_d(3, 3) = -(m_lf * m_lf * m_cf + m_lr * m_lr * m_cr) / (m_iz * vel);
  a_d(3, 4) = (m_lf * m_cf) / m_iz;
  a_d(4, 4) = - 1.0 / m_steer_tau; 

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m_dim_x, m_dim_x);
  Eigen::MatrixXd a_d_inverse = (I - dt * 0.5 * a_d).inverse();

  a_d = a_d_inverse * (I + dt * 0.5 * a_d);  // bilinear discretization

  b_d = Eigen::MatrixXd::Zero(m_dim_x, m_dim_u);
  b_d(0, 0) = 0.0;
  b_d(1, 0) = 0.0;
  b_d(2, 0) = 0.0;
  b_d(3, 0) = 0.0;
  b_d(4, 0) = 1.0 / m_steer_tau;

  w_d = Eigen::MatrixXd::Zero(m_dim_x, 1);
  w_d(0, 0) = 0.0;
  w_d(1, 0) = (m_lr * m_cr - m_lf * m_cf) / (m_mass * vel) - vel;
  w_d(2, 0) = 0.0;
  w_d(3, 0) = -(m_lf * m_lf * m_cf + m_lr * m_lr * m_cr) / (m_iz * vel);
  w_d(4, 0) = 0.0;

  b_d = (a_d_inverse * dt) * b_d;
  w_d = (a_d_inverse * dt * m_curvature * vel) * w_d;

  c_d = Eigen::MatrixXd::Zero(m_dim_y, m_dim_x);
  c_d(0, 0) = 1.0;
  c_d(1, 2) = 1.0;
}

void DynamicsBicycleModel::calculateReferenceInput(Eigen::MatrixXd & u_ref)
{
  const double vel = std::max(m_velocity, 0.01);
  const double Kv =
    m_lr * m_mass / (2 * m_cf * m_wheelbase) - m_lf * m_mass / (2 * m_cr * m_wheelbase);
  u_ref(0, 0) = m_wheelbase * m_curvature + Kv * vel * vel * m_curvature;
}
}  // namespace autoware::motion::control::mpc_lateral_controller
