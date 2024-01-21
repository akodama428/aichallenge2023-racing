// Copyright 2023 Tier IV, Inc. All rights reserved.
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
#ifndef PATH_TO_TRAJECTORY__PATH_TO_TRAJECTORY_HPP_
#define PATH_TO_TRAJECTORY__PATH_TO_TRAJECTORY_HPP_

#include "autoware_auto_planning_msgs/msg/path.hpp"
#include "autoware_auto_planning_msgs/msg/path_point.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory_point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"
#include "corner_path_planner/csv_loader.hpp"

class CornerPathPlanner : public rclcpp::Node {
 public:
  using PathWithLaneId = autoware_auto_planning_msgs::msg::PathWithLaneId;
  using Path = autoware_auto_planning_msgs::msg::Path;
  using PathPoint = autoware_auto_planning_msgs::msg::PathPoint;
  using Trajectory = autoware_auto_planning_msgs::msg::Trajectory;
  using TrajectoryPoint = autoware_auto_planning_msgs::msg::TrajectoryPoint;
  using Point = geometry_msgs::msg::Point;
 public:
  CornerPathPlanner();

 private:
  double max_curvature_;
  double min_curvature_;
  int smoothing_window_size_;
  double max_ratio_;
  double min_ratio_;
  int prepare_idx1_;
  int prepare_idx2_;
  int prepare_idx3_;
  std::string csv_map_path_; 
  std::vector<PathPoint> csv_points_;

  void toPath(const PathWithLaneId & input, Path & output);
  void callback(const PathWithLaneId::SharedPtr msg);
  void readPathFromCSV();
  void updateLongitudinalVelocity(const Path & original_path, Path & output_path);
  std::vector<double> calcRatioFromCurvature(std::vector<double> curvature_vec);
  int calcPathPointsByRatio(Path & path, const std::vector<double> & ratio);
  int calcPathBoundsArray(Path & path); 
  rclcpp::Subscription<PathWithLaneId>::SharedPtr sub_;
  rclcpp::Publisher<Path>::SharedPtr pub_;
  // parameter callback
  rcl_interfaces::msg::SetParametersResult onParam(
    const std::vector<rclcpp::Parameter> & parameters);
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

};

#endif  // PATH_TO_TRAJECTORY__PATH_TO_TRAJECTORY_HPP_