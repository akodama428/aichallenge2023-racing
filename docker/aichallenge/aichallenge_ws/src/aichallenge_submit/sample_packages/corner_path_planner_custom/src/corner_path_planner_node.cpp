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

#include "corner_path_planner/corner_path_planner_node.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"
#include "tier4_autoware_utils/geometry/pose_deviation.hpp"
#include "tier4_autoware_utils/math/constants.hpp"

template <class T>
std::vector<double> calcCurvature(const T & points)
{
  std::vector<double> curvature_vec(points.size());

  for (size_t i = 1; i < points.size() - 1; ++i) {
    const auto p1 = tier4_autoware_utils::getPoint(points.at(i - 1));
    const auto p2 = tier4_autoware_utils::getPoint(points.at(i));
    const auto p3 = tier4_autoware_utils::getPoint(points.at(i + 1));
    curvature_vec.at(i) = (tier4_autoware_utils::calcCurvature(p1, p2, p3));
  }
  curvature_vec.at(0) = curvature_vec.at(1);
  curvature_vec.at(curvature_vec.size() - 1) = curvature_vec.at(curvature_vec.size() - 2);

  return curvature_vec;
}

std::optional<geometry_msgs::msg::Point> intersect(
  const geometry_msgs::msg::Point & p1, const geometry_msgs::msg::Point & p2,
  const geometry_msgs::msg::Point & p3, const geometry_msgs::msg::Point & p4)
{
  // calculate intersection point
  const double det = (p1.x - p2.x) * (p4.y - p3.y) - (p4.x - p3.x) * (p1.y - p2.y);
  if (det == 0.0) {
    return std::nullopt;
  }

  const double t = ((p4.y - p3.y) * (p4.x - p2.x) + (p3.x - p4.x) * (p4.y - p2.y)) / det;
  const double s = ((p2.y - p1.y) * (p4.x - p2.x) + (p1.x - p2.x) * (p4.y - p2.y)) / det;
  if (t < 0 || 1 < t || s < 0 || 1 < s) {
    return std::nullopt;
  }

  geometry_msgs::msg::Point intersect_point;
  intersect_point.x = t * p1.x + (1.0 - t) * p2.x;
  intersect_point.y = t * p1.y + (1.0 - t) * p2.y;
  return intersect_point;
}

bool isLeft(const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Point & target_pos)
{
  const double base_theta = tf2::getYaw(pose.orientation);
  const double target_theta = tier4_autoware_utils::calcAzimuthAngle(pose.position, target_pos);
  const double diff_theta = tier4_autoware_utils::normalizeRadian(target_theta - base_theta);
  return diff_theta > 0;
}

double calcLateralDistToBounds(
  const geometry_msgs::msg::Pose & pose, const std::vector<geometry_msgs::msg::Point> & bound, const bool is_left_bound = true)
{
  constexpr double max_lat_offset_for_left = 5.0;
  constexpr double min_lat_offset_for_left = -5.0;

  const double max_lat_offset = is_left_bound ? max_lat_offset_for_left : -max_lat_offset_for_left;
  const double min_lat_offset = is_left_bound ? min_lat_offset_for_left : -min_lat_offset_for_left;
  const auto max_lat_offset_point =
    tier4_autoware_utils::calcOffsetPose(pose, 0.0, max_lat_offset, 0.0).position;
  const auto min_lat_offset_point =
    tier4_autoware_utils::calcOffsetPose(pose, 0.0, min_lat_offset, 0.0).position;

  double closest_dist_to_bound = max_lat_offset_for_left;
  for (size_t i = 0; i < bound.size() - 1; ++i) {
    const auto intersect_point =
      intersect(min_lat_offset_point, max_lat_offset_point, bound.at(i), bound.at(i + 1));
    if (intersect_point) {
      const bool is_point_left = isLeft(pose, *intersect_point);
      if ((is_left_bound && !is_point_left) || (!is_left_bound && is_point_left)) {
        return -1;
      }
      const double dist_to_bound = tier4_autoware_utils::calcDistance2d(pose.position, *intersect_point);
      closest_dist_to_bound = std::min(dist_to_bound, closest_dist_to_bound);
    }
  }
  return closest_dist_to_bound;
}

std::vector<double> smoothRatios(const std::vector<double>& ratios, int windowSize) {
    int n = ratios.size();
    std::vector<double> smoothedRatios(n, 0.0);

    if (n <= windowSize) {
        // ウィンドウサイズがデータサイズ以上の場合は何もしない
        return ratios;
    }

    // 移動平均計算
    for (int i = 0; i < n; ++i) {
        int start = std::max(0, i - windowSize + 1);
        int end = std::min(n - 1, i + windowSize - 1);
        for (int j = start; j <= end; ++j) {
            smoothedRatios[i] += ratios[j];
        }
        smoothedRatios[i] /= (end - start + 1);
    }

    return smoothedRatios;
}

CornerPathPlanner::CornerPathPlanner() : Node("corner_path_planner_node") {
  using std::placeholders::_1;

  pub_ = this->create_publisher<Path>("output", 1);
  sub_ = this->create_subscription<PathWithLaneId>(
      "input", 1, std::bind(&CornerPathPlanner::callback, this, _1));

  { // parameter
    csv_map_path_ = declare_parameter("csv_map_path", std::string("empty"));
    max_curvature_ = declare_parameter<double>("max_curvature");
    min_curvature_ = declare_parameter<double>("min_curvature");
    prepare_idx1_ = declare_parameter<int>("prepare_idx1");
    prepare_idx2_ = declare_parameter<int>("prepare_idx2");
    prepare_idx3_ = declare_parameter<int>("prepare_idx3");
    max_ratio_ = declare_parameter<double>("max_ratio");
    min_ratio_ = declare_parameter<double>("min_ratio");
    smoothing_window_size_ = declare_parameter<int>("smoothing_window_size");
  }
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&CornerPathPlanner::onParam, this, std::placeholders::_1));
  
  readPathFromCSV();
}

rcl_interfaces::msg::SetParametersResult CornerPathPlanner::onParam(
  const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;
  updateParam<double>(parameters, "max_curvature", max_curvature_);
  updateParam<double>(parameters, "min_curvature", min_curvature_);
  updateParam<double>(parameters, "max_ratio", max_ratio_);
  updateParam<double>(parameters, "min_ratio", min_ratio_);
  updateParam<int>(parameters, "prepare_idx1", prepare_idx1_);
  updateParam<int>(parameters, "prepare_idx2", prepare_idx2_);
  updateParam<int>(parameters, "prepare_idx3", prepare_idx3_);
  
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void CornerPathPlanner::readPathFromCSV()
{
  CSVLoader csv(csv_map_path_);
  std::vector<std::vector<std::string>> map;
  if (!csv.readCSV(map)) {
    RCLCPP_INFO(get_logger(), "csv read error!");
    return;
  }

  const std::vector<double> x_array = CSVLoader::getColumnArray(map, 0);
  const std::vector<double> y_array = CSVLoader::getColumnArray(map, 1);
  const std::vector<double> z_array = CSVLoader::getColumnArray(map, 2);
  const std::vector<double> yaw_array = CSVLoader::getColumnArray(map, 3);
  const std::vector<double> v_array = CSVLoader::getColumnArray(map, 4);
  const std::vector<double> a_array = CSVLoader::getColumnArray(map, 5);

  if ((y_array.size() != x_array.size())
      || (z_array.size() != x_array.size())
      || (yaw_array.size() != x_array.size())
      || (v_array.size() != x_array.size())
      || (a_array.size() != x_array.size())) {
    RCLCPP_INFO(get_logger(), "array size error!");
    return;
  }

  for (size_t i = 0; i < x_array.size(); i++) {
    PathPoint point;
    point.pose.position.x = x_array.at(i);
    point.pose.position.y = y_array.at(i);
    point.pose.position.z = z_array.at(i);
    point.pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw_array.at(i));
    point.longitudinal_velocity_mps = v_array.at(i);
    // かなり大着で、目標加速度を横速度にセット
    point.lateral_velocity_mps = a_array.at(i);
    csv_points_.push_back(point);
  }
}

void CornerPathPlanner::toPath(const PathWithLaneId & input, Path & output)
{
  output.header = input.header;
  output.left_bound = input.left_bound;
  output.right_bound = input.right_bound;
  output.points.resize(input.points.size());
  for (size_t i = 0; i < input.points.size(); ++i) {
    output.points.at(i) = input.points.at(i).point;
  }
}

void CornerPathPlanner::callback(const PathWithLaneId::SharedPtr msg) {
  Path original_path{};
  toPath(*msg, original_path);
  Path output_path = original_path;
  // updateLongitudinalVelocity(original_path, output_path);
  calcPathBoundsArray(output_path);
  pub_->publish(output_path);
}

void CornerPathPlanner::updateLongitudinalVelocity(const Path & original_path, Path & output_path)
{
  output_path = original_path;
  const double vehicle_width = 0.196 + 0.145*2; // TODO：vehicle_infoから読み込む。wheel_width+left_overhang+right_overhang
  bool is_narrow_path = false;

  // original_pathの各ポイントと最も近いcsv_pointを見つけて、目標車速を割り当てる
  for (auto & output_p : output_path.points) {
    double nearest_dist = 10e+10;  // 十分大きい値
    for (const auto & csv_p : csv_points_) {
      double dist = tier4_autoware_utils::calcDistance2d(output_p, csv_p);
      if(dist < nearest_dist) {
        nearest_dist = dist;
        output_p.longitudinal_velocity_mps = csv_p.longitudinal_velocity_mps;
        output_p.lateral_velocity_mps = csv_p.lateral_velocity_mps;
      }
    }
    const double right_bound_dist = calcLateralDistToBounds(output_p.pose, output_path.right_bound, false);
    const double left_bound_dist  = calcLateralDistToBounds(output_p.pose, output_path.left_bound, true);
    if ((right_bound_dist + left_bound_dist) < vehicle_width) {
      is_narrow_path = true;
    }
  }

  // 他車両によりコースを通過できない場合は、目標車速を落とす
  if (is_narrow_path) {
      RCLCPP_INFO(get_logger(), "narrow path!");
      for (auto & output_p : output_path.points) {
        output_p.longitudinal_velocity_mps = 30 / 3.6;  // TODO: 他車両の速度に合わせる
        output_p.lateral_velocity_mps = 0.0;
      }
  }

  // RCLCPP_INFO(get_logger(), "replace path velocity!");
}

std::vector<double> CornerPathPlanner::calcRatioFromCurvature(std::vector<double> curvature_vec) 
{
  std::vector<double> ratio_vec;
  for (size_t i = 0; i < curvature_vec.size(); ++i) {
    const double cliped_curvature = std::min(std::max(curvature_vec.at(i), min_curvature_), max_curvature_);
    ratio_vec.emplace_back((cliped_curvature - min_curvature_) / (max_curvature_ - min_curvature_));
  }
  return ratio_vec;
}

int CornerPathPlanner::calcPathPointsByRatio(Path & path, const std::vector<double> & ratio) 
{
  if (path.points.size() != ratio.size()) {
    RCLCPP_INFO(get_logger(), "ratio size is not same as path size");
    return 0;
  }

  for (size_t i = 0; i < path.points.size(); i++) {
    geometry_msgs::msg::Pose path_pose = path.points.at(i).pose;
    double right_bound_dist = calcLateralDistToBounds(path_pose, path.right_bound, false);
    double left_bound_dist  = calcLateralDistToBounds(path_pose, path.left_bound, true);
    Point right_point = tier4_autoware_utils::calcOffsetPose(path_pose, 0.0, -right_bound_dist, 0.0).position;
    Point left_point  = tier4_autoware_utils::calcOffsetPose(path_pose, 0.0, left_bound_dist, 0.0).position;
    path.points.at(i).pose.position 
    //  = tier4_autoware_utils::calcInterpolatedPoint(right_point, left_point, 0.5);
     = tier4_autoware_utils::calcInterpolatedPoint(right_point, left_point, ratio.at(i));
  }
  return 0;
}

int CornerPathPlanner::calcPathBoundsArray(Path & path) 
{
  std::vector<Point> left_bound_array, right_bound_array;
  for (size_t i = 0; i < path.points.size(); i++) {
    geometry_msgs::msg::Pose path_pose = path.points.at(i).pose;
    double right_bound_dist = calcLateralDistToBounds(path_pose, path.right_bound, false);
    double left_bound_dist  = calcLateralDistToBounds(path_pose, path.left_bound, true);
    Point right_point = tier4_autoware_utils::calcOffsetPose(path_pose, 0.0, -right_bound_dist, 0.0).position;
    Point left_point  = tier4_autoware_utils::calcOffsetPose(path_pose, 0.0, left_bound_dist, 0.0).position;
    right_bound_array.push_back(right_point);
    left_bound_array.push_back(left_point);
  }
  path.right_bound = right_bound_array;
  path.left_bound  = left_bound_array;
  return 0;
}

int main(int argc, char const* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CornerPathPlanner>());
  rclcpp::shutdown();
  return 0;
}
