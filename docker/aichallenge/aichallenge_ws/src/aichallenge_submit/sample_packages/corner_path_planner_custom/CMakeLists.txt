cmake_minimum_required(VERSION 3.8)
project(corner_path_planner_custom)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(corner_path_planner_node
  src/corner_path_planner_node.cpp
  src/csv_loader.cpp
)

ament_auto_package()