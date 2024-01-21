# rl_planner_custom
強化学習plannerノード


# 単体実行
- 下記の通り、ノード単体起動をする必要あり
```
ros2 run rl_planner_custom rl_planner_custom_node

  --ros-args -r __node:=rl_planner_custom -r ~/input/path:=/planning/scenario_planning/lane_driving/behavior_planning/path -r ~/input/acceleration:=/localization/acceleration -r ~/input/odometry:=/localization/kinematic_state -r ~/input/perception:=/perception/object_recognition/objects -r ~/output/path:=/crank_driving_planner/path 
```
- launch実行では、bashを使用しているためプロセス操作がうまくいかない
```
ros2 launch rl_planner_custom rl_planner_custom.launch.xml
```