#!/bin/bash

sudo ip link set multicast on lo

rm ~/awsim-logs/*.json
rm /aichallenge/output/result.json 

source /aichallenge/aichallenge_ws/install/setup.bash
# rm -f result.json
ros2 launch aichallenge_launch aichallenge.launch.xml
