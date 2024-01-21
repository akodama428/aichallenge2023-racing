#!/bin/bash

sudo ip link set multicast on lo

source /aichallenge/aichallenge_ws/install/setup.bash
/aichallenge/AWSIM/AWSIM.x86_64
