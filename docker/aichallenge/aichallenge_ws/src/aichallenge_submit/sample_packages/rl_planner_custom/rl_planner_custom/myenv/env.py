import rclpy  # ROS2のPythonモジュール
from rclpy.node import Node
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String 
from std_msgs.msg import Header

from autoware_auto_planning_msgs.msg import Path
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_vehicle_msgs.msg import VelocityReport
from autonoma_msgs.msg import VehicleInputs
from tier4_vehicle_msgs.msg import ActuationCommandStamped

import gymnasium as gym
from gymnasium import spaces
# from stable_baselines3.common.env_util import make_vec_env

# 模倣学習用
from imitation.data.types import Trajectory as Ts
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import sys
import os
sys.path.append(os.path.dirname(__file__))
from util import *
from predicted_objects_info import PredictedObjectsInfo

import numpy as np
import threading
import subprocess
import psutil
import shutil
from time import sleep
import json
import pickle

# エキスパートのデータ保存中は、VehicleInputsを受信して、トラジェクトリとして保存する
record_expart_data = True
expert_data_path = "/aichallenge/output/expert_data.pickle"
class CustomEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # ROS2ノードの設定（このノードを通してSimと通信する）
        self.autoware_if_node = AutowareIfNode()
        thread = threading.Thread(target=rclpy.spin, args=(self.autoware_if_node,), daemon=True)
        thread.start()
        
        ## For plot
        from debug_plot import PlotMarker
        self.plot_marker = PlotMarker()

        self.path_sampling = 2
        self.path_length = int(120 / self.path_sampling)

        # 行動空間：[throttle_cmd, brake_cmd]
        max_cmd = np.array([1.0, 1.0])
        min_cmd = np.array([0, 0])
        self.action_space = gym.spaces.Box(low=min_cmd, high=max_cmd, shape=(2,), dtype=np.float32)
        # 状態空間：[x座標、y座標、車速x、コース左端x座標、コース左端y座標、コース右端x座標、コース右端y座標]
        max_x = 23145
        min_x = 21878
        max_y = 53130
        min_y = 50861
        # この最大最小の範囲は要見直し。path indexに応じて変更する必要あり
        max_value = np.array([max_x, max_y, 70]) ## x座標、y座標、車速 
        max_value = np.append(max_value, np.ones(self.path_length)*200) ## コース左端x座標 
        max_value = np.append(max_value, np.ones(self.path_length)*10) ## コース左端y座標 
        max_value = np.append(max_value, np.ones(self.path_length)*200) ## コース右端x座標 
        max_value = np.append(max_value, np.ones(self.path_length)*5) ## コース右端y座標 
        print(f"max value length:{len(max_value)}")

        min_value = np.array([min_x, min_y, -5]) ## x座標、y座標、車速 
        min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース左端x座標 
        min_value = np.append(min_value, np.ones(self.path_length)*(-5)) ## コース左端y座標 
        min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース右端x座標 
        min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース右端y座標
        print(f"min value length:{len(min_value)}")

        self.observation_space = gym.spaces.Box(low=min_value, high=max_value, shape=(3+self.path_length*4,), dtype=np.float32)

        # カウンタ
        self.step_count = 0
        self.max_episode_len = 2000

        # 走行開始フラグ
        self.started = False

        # 最終評価（走行距離）
        self.evaluation = 0.0

        # 模倣学習用
        self.trajectorys = []

        # 初回のシミュレータを起動
        self.startSimulation()

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        print("reset call!")
        super().reset(seed=seed, options=options)
        # シミュレーションのシャットダウン
        self.shutdownSimulation()
        # AutoreIFノードの初期化
        self.autoware_if_node.reset_if()
        # シミュレーションの開始
        self.startSimulation()
        self.step_count = 0

        state = self.get_state()

        # 模倣学習用データ初期化
        self.actions = []
        self.observations = [state]
        self.infos = []

        return state.astype(np.float32), {}  # empty info dict
        # return np.array([self.agent_pos]).astype(np.float32), {}  # empty info dict

    def step(self, action):
        self.step_count += 1
        # print("action: {} , action type: {}".format(action[0], action[0].dtype))
        # 目標値を設定する
        if record_expart_data == False:
            self.autoware_if_node.throttle_cmd = action[0]
            self.autoware_if_node.brake_cmd    = action[1]

        # 制御出力されるまで待機
        # sleep(0.1)

        # 状態量の取得
        state = self.get_state()
        # print(f"state length:{len(state)}")

        # 報酬の計算
        velocity = self.autoware_if_node.get_velocity()
        # 車速に応じて報酬を与える
        reward = velocity
        # ただし、車速が30kph以下で減速した場合はペナルティを与える
        if velocity < 10.0:
            if self.autoware_if_node.brake_cmd > 0.0:
                reward -= 50.0

        # ただし、コースアウトや衝突した場合は、ペナルティを与える
        if (self.left_bound_at_ego < 0) or (self.right_bound_at_ego > 0):
                reward -= 50.0

        # 走り始めてから再度車速が０以下たは規定ステップ数を超えたら終了
        if velocity > 0:
            self.started = True
        if self.started == True: # シミュレーション開始時に終了するのを防止    
            stopped_flag = (velocity <= 0.0)        
        # terminated = stopped_flag or (self.step_count > self.max_episode_len)
        terminated = (self.step_count > self.max_episode_len) or reward < -50.0
        # we do not limit the number of steps here
        truncated = False
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        if record_expart_data == True:
            self.actions.append(self.get_action())
            self.observations.append(state)
            self.infos.append(info)

        if terminated:
            if record_expart_data == True:
                ts = Ts(obs=np.array(self.observations), acts=np.array(self.actions), infos=np.array(self.infos), terminal = True)
                self.trajectorys.append(ts)
                print("terminated!")
                with open(expert_data_path, mode="wb") as f:
                    pickle.dump(self.trajectorys, f)
                print("finish to make pickle file!")

            # シミュレーションのシャットダウン
            self.shutdownSimulation()
            # jsonから評価値を読み込む
            result_json = '/aichallenge/output/result.json'
            self.evaluation = self.evaluateScore(result_json)
            print("evaluation: {}".format(self.evaluation))

            # まずは走行距離と最後の車速を最終報酬とする
            reward = self.evaluation + velocity

        # print(f"{self.step_count} state: ", end="")
        # for i in range(len(state)):
        #     print(f"{state[i]:.1f},", end="")
        print(f"step:{self.step_count}, action:{self.autoware_if_node.throttle_cmd:.1f}, {self.autoware_if_node.brake_cmd:.1f}, reward:{reward:.1f}")

        return (
            state.astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        pass

    def close(self):
        # シミュレーションのシャットダウン
        self.shutdownSimulation()

    def get_state(self):
        ego_pose_array = ConvertPoint2List(self.autoware_if_node.ego_pose)
        left_bound     = self.autoware_if_node.left_bound
        right_bound    = self.autoware_if_node.right_bound
        self.ego_x     = ego_pose_array[0]
        self.ego_y     = ego_pose_array[1]
        yaw            = ego_pose_array[2]
        ego_rot_inv = np.array([[math.cos(yaw), -math.sin(yaw)],
                                [math.sin(yaw), math.cos(yaw)]])
        left_bound  = left_bound[:, 0:2] - ego_pose_array[0:2] 
        left_bound  = left_bound[:, 0:2] @ ego_rot_inv
        right_bound = right_bound[:, 0:2] - ego_pose_array[0:2] 
        right_bound = right_bound[:, 0:2] @ ego_rot_inv

        for i in range(len(left_bound)):
            if left_bound[i, 0] > 0:
                self.left_bound_at_ego = left_bound[i, 1]
                break
        for i in range(len(right_bound)):
            if right_bound[i, 0] > 0:
                self.right_bound_at_ego = right_bound[i, 1]
                break
        
        # パス長さが想定の長さより短かった場合は、最後の値で埋める
        state_path_length = self.path_length*self.path_sampling
        if len(left_bound) < state_path_length:
            while len(left_bound) < state_path_length + 1:
                left_bound = np.vstack((left_bound, left_bound[-1, :]))
            print(f"left_bound_length:{len(left_bound)}")
        if len(right_bound) < state_path_length:
            while len(right_bound) < state_path_length + 1:
                right_bound = np.vstack((right_bound, right_bound[-1, :]))
            print(f"right_bound_length:{len(right_bound)}")

        state = np.array([self.ego_x, self.ego_y])
        state = np.append(state, self.autoware_if_node.get_velocity())
        state = np.append(state,  left_bound[:state_path_length:self.path_sampling, 0])
        state = np.append(state,  left_bound[:state_path_length:self.path_sampling, 1])
        state = np.append(state, right_bound[:state_path_length:self.path_sampling, 0])
        state = np.append(state, right_bound[:state_path_length:self.path_sampling, 1])

        self.plot_marker.plot_status(ego_pose = ego_pose_array,
                                     object_pose = self.autoware_if_node.obj_pose,
                                     left_bound  = self.autoware_if_node.left_bound,
                                     right_bound = self.autoware_if_node.right_bound,
                                     path=None,
                                     path_index_left=None,
                                     path_index_next_left=None,
                                     path_index_right= None,
                                     path_index_next_right=None,
                                     rotation = True,
                                     predicted_goal_pose=None,
                                     predicted_trajectory=None,
                                     curve_plot=None,
                                     curve_forward_point=None,
                                     curve_backward_point=None,
                                     vis_point=None
                                     )
        return state
    
    def get_action(self):
        action = [0.0, 0.0]
        action[0] = self.autoware_if_node.throttle_cmd
        action[1] = self.autoware_if_node.brake_cmd
        return action

    def startSimulation(self):
        # Autowareを起動
        p_autoware = subprocess.Popen("exec " + "bash /aichallenge/run_autoware.sh", shell=True)
        # シミュレータを起動
        p_awsim = subprocess.Popen("exec " + "/aichallenge/AWSIM/AWSIM.x86_64", shell=True)
        
        # AutowareIFノードが、トピック受信できるまで待機
        while not self.autoware_if_node.isReady():
            print("Not ready AutowareIfNode")
            sleep(1)
        print(f"path length {len( self.autoware_if_node.left_bound[:,0])}")

    def shutdownSimulation(self):
        # 現在のPythonプロセス以外のすべてのプロセスをキル
        current_process = psutil.Process()
        print("current_process: {}".format(current_process))
        for process in psutil.process_iter(attrs=['pid', 'name']):
            if (process.info['name'] != "ros2") & (process.info['name'] != "rl_planner_cust") :
                print("eprocess kill: {}".format(process.info['name']))
                process.terminate()  # プロセスを終了させる

    def evaluateScore(self, result_json):
        p_result_copy = subprocess.Popen("exec " + "bash /aichallenge/copy_result.sh", shell=True)
        # ファイルが存在するまで待機
        while not os.path.exists(result_json):
            sleep(1)
        try:
            with open(result_json, 'r') as results_file:
                results = json.load(results_file)
            # evaluation = results['rawDistanceScore']
            evaluation = results['distanceScore']
            isLapCompleted = results['isLapCompleted']
            # １周するとScoreが０となるため、ゴールしたらスコアは一律1200とする
            if isLapCompleted == True:
                evaluation = 1200
            else:
                # ずっと止まっているとScoreが１周分をカウントして1123となるので、１周していなくて1122以上の場合は、０にする
                if evaluation > 1122:
                    evaluation = 0
            
            # ペナルティの計算
            penalty = results['lapTime'] - results['rawLapTime'] 
            evaluation -= penalty 
        except:
            evaluation = 0.0

        return evaluation

# Autowareとトピック通信するためのノード
class AutowareIfNode(Node):
    def __init__(self, num = 0):
        super().__init__(f'autoware_if_node{num}')
        ## Timer callback function. Publish command ##
        self.ctl_period = 0.005
        self.create_timer(self.ctl_period, self.onTimer)
        ## Remapがうまくできないので、トピック名を直接設定
        # self.create_subscription(Path ,"/planning/scenario_planning/lane_driving/behavior_planning/path", self.onPath, 10)
        self.create_subscription(Path ,"/planning/scenario_planning/corner_path_planning/path", self.onPath, 10)
        self.create_subscription(AccelWithCovarianceStamped, "/localization/acceleration", self.onAcceleration, 10)
        self.create_subscription(Odometry, "/localization/kinematic_state", self.onOdometry, 10)
        self.create_subscription(PredictedObjects, "/perception/object_recognition/objects", self.onPerception, 10)
        self.create_subscription(VehicleInputs, "/rl_planner_inputs", self.onInputsRaw, 10)
        self.pub_cmd_ = self.create_publisher(VehicleInputs, "/vehicle_inputs", 10)

        # IFの初期化
        self.reset_if()

    def get_bound_distances(self):
        self.bound_distances = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return self.bound_distances
    
    def get_velocity(self):
        return self.current_vel_x
    
    def reset_if(self):
        self.reference_path = None
        self.current_accel = None
        self.current_odometry = None
        self.ego_pose = None
        self.current_vel_x = None
        self.current_vel_y = None
        self.dynamic_objects = None
        self.vehicle_inputs_raw = None
        self.throttle_cmd = 0.0
        self.brake_cmd = 0.0
        self.steering_cmd = 0.0
        self.gear_cmd = 0

    ## Check if input data is initialized. ##
    def isReady(self):
        if self.reference_path is None:
            self.get_logger().info("The reference path data has not ready yet.")
            return False
        if self.current_accel is None:
            self.get_logger().info("The accel data has not ready yet.")
            return False
        if self.ego_pose is None:
            self.get_logger().info("The ego pose data has not ready yet.")
            return False
        if self.dynamic_objects is None:
            self.get_logger().info("The dynamic objects data has not ready yet.")
            return False
        if self.vehicle_inputs_raw is None:
            self.get_logger().info("The vehicle_inputs_raw data has not ready yet.")
            return False
        self.get_logger().info("autoware if node is ready!")
        return True

    ## Callback function for timer ##
    def onTimer(self):
        vehicle_inputs_msg = VehicleInputs()
        vehicle_inputs_msg.header = Header(stamp=self.get_clock().now().to_msg())
        vehicle_inputs_msg.throttle_cmd = float(self.throttle_cmd) * 120.0
        vehicle_inputs_msg.brake_cmd = float(self.brake_cmd) * 6000.0
        vehicle_inputs_msg.steering_cmd  = float(self.steering_cmd) # * 180.0 / 3.141 * 19.5
        vehicle_inputs_msg.gear_cmd  = int(self.gear_cmd)
        self.pub_cmd_.publish(vehicle_inputs_msg)

    ## Callback function for path subscriber ##
    def onPath(self, msg: Path):
        self.reference_path = msg
        self.reference_path_array = ConvertPath2Array(msg)
        ## Set left and right bound
        self.left_bound = ConvertPointSeq2Array(self.reference_path.left_bound)
        self.right_bound = ConvertPointSeq2Array(self.reference_path.right_bound)

    ## Callback function for accrel subscriber ##
    def onAcceleration(self, msg: AccelWithCovarianceStamped):
        # return geometry_msgs/Accel 
        self.current_accel = [msg.accel.accel.linear.x, msg.accel.accel.linear.y, msg.accel.accel.linear.z]

    ## Callback function for odometry subscriber ##
    def onOdometry(self, msg: Odometry):
        self.ego_pose      = msg.pose.pose
        self.current_vel_x = msg.twist.twist.linear.x
        self.current_vel_y = msg.twist.twist.linear.y
        # self.get_logger().info("current_vel {}, {}".format(self.current_vel_x, self.current_vel_y))

    ## Callback function for predicted objects ##
    def onPerception(self, msg: PredictedObjects):
        self.dynamic_objects = msg
        obj_info = PredictedObjectsInfo (self.dynamic_objects.objects)
        self.obj_pose = obj_info.objects_rectangle

    ## Callback function for ActuationCommand from autoware ##
    def onInputsRaw(self, msg: VehicleInputs):
        self.vehicle_inputs_raw = msg
        self.steering_cmd = msg.steering_cmd
        self.gear_cmd     = msg.gear_cmd
        if record_expart_data == True:
            self.throttle_cmd = msg.throttle_cmd / 100.0
            self.brake_cmd    = msg.brake_cmd / 6000.0