
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

import sys
import os
sys.path.append(os.path.dirname(__file__))
from util import *
from predicted_objects_info import PredictedObjectsInfo
# from myenv import *

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import A2C, PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
# 模倣学習用
from imitation.data.types import Trajectory as Ts
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data import serialize
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env, save_policy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

import numpy as np
import threading
import subprocess
import psutil
import shutil
from time import sleep
import json
import pickle

# エキスパートのデータ保存中は、VehicleInputsを受信して、トラジェクトリとして保存する
record_expart_data = False
MAX_EPISODE_LEN = 2200
# expert_data_path = "/aichallenge/output/expert_data_only_edge_x"
expert_data_path = "/aichallenge/output/expert_data_acc_cmd2"
output_policy_path = "/aichallenge/output/expert_policy_v0.pt"
publish_target_gx = True

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

        # 更新周期
        planner_cycle = 0.03
        self.action_update_cycle = int(planner_cycle / self.autoware_if_node.ctl_period)
        print(f"action update cycle :{self.action_update_cycle}")

        self.path_sampling = 2
        self.path_length = int(120 / self.path_sampling)

        # 行動空間
        if publish_target_gx == True:
            # [target gx]
            max_cmd = 100.0
            min_cmd = -100.0
            self.action_space = gym.spaces.Box(low=min_cmd, high=max_cmd, shape=(1,), dtype=np.float32)
        else:
            # [throttle_cmd, brake_cmd]
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
        # max_value = np.append(max_value, np.ones(self.path_length)*200) ## コース左端x座標 
        max_value = np.append(max_value, np.ones(self.path_length)*10) ## コース左端y座標 
        # max_value = np.append(max_value, np.ones(self.path_length)*200) ## コース右端x座標 
        max_value = np.append(max_value, np.ones(self.path_length)*5) ## コース右端y座標 
        # print(f"max value length:{len(max_value)}")

        min_value = np.array([min_x, min_y, -5]) ## x座標、y座標、車速 
        # min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース左端x座標 
        min_value = np.append(min_value, np.ones(self.path_length)*(-5)) ## コース左端y座標 
        # min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース右端x座標 
        min_value = np.append(min_value, np.ones(self.path_length)*(-10)) ## コース右端y座標
        # print(f"min value length:{len(min_value)}")

        # self.observation_space = gym.spaces.Box(low=min_value, high=max_value, shape=(3+self.path_length*4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=min_value, high=max_value, shape=(3+self.path_length*2,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(low=min_value, high=max_value, shape=(3,), dtype=np.float32)

        # カウンタ
        self.step_count = 0
        self.episode_count = 0
        self.max_episode_len = MAX_EPISODE_LEN

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
            if publish_target_gx == True:
                self.autoware_if_node.acc_cmd = action[0]
            else:
                self.autoware_if_node.throttle_cmd = action[0]
                self.autoware_if_node.brake_cmd    = action[1]

        # Autowareとの同期をとる。actionコマンドが一定周期publishされると更新する
        while self.autoware_if_node.publish_action_count < int(self.action_update_cycle):
            sleep(0.001)
        if self.autoware_if_node.publish_action_count != self.action_update_cycle:
            print(f"step cycle delay error! pulish action count: {self.autoware_if_node.publish_action_count}")
        self.autoware_if_node.publish_action_count = 0  # リセット

        # 状態量の取得
        state = self.get_state()
        # print(f"state length:{len(state)}")

        # 報酬の計算
        velocity = self.autoware_if_node.get_velocity()
        # 車速に応じて報酬を与える
        reward = velocity
        # ただし、車速が30kph以下で減速した場合はペナルティを与える
        if velocity < 10.0:
            if publish_target_gx == True:
                if self.autoware_if_node.acc_cmd < 0.0:
                    reward -= 50.0
            else:
                if self.autoware_if_node.brake_cmd >= 0.0:
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
        terminated = (self.step_count > self.max_episode_len)
        # we do not limit the number of steps here
        truncated = False
        # Optionally we can pass additional info, we are not using that for now
        info = {}

        if record_expart_data == True:
            self.actions.append(self.get_action())
            self.observations.append(state)
            self.infos.append(info)

        if terminated:
            self.episode_count += 1
            if (record_expart_data == True) and (reward > 0.0) :  # 途中で衝突したときは、トラジェクトリを保存しない
                ts = Ts(obs=np.array(self.observations), acts=np.array(self.actions), infos=np.array(self.infos), terminal = False)
                self.trajectorys.append(ts)
                serialize.save(expert_data_path, self.trajectorys)
                print(f"episode{self.episode_count} :add trajectory!")

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
        if self.step_count % 10 == 0:
            if publish_target_gx == True:
                print(f"step:{self.step_count}, action:{self.autoware_if_node.acc_cmd:.1f}, reward:{reward:.1f}")
            else:
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
        # self.shutdownSimulation()
        print("close!")
        serialize.save(expert_data_path, self.trajectorys)
        # with open(expert_data_path, mode="wb") as f:
        #     pickle.dump(self.trajectorys, f)
        print("finish to make trajectory file!")

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
        # state = np.append(state,  left_bound[:state_path_length:self.path_sampling, 0])
        state = np.append(state,  left_bound[:state_path_length:self.path_sampling, 1])
        # state = np.append(state, right_bound[:state_path_length:self.path_sampling, 0])
        state = np.append(state, right_bound[:state_path_length:self.path_sampling, 1])

        # プロットは処理が重いので、デバッグ時以外は描画してはダメ。処理が追い付かない
        # self.plot_marker.plot_status(ego_pose = ego_pose_array,
        #                              object_pose = self.autoware_if_node.obj_pose,
        #                              left_bound  = self.autoware_if_node.left_bound,
        #                              right_bound = self.autoware_if_node.right_bound,
        #                              path=None,
        #                              path_index_left=None,
        #                              path_index_next_left=None,
        #                              path_index_right= None,
        #                              path_index_next_right=None,
        #                              rotation = True,
        #                              predicted_goal_pose=None,
        #                              predicted_trajectory=None,
        #                              curve_plot=None,
        #                              curve_forward_point=None,
        #                              curve_backward_point=None,
        #                              vis_point=None
        #                              )
        return state
    
    def get_action(self):
        if publish_target_gx == True:
            action = [0.0]
            action[0] = self.autoware_if_node.acc_cmd
        else:
            action = [0.0, 0.0]
            action[0] = self.autoware_if_node.throttle_cmd
            action[1] = self.autoware_if_node.brake_cmd
        return action

    def startSimulation(self):
        # Autowareを起動
        p_autoware = subprocess.Popen("exec " + "bash /aichallenge/run_autoware.sh", shell=True)
        sleep(1)
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
        sleep(3)

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
        self.ctl_period = 0.01
        self.create_timer(self.ctl_period, self.onTimer)
        ## Remapがうまくできないので、トピック名を直接設定
        # self.create_subscription(Path ,"/planning/scenario_planning/lane_driving/behavior_planning/path", self.onPath, 10)
        self.create_subscription(Path ,"/planning/scenario_planning/corner_path_planning/path", self.onPath, 10)
        self.create_subscription(AccelWithCovarianceStamped, "/localization/acceleration", self.onAcceleration, 10)
        self.create_subscription(Odometry, "/localization/kinematic_state", self.onOdometry, 10)
        self.create_subscription(PredictedObjects, "/perception/object_recognition/objects", self.onPerception, 10)
        if publish_target_gx == True:
            self.create_subscription(AckermannControlCommand, "/control/simple_pure_pursuit/control_cmd", self.onCommandRaw, 10)
            self.pub_cmd_ = self.create_publisher(AckermannControlCommand, "/control/command/control_cmd", 10) # launchファイルの一部を/control/rl_planner/control_cmdに修正すること
        else:
            self.create_subscription(VehicleInputs, "/rl_planner_inputs", self.onInputsRaw, 10)  # launchファイルの/vehicle_inputsを/rl_planner_inputsに修正すること
            self.pub_cmd_ = self.create_publisher(VehicleInputs, "/vehicle_inputs", 10)

        # IFの初期化
        self.reset_if()
        self.is_ready = False

        # CustomEnvとの同期用
        self.publish_action_count = 0

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
        self.control_cmd = None
        self.acc_cmd = 0.0
        self.vehicle_inputs_raw = None
        self.throttle_cmd = 0.0
        self.brake_cmd = 0.0
        self.steering_cmd = 0.0
        self.gear_cmd = 0

    ## Check if input data is initialized. ##
    def isReady(self):
        self.is_ready = False
        if self.reference_path is None:
            self.get_logger().info("The reference path data has not ready yet.")
            return self.is_ready
        if self.current_accel is None:
            self.get_logger().info("The accel data has not ready yet.")
            return self.is_ready
        if self.ego_pose is None:
            self.get_logger().info("The ego pose data has not ready yet.")
            return self.is_ready
        if self.dynamic_objects is None:
            self.get_logger().info("The dynamic objects data has not ready yet.")
            return self.is_ready
        if (publish_target_gx == True) and (self.control_cmd is None):
            self.get_logger().info("The acc_cmd data has not ready yet.")
            return self.is_ready
        if (publish_target_gx == False) and (self.vehicle_inputs_raw is None):
            self.get_logger().info("The vehicle_inputs_raw data has not ready yet.")
            return self.is_ready
        self.get_logger().info("autoware if node is ready!")
        self.is_ready = True
        return self.is_ready

    ## Callback function for timer ##
    def onTimer(self):
        if self.control_cmd is not None:
            if publish_target_gx == True:
                cmd = AckermannControlCommand()
                cmd = self.control_cmd
                cmd.longitudinal.acceleration = float(self.acc_cmd) *1.4
                self.pub_cmd_.publish(cmd)
        if self.vehicle_inputs_raw is not None:
            if publish_target_gx == False:
                vehicle_inputs_msg = VehicleInputs()
                vehicle_inputs_msg.header = Header(stamp=self.get_clock().now().to_msg())
                vehicle_inputs_msg.throttle_cmd = float(self.throttle_cmd) * 100.0 * 1.3
                vehicle_inputs_msg.brake_cmd = float(self.brake_cmd) * 6000.0
                vehicle_inputs_msg.steering_cmd  = float(self.steering_cmd) # * 180.0 / 3.141 * 19.5
                vehicle_inputs_msg.gear_cmd  = int(self.gear_cmd)
                self.pub_cmd_.publish(vehicle_inputs_msg)
        self.publish_action_count += 1

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
        # self.get_logger().info(f"gx:{self.current_accel[0]}, gy:{self.current_accel[1]}, total_g:{self.current_accel[0]**2+self.current_accel[1]**2}")

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

    ## Callback function for ActuationCommand from autoware ##((
    def onCommandRaw(self, msg: AckermannControlCommand):
        self.control_cmd = msg
        if publish_target_gx == True:
            if record_expart_data == True:
                self.acc_cmd = msg.longitudinal.acceleration

    def onInputsRaw(self, msg: VehicleInputs):
        self.vehicle_inputs_raw = msg
        self.steering_cmd = msg.steering_cmd
        self.gear_cmd     = msg.gear_cmd
        if publish_target_gx == False:
            if record_expart_data == True:
                self.throttle_cmd = msg.throttle_cmd / 100.0
                self.brake_cmd    = msg.brake_cmd / 6000.0


def main():
    rclpy.init()

    # ログフォルダの準備
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 250000,
        "env_name": "myenv-v0"
    }

    # 独自の環境をStable-Baselines3が扱えるように変換
    # def make_env():
    #     env = CustomEnv()
    #     env = Monitor(env, log_dir, allow_early_resets=True) # Monitorの利用
    #     return env

    # 模倣学習用環境
 
    def make_env():
        _env = CustomEnv()
        _env = Monitor(_env, log_dir, allow_early_resets=True) # Monitorの利用
        _env = TimeLimit(_env, max_episode_steps=MAX_EPISODE_LEN + 10)
        _env = RolloutInfoWrapper(_env)
        return _env

    env = DummyVecEnv([make_env])

    # If the environment don't follow the interface, an error will be thrown
    # check_env(env, warn=True)


    # エキスパートエピソードの保存
    if record_expart_data == True:
        record_episodes = 300
        for i in range(0, record_episodes):
            state = env.reset()
            while True:
                if publish_target_gx == True:
                    action = [0.0]
                else:
                    action = [0.0,0.0]
                state, reward, done, info = env.step(action)
                if done:
                    break
        env.close()
    
    # 学習アルゴリズム
    if record_expart_data == False:
        print(f"start imitation learning! observation_space:{env.observation_space} action_space:{env.action_space}")

        # 模倣学習
        SEED = 1
        N_RL_TRAIN_STEPS = 100_000
        rng = np.random.default_rng(SEED)
        # with open(expert_data_path, "rb") as f:
        #     trajectories = pickle.load(f)
        trajectories = serialize.load(expert_data_path)
        transitions = rollout.flatten_trajectories(trajectories)

        # 1. BC（これだけではうまくいかないけど、事前学習としては使える）
        bc_trainer = bc.BC(
            observation_space = env.observation_space,
            action_space      = env.action_space,
            demonstrations    = transitions,
            rng               = rng,
            device            = 'cpu',  # TODO:GPU使えるようにする。このせいでうまくいってない？？
        )
        bc_trainer.train(n_epochs=1)
        save_policy(bc_trainer.policy, output_policy_path)
        print("finish imitation learning!")

        # # 2.AIRL
        # learner = PPO(
        #     env=env,
        #     policy=MlpPolicy,
        #     batch_size=64,
        #     ent_coef=0.0,
        #     learning_rate=0.0005,
        #     gamma=0.95,
        #     clip_range=0.1,
        #     vf_coef=0.1,
        #     n_epochs=5,
        #     seed=SEED,
        # )
        # reward_net = BasicShapedRewardNet(
        #     observation_space = env.observation_space,
        #     action_space = env.action_space,
        #     normalize_input_layer = RunningNorm,
        # )
        # airl_trainer = AIRL(
        #     demonstrations = transitions,
        #     demo_batch_size = MAX_EPISODE_LEN,
        #     gen_replay_buffer_capacity = 512,
        #     n_disc_updates_per_round = 16,
        #     venv = env,
        #     gen_algo = learner,
        #     reward_net = reward_net,)
        # env.seed(SEED)
        # learner_rewards_before_training, _ = evaluate_policy(
        #     learner, env, 100, return_episode_rewards=True
        # )
        # airl_trainer.train(N_RL_TRAIN_STEPS)
        # env.seed(SEED)
        # learner_rewards_after_training, _ = evaluate_policy(
        #     learner, env, 100, return_episode_rewards=True)
        # save_policy(airl_trainer.policy, output_policy_path)
        
        # # 3. GAIL
        # learner = PPO(
        #     env=env,
        #     policy=MlpPolicy,
        #     batch_size=64,
        #     ent_coef=0.0,
        #     learning_rate=0.0004,
        #     gamma=0.95,
        #     n_epochs=5,
        #     seed=SEED,
        # )

        # BCで事前学習した結果を使用
        class CopyPolicy(ActorCriticPolicy):
            def __new__(cls, *args, **kwargs):
                return bc_trainer.policy
        learner = PPO(CopyPolicy, env, verbose=0)
        reward_net = BasicRewardNet(
            observation_space     = env.observation_space,
            action_space          = env.action_space,
            normalize_input_layer = RunningNorm,
        )
        gail_trainer = GAIL(
            demonstrations=transitions,
            demo_batch_size=MAX_EPISODE_LEN,
            gen_replay_buffer_capacity=512,
            n_disc_updates_per_round=8,
            venv=env,
            gen_algo=learner,
            reward_net=reward_net,
        )
        print(f"start gail training! observation_space:{env.observation_space} action_space:{env.action_space}")
        gail_trainer.train(total_timesteps = 400_000)  # Train for 800_000 steps to match expert.
        gail_trainer.gen_algo.save("gail_planner_ppo")

        # # 学習コード
        # model = PPO(config["policy_type"], env, verbose=1).learn(total_timesteps = config["total_timesteps"])
        # model.save("rl_planner_ppo")
        # print("learning complete!")


        # モデルのテスト
        # model = PPO.load("gail_planner_ppo")
        model = bc.reconstruct_policy(output_policy_path) # 模倣学習BCモデル
        state = env.reset()
        while True:
            # モデルの推論
            action, _ = model.predict(state)
            # 1step実行
            state, reward, done, info = env.step(action)
            # エピソード完了
            if done:
                break
    # while True:
    #     key = input("please input q-key to end:")
    #     if key == "q":
    #         break

    # 現在のPythonプロセス以外のすべてのプロセスをキル
    current_process = psutil.Process()
    print("current_process: {}".format(current_process))
    for process in psutil.process_iter(attrs=['pid', 'name']):
        if (process.info['name'] != "ros2") & (process.info['name'] != "rl_planner_cust") :
            print("eprocess kill: {}".format(process.info['name']))
            process.terminate()  # プロセスを終了させる

    rclpy.shutdown()

if __name__ == '__main__':
    main()