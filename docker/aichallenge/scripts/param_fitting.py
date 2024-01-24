from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_idl, get_types_from_msg, register_types

# スクリプトの使い方
# 1. ログデータ取得
#    ros2 bag record -a
# 2. 以下に上記で取得したログデータ保存フォルダを指定して実行する


bag_file = '/aichallenge/scripts/rosbag2_2024_01_23-23_14_44'


# ローパスフィルタ関数
def apply_first_order_lowpass_filter(raw_data, cutoff_frequency):
    alpha = 1 / (2 * np.pi * cutoff_frequency)
    filtered_speed = [raw_data[0]]

    for i in range(1, len(raw_data)):
        filtered_speed.append(filtered_speed[-1] + alpha * (raw_data[i] - filtered_speed[-1]))

    return filtered_speed

# ムダ時間あり一次遅れ応答関数
def calculate_first_order_dynamic_with_delay(k, time_constant, input_time, input_signal, delay_time=0):
    output_steer = [0.0]  # 初期値
    i_delay = 0

    for i in range(len(input_time) - 1):
        # 遅れを考慮して次のインデックスを探す
        while i_delay < len(input_time) - 1 and input_time[i_delay] + delay_time <= input_time[i]:
            i_delay += 1
        if i == 1:
            print(i_delay)
        # delta_ctrを遅れを考慮して計算
        delta_ctr = k / time_constant * (input_signal[i_delay] - output_steer[i]) * (input_time[i_delay+1] - input_time[i_delay])
        output_steer.append(output_steer[i] + delta_ctr)

    result_df = pd.DataFrame({'Time': input_time, 'Output': output_steer})

    return result_df

# 車速から加速度を計算(timeが正しくスケーリングされてないのでうまく計算できない。。)
def calcAccFromVel(time, vel):
    acc_array = []
    for i in range(len(time) - 1):
        acc = (vel[i+1] - vel[i]) / (time[i+1] - time[i])
        acc_array.append(acc)
    # 最後の要素は同じ値をセット
    acc_array.append(acc)

    # LF適用
    acc_array = apply_first_order_lowpass_filter(acc_array, 3.0)
    return acc_array

# メッセージ定義を保持するための辞書
add_types = {}
# IDLファイルのリスト
idl_files = [
    '/autoware/install/autoware_auto_control_msgs/share/autoware_auto_control_msgs/msg/AckermannControlCommand.idl',
    '/autoware/install/autoware_auto_control_msgs/share/autoware_auto_control_msgs/msg/AckermannLateralCommand.idl',
    '/autoware/install/autoware_auto_control_msgs/share/autoware_auto_control_msgs/msg/LongitudinalCommand.idl',
    '/autoware/install/autoware_auto_vehicle_msgs/share/autoware_auto_vehicle_msgs/msg/SteeringReport.idl',
    '/opt/ros/humble/share/nav_msgs/msg/Odometry.idl',
    '/opt/ros/humble/share/geometry_msgs/msg/AccelWithCovarianceStamped.idl'
]
# 各IDLファイルから定義を取得し、辞書に追加
for idl_file in idl_files:
    idl_text = Path(idl_file).read_text()
    add_types.update(get_types_from_idl(idl_text))
# タイプをrosbagsのシリアライザ/デシリアライザで利用可能にする
register_types(add_types)

# データを格納するためのデータフレームを作成
str_cmd_data = {'cmd_time': [], 'tire_angle_cmd': [], 'tire_angle_rate_cmd':[], 'speed_cmd': [], 'acc_cmd': []}
str_data = {'str_time': [], 'str_time_delta': [], 'current_steer_ang': [], 'current_tire_ang_rate': [], 'filt_current_tire_ang_rate': []}
vel_data = {'vel_time': [], 'current_vel': [], 'yawrate': []}
acc_data = {'acc_time': [], 'current_ax': []}

df1 = pd.DataFrame(str_cmd_data)
df2 = pd.DataFrame(str_data)
df3 = pd.DataFrame(vel_data)
df4 = pd.DataFrame(acc_data)

# 初期の前回タイヤ角度を設定
prev_tire_angle = None
prev_time = None

# バッグファイルを開く
velocity = 28.0
with Reader(bag_file) as reader:
    for connection, timestamp, rawdata in reader.messages():
        time = timestamp * 10e-10
        if connection.topic == '/control/command/control_cmd':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            tire_angle_cmd = msg.lateral.steering_tire_angle
            tire_angle_rate_cmd = msg.lateral.steering_tire_rotation_rate
            speed_cmd = msg.longitudinal.speed
            acc_cmd = msg.longitudinal.acceleration
            df1 = df1.append({'cmd_time': time, 'tire_angle_cmd': tire_angle_cmd, 'tire_angle_rate_cmd': tire_angle_rate_cmd, 'speed_cmd': speed_cmd, 'acc_cmd': acc_cmd}, ignore_index=True)
        elif connection.topic == '/vehicle/status/steering_status':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            current_tire_angle = msg.steering_tire_angle
            # 初回の場合は速度を計算できないので、前回のタイヤ角度を更新
            if (prev_tire_angle is None) or (prev_time is None):
                prev_tire_angle = current_tire_angle
                prev_time = time
                continue
            # 前回との時間間隔を計算
            str_time_delta = time - prev_time
            # タイヤ角度の速度を計算
            current_tire_ang_rate = (current_tire_angle - prev_tire_angle) / str_time_delta
            # 前回の値を更新
            prev_tire_angle = current_tire_angle
            prev_time = time
            # データフレームに行を追加
            df2 = df2.append({'str_time': time, 'str_time_delta': str_time_delta, 'current_steer_ang': current_tire_angle, 'current_tire_ang_rate': current_tire_ang_rate}, ignore_index=True)
        elif connection.topic == '/localization/kinematic_state':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            velocity = msg.twist.twist.linear.x
            yawrate = msg.twist.twist.angular.z
            df3 = df3.append({'vel_time': time, 'current_vel': velocity, 'yawrate': yawrate}, ignore_index=True)
        elif connection.topic == '/localization/acceleration':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            accelx = msg.accel.accel.linear.x
            accely = msg.accel.accel.linear.y
            df4 = df4.append({'acc_time': time, 'current_ax': accelx, 'current_ay': accely}, ignore_index=True)
    
        if velocity < 1.0:
            break

# タイヤ角速度にLF適用
cutoff_frequency = 1.0
df2['filt_current_tire_ang_rate'] = apply_first_order_lowpass_filter(df2['current_tire_ang_rate'], cutoff_frequency)

# パラメータの設定
k = 1.0
time_constant = 0.05  # 時定数（単位：秒）
delay_time = 0.05  # 遅れ時間（単位：秒）
# 一次遅れ系の出力を計算
# result_df_dynamic = calculate_first_order_dynamic_with_delay(k, time_constant, df1['cmd_time'], df1['tire_angle_cmd'], delay_time)


# 各指令値に対する応答性確認
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
ax1.plot(df1['cmd_time'], df1['tire_angle_cmd'], label='Front Wheel Angle Command')
ax1.plot(df2['str_time'], df2['current_steer_ang'] / 19.4854, label='Current Steering Angle')
# ax1.plot(df1['cmd_time'], result_df_dynamic['Output'], label='1st delay model')
ax1.set_ylabel('Steering Angle (rad)')
ax1.legend()
ax1.grid(True)
ax2.plot(df1['cmd_time'], df1['tire_angle_rate_cmd'], label='Steering Angle Rate Command')
ax2.plot(df2['str_time'], df2['filt_current_tire_ang_rate'], label='Filtered Steering Angle Rate')
ax2.set_ylabel('Steering Angle Rate (rad/s)')
ax2.legend()
ax2.grid(True)
ax3.plot(df1['cmd_time'], df1['speed_cmd'], label='Velocity Command')
ax3.plot(df3['vel_time'], df3['current_vel'], label='Velocity')
ax3.set_ylabel('Velocity (m/s)')
ax3.legend()
ax3.grid(True)
# ax4.plot(df1['cmd_time'], df1['acc_cmd'], label='Accelaration Command')
ax4.plot(df4['acc_time'], df4['current_ax'], label='Ax')
ax4.plot(df4['acc_time'], df4['current_ay'], label='Ay')
# ax4.plot(df3['vel_time'], calcAccFromVel(df3['vel_time'], df3['current_vel']), label='calcAcc')
ax4.set_ylabel('Accelaration (m/s2)')
ax4.legend()
ax4.grid(True)

# 車速とACCをプロット
fig, ax = plt.subplots()
acc_map_array = []
vel_idx = 0
# acc_map作成時
# for i in range(np.min([len(df4['acc_time']), len(df3['vel_time'])])-1):
#     # accログとvelociyログの時間合わせ
#     j = i
#     while df3['vel_time'][j] < df4['acc_time'][i]:
#         j +=1
#     # 車速とACCのプロット
#     ax.scatter(df3['current_vel'][j], df4['current_ax'][i], color="blue")

#     if (df3['current_vel'][j] > 1.39 * vel_idx):
#         acc_map_array.append(df4['current_ax'])
#         # print("vel: {}, acc: {}".format(df3['current_vel'][j], df4['current_ax'][i]))
#         # print("{},".format(df3['current_vel'][j]), end="")
#         print("{},".format(df4['current_ax'][i]), end="")
#         vel_idx += 1
#         ax.scatter(df3['current_vel'][j], df4['current_ax'][i], color="green")
        
#     if (df4['current_ax'][i] < 0):
#         break

# brake_map作成時
for i in range(np.min([len(df4['acc_time']), len(df3['vel_time'])])-1):
    # accログとvelociyログの時間合わせ
    j = i
    while df3['vel_time'][j] < df4['acc_time'][i]:
        j +=1

    if (df4['current_ax'][i] > 0):
        continue

    if (df3['current_vel'][j] > 45):
        continue

    # 車速とACCのプロット
    ax.scatter(df3['current_vel'][j], df4['current_ax'][i], color="blue")

    if (df3['current_vel'][j] < 45 - 1.39 * vel_idx):
        acc_map_array.append(df4['current_ax'][i])
        # print("{},".format(df3['current_vel'][j]), end="")
        vel_idx += 1
        ax.scatter(df3['current_vel'][j], df4['current_ax'][i], color="green")
for i in range(len(acc_map_array)-1):
    print("{},".format(acc_map_array[len(acc_map_array)-1-i]), end="")

ax.set_xlabel('velocity')
ax.set_ylabel('accelaration')

# # 2輪モデルによる推定
# # 既知の値
# MASS = 815.11  # [kg]
# LF = 1.6785  # [m]
# LR = 1.2933  # [m]
# I = 1770  # [kgm^2]
# GEAR_RATIO = 19.4913
# # 適合値
# cf = 77747.0
# cr = 155494.663
# KF = cf / 2  # [N/rad]
# KR = cr / 2 # [N/rad]

# class TwoWheelModel:
#     def __init__(self):
#         self.x = np.array([[0.0], [0.0]])

#     # 状態量：[スリップ角、ヨーレート]
#     def forward(self, vel, delta, dt):
#         vel = np.max([vel, 0.1])
#         self.A = np.array([[-2.0 * (KF + KR) / (MASS * vel), -2.0 * (KF * LF - KR * LR) / (MASS * vel * vel) - 1.0],
#                            [-2.0 * (KF * LF - KR * LR) / I, -2.0 * (KF * LF * LF + KR * LR * LR) / (I * vel)]])
#         self.B = np.array([[2.0 * KF / (MASS * vel)], [2.0 * KF * LF / I]])
#         # x_dot = A @ self.x + B * delta
#         # 上記行列計算だとエラーが出たので、ベタで計算
#         x_dot = np.array([[0.0], [0.0]])
#         x_dot[0] = self.A[0, 0] * self.x[0] + self.A[0, 1] * self.x[1] + self.B[0] * delta
#         x_dot[1] = self.A[0, 1] * self.x[0] + self.A[1, 1] * self.x[1] + self.B[1] * delta
#         self.x += x_dot * dt

# model = TwoWheelModel()
# model_slip = np.array([])
# model_yawrate = np.array([])
# for i in range(np.min([len(df2['str_time']), len(df3['vel_time'])])-1):
#     # steerログとvelociyログの時間合わせ
#     j = i
#     while df3['vel_time'][j] < df2['str_time'][i]:
#         j +=1
#     # ２輪モデルでスリップ角・ヨーレートを推定
#     model.forward(df3['current_vel'][j], df2['current_steer_ang'][i] / GEAR_RATIO, df2['str_time_delta'][i])
#     # print(df2['str_time'][i] - df2['str_time'][0], model.x[0], model.x[1])
#     model_slip = np.append(model_slip, model.x[0])
#     model_yawrate = np.append(model_yawrate, model.x[1])

# # モデルと実車データの比較
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# ax1.plot(df2['str_time'][:-1], model_yawrate, label='yawrate(model)')
# ax1.plot(df3['vel_time'], df3['yawrate'], label='yawrate')
# ax1.set_ylabel('yaw rate[rad/s]')
# ax1.legend()
# ax2.plot(df2['str_time'][:-1], model_slip, label='slip angle(model)')
# ax2.set_ylabel('slip angle[rad]')
# ax2.legend()

# グラフを表示
plt.tight_layout()
plt.show()