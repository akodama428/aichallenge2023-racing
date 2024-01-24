from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_idl, get_types_from_msg, register_types

# ローパスフィルタ関数
def apply_first_order_lowpass_filter(tire_angle_speed, cutoff_frequency):
    alpha = 1 / (2 * np.pi * cutoff_frequency)
    filtered_speed = [tire_angle_speed[0]]

    for i in range(1, len(tire_angle_speed)):
        filtered_speed.append(filtered_speed[-1] + alpha * (tire_angle_speed[i] - filtered_speed[-1]))

    return filtered_speed

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
str_data = {'str_time': [], 'current_tire_ang': [], 'current_tire_ang_rate': [], 'filt_current_tire_ang_rate': []}
vel_data = {'vel_time': [], 'current_vel': []}
acc_data = {'acc_time': [], 'current_acc': []}
replan_req_for_lane_departure_data = {'replan_req_for_lane_departure_time': [], 'replan_req_for_lane_departure': []}

df1 = pd.DataFrame(str_cmd_data)
df2 = pd.DataFrame(str_data)
df3 = pd.DataFrame(vel_data)
df3_2 = pd.DataFrame(acc_data)
df4 = pd.DataFrame(replan_req_for_lane_departure_data)

# 初期の前回タイヤ角度を設定
prev_tire_angle = None
prev_time = None

# バッグファイルを開く
bag_file = '/aichallenge/my_bag'
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
            # タイヤ角度の速度を計算
            current_tire_ang_rate = (current_tire_angle - prev_tire_angle) / (time - prev_time)
            # 前回の値を更新
            prev_tire_angle = current_tire_angle
            prev_time = time
            # データフレームに行を追加
            df2 = df2.append({'str_time': time, 'current_tire_ang': current_tire_angle, 'current_tire_ang_rate': current_tire_ang_rate}, ignore_index=True)
        elif connection.topic == '/localization/kinematic_state':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            velocity = msg.twist.twist.linear.x
            df3 = df3.append({'vel_time': time, 'current_vel': velocity}, ignore_index=True)
        elif connection.topic == '/localization/acceleration':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            accel = msg.accel.accel.linear.x
            df3_2 = df3_2.append({'acc_time': time, 'current_acc': accel}, ignore_index=True)
        elif connection.topic == '/replan_req_for_lane_departure':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            flag = msg.data
            df4 = df4.append({'replan_req_for_lane_departure_time': time, 'replan_req_for_lane_departure': flag}, ignore_index=True)

# タイヤ角速度にLF適用
cutoff_frequency = 1.0
df2['filt_current_tire_ang_rate'] = apply_first_order_lowpass_filter(df2['current_tire_ang_rate'], cutoff_frequency)

# サブプロットを作成
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

# データの描画
ax1.plot(df1['cmd_time'], df1['tire_angle_cmd'], label='Steering Angle Command')
ax1.plot(df2['str_time'], df2['current_tire_ang'], label='Current Steering Angle')
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

ax4.plot(df1['cmd_time'], df1['acc_cmd'], label='Accelaration Command')
ax4.plot(df3_2['acc_time'], df3_2['current_acc'], label='Accelaration')
ax4.set_ylabel('Accelaration (m/s2)')
ax4.legend()
ax4.grid(True)

ax5.plot(df4['replan_req_for_lane_departure_time'], df4['replan_req_for_lane_departure'], label='replan_req_for_lane_departure')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('flag')
ax5.legend()
ax5.grid(True)

# グラフを表示
plt.tight_layout()
plt.show()