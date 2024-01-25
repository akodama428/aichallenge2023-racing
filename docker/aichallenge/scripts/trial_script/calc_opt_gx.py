import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

# ファイルパス
raceline_org = './modify_raceline_zero_start4.csv'
raceline = './output.csv'

# ファイルの読み込み
raceline_df_org = pd.read_csv(raceline_org)
raceline_df = pd.read_csv(raceline)

def CalcDist(traj):
    dist = []
    dist.append(0.0)
    for i in range(0, len(traj['x']) - 1):
        delta_dist = np.sqrt((traj['x'][i+1] - traj['x'][i])**2 + (traj['y'][i+1] - traj['y'][i])**2)
        dist.append(dist[i] + delta_dist)
    return dist
raceline_df_org['dist'] = CalcDist(raceline_df_org)
raceline_df['dist'] = CalcDist(raceline_df)

print(f"traj len:{len(raceline_df['V'])}")
# print(f"dist len:{len(raceline_df['dist'])}")

def CalcTime(traj):
    time = []
    time.append(0.0)
    for i in range(0, len(traj['x']) - 1):
        if traj['V'][i] < 0.01:
            traj['V'][i] = 0.01
        delta_time = (traj['dist'][i+1] - traj['dist'][i]) / traj['V'][i]
        time.append(time[i] + delta_time)
    return time
raceline_df['time'] = CalcTime(raceline_df)

# アクセルブレーキマップの読み込み
accel_map = './accel_map.csv'
brake_map = './brake_map.csv'
accel_map_df = pd.read_csv(accel_map, header=None, index_col=0)
accel_map_df = accel_map_df.transpose()
brake_map_df = pd.read_csv(brake_map, header=None, index_col=0)
brake_map_df = brake_map_df.transpose()

# # ここから最適化
from casadi import *
import casadi as ca

def calcMaxAccel(velocity):
    max_accel = accel_map_df['1.0'][1]
    for i in range(1, len(accel_map_df['default'])):
        if velocity > accel_map_df['default'][i]:
            i +=1
            max_accel = accel_map_df['1.0'][i]
            continue
        else:
            break
    # print(f'max_accel: {max_accel}')
    return max_accel
def calcMaxBrake(velocity):
    max_brake = brake_map_df['1.0'][1]
    for i in range(1, len(brake_map_df['default'])):
        if velocity > brake_map_df['default'][i]:
            i +=1
            max_brake = brake_map_df['1.0'][i]
            continue
        else:
            break
    # print(f'max_brake: {max_brake}')
    return max_brake
def calcCurvature(x,y):
    curvatures = [0.0]
    for i in range(1, len(x)-1):
        dxn = x[i] - x[i-1]
        dxp = x[i+1] - x[i]
        dyn = y[i] - y[i-1]
        dyp = y[i+1] - y[i]
        dn = np.hypot(dxn, dyn)
        dp = np.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp *dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curv = (ddy * dx - ddx * dy) /((dx**2 + dy**2)**1.5)
        curvatures.append(curv)
    curvatures.append(curv)
    return curvatures

from scipy import signal
def filtfiltpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "low")
    y = signal.filtfilt(b,a,x)
    return y

samplerate = 1 / 0.016
fp = 3.0
fs = 20.0
gpass = 3
gstop = 40
curvurvatures = calcCurvature(raceline_df['x'], raceline_df['y'])
curvurvatures = filtfiltpass(curvurvatures, samplerate, fp, fs, gpass, gstop) 

nx = 2      # 状態空間の次元[v , t]
nu = 1      # 制御入力の次元[a]
gravity = 9.81  # 重力加速度

# 以下で非線形計画問題(NLP)を定式化
w = []    # 最適化変数を格納する list
w0 = []   # 最適化変数(w)の初期推定解を格納する list
lbw = []  # 最適化変数(w)の lower bound を格納する list
ubw = []  # 最適化変数(w)の upper bound を格納する list
J = 0     # コスト関数 
g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
lbg = []  # 制約関数(g)の lower bound を格納する list
ubg = []  # 制約関数(g)の upper bound を格納する list

Xk = MX.sym('X0', nx) # 初期時刻の状態ベクトル x0
w += [Xk]             # x0 を 最適化変数 list (w) に追加
# 初期状態は given という条件を等式制約として考慮.ただし、距離ベースで計算しているので、初期車速0だと計算できないため1.0m/sを初期車速とする
lbw += [1.0, 0.]  # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
ubw += [1.0, 0.]  # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
w0 +=  [1.0, 0.]  # x0 の初期推定解

# 離散化ステージ 0~N-1 までのコストと制約を設定
for k in range(len(raceline_df['dist'])-1):
    Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
    v = Xk[0]     # 車速[m/s]
    t = Xk[1]     # 走行時間[sec]
    a = Uk[0]     # 加速度[m/ss]（制御入力）
    w += [Uk]     # uk を最適化変数 list に追加

    # 入力制約
    # 以下のような制約側で変数を使うことはできない
    ### lbw += [brake_coeff[0]*v**2 + brake_coeff[1]*v + brake_coeff[2]]
    ### ubw += [accel_coeff[0]*v**2 + accel_coeff[1]*v + accel_coeff[2]]
    # 仕方がないので、地図の車速からマップ引きする。何回かマップ更新すれば、正しい値に収束す
    map_v = raceline_df['V'][k]
    lbw += [calcMaxBrake(map_v)]
    ubw += [calcMaxAccel(map_v)]
    w0  += [0]  # uk の初期推定解    
    # ステージコストは0 
    J = 0
    # ダイナミクス 
    d_dist = raceline_df['dist'][k+1] - raceline_df['dist'][k]     # 距離[m]
    Xk_next = vertcat(ca.sqrt(ca.fmax(v**2 + 2 * a * d_dist, 0.1)), 
                      t + d_dist / v)
    Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
    w   += [Xk1]                       # xk+1 を最適化変数 list に追加
    lbw += [1.0, 0.0]    # xk+1 の lower-bound （指定しない要素は -inf）
    ubw += [inf, inf]    # xk+1 の upper-bound （指定しない要素は inf）
    w0 += [1.0, 0.0]     # xk+1 の初期推定解

    # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
    g   += [Xk_next-Xk1]
    lbg += [0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    ubg += [0, 0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定

    # Gの制約
    curv = curvurvatures[k]
    g  += [1 - (a/gravity)**2 - (curv*v**2/gravity)**2]
    lbg += [0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    ubg += [inf] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定

    Xk = Xk1

# 終端コスト
Vf = Xk[1]  # 最終時刻を終端コストとして、最速で走るプランを生成する
J = J + Vf  # コスト関数に終端コストを追加

# 非線形計画問題(NLP)
nlp = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)} 
# Ipopt ソルバー，最小バリアパラメータを0.001に設定
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt':{'mu_min':0.001}}) 
# SQP ソルバー（QPソルバーはqpOASESを使用），QPソルバーの regularization 無効，QPソルバーのプリント無効
# solver = nlpsol('solver', 'sqpmethod', nlp, {'max_iter':100, 'qpsol_options':{'enableRegularisation':False, 'printLevel':None}})

# NLPを解く
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()
v_opt = w_opt[0::3]
t_opt = w_opt[1::3]
a_opt = w_opt[2::3]

# CSVファイルとしてエクスポートする
raceline_df['V'] = v_opt
raceline_df['a'][:-1] = a_opt
export_path = 'output.csv'
raceline_df.to_csv(export_path, index=False)

plt.figure(1)
plt.plot(raceline_df_org['dist'], raceline_df_org['V'], '--')
plt.plot(raceline_df['dist'], raceline_df['V'], '-')
plt.xlabel('dist')
plt.legend(['v_org','v'])
plt.grid()

plt.figure(2)
plt.plot(t_opt, v_opt, '--')
plt.plot(t_opt[:-1], a_opt, '-')
plt.xlabel('t')
plt.legend(['v_opt','a_opt'])
plt.grid()
plt.show()