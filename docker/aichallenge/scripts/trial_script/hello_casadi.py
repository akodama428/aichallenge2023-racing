import casadi

opti = casadi.Opti()

# 変数を定義
x1 = opti.variable()
x2 = opti.variable()

# 初期値を指定
opti.set_initial(x1, 3)
opti.set_initial(x2, 3)

# 目的関数を定義
obj = x1**2 + x2**2
opti.minimize(obj)

# 制約条件を定義
opti.subject_to( x1*x2 >= 1 )

# 変数の範囲を定義
opti.subject_to( opti.bounded(0, x1, 4) )
opti.subject_to( opti.bounded(0, x2, 4) )

opti.solver('ipopt') # 最適化ソルバを設定
sol = opti.solve() # 最適化計算を実行

print(f'評価関数：{sol.value(obj)}')
print(f'x1: {sol.value(x1)}')
print(f'x2: {sol.value(x2)}')