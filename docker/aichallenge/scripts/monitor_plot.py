import pandas as pd
import matplotlib.pyplot as plt

# monitor.csvの読み込み
df = pd.read_csv('/aichallenge/aichallenge_ws/logs/monitor.csv', names=['r', 'l','t'])
df = df.drop(range(2)) # 1〜2行目の削除

# 報酬のプロット
x = range(len(df['r']))
y = df['r'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()

# エピソード長のプロット
x = range(len(df['l']))
y = df['l'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.show()