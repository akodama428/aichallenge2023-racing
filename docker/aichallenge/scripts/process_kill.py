import psutil

# 全てのプロセスをkillする
for process in psutil.process_iter(attrs=['pid', 'name']):
  process.terminate()

