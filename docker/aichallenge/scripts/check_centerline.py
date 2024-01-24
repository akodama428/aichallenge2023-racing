import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# OSMデータをファイルから読み込む
osm_file = "lanelet2_map.osm"

# OSMデータを解析
tree = ET.parse(osm_file)
root = tree.getroot()

# ラインの座標をリストに追加する関数
def add_line_coordinates(way_id):
    way_coords = []
    for way in root.findall('.//way'):
        if way.get('id') == way_id:
            for nd in way.findall('nd'):
                node_ref = nd.get('ref')
                for node in root.findall('.//node'):
                    if node.get('id') == node_ref:
                        local_x = float(node.find(".//tag[@k='local_x']").get('v'))
                        local_y = float(node.find(".//tag[@k='local_y']").get('v'))
                        way_coords.append((local_x, local_y))
    return way_coords

# L184とL185の座標を取得
coords184 = add_line_coordinates("184")
coords185 = add_line_coordinates("185")

# ライン全体の長さを計算
def calculate_length(coords):
    length = 0
    for i in range(1, len(coords)):
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        length += segment_length
    return length

length184 = calculate_length(coords184)
length185 = calculate_length(coords185)

# リサンプルポイント数 (N=100分割)
N = 100

# ds184とds185を計算
ds184 = length184 / (N - 1)
ds185 = length185 / (N - 1)

# ライン上でds184/ds185分ずつ進んだ点をリサンプル点として生成
resampled_coords184 = [coords184[0]]
resampled_coords185 = [coords185[0]]
midpoints = []

for i in range(1, N):
    target_length184 = i * ds184
    target_length185 = i * ds185
    
    # L184のリサンプル点を求める
    current_length = 0
    j = 1
    while current_length < target_length184 and j < len(coords184):
        x1, y1 = coords184[j-1]
        x2, y2 = coords184[j]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if current_length + segment_length >= target_length184:
            t = (target_length184 - current_length) / segment_length
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            resampled_coords184.append((x, y))
            current_length += segment_length
        else:
            current_length += segment_length
            j += 1
    
    # L185のリサンプル点を求める
    current_length = 0
    j = 1
    while current_length < target_length185 and j < len(coords185):
        x1, y1 = coords185[j-1]
        x2, y2 = coords185[j]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if current_length + segment_length >= target_length185:
            t = (target_length185 - current_length) / segment_length
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            resampled_coords185.append((x, y))
            current_length += segment_length
        else:
            current_length += segment_length
            j += 1

# リサンプルした座標をプロット
for i in range(N-1):
    plt.plot([resampled_coords184[i][0], resampled_coords185[i][0]], [resampled_coords184[i][1], resampled_coords185[i][1]], color='blue')

# 各線分の中点を計算し、赤点でプロット
for i in range(N-1):
    x_midpoint = (resampled_coords184[i][0] + resampled_coords185[i][0]) / 2
    y_midpoint = (resampled_coords184[i][1] + resampled_coords185[i][1]) / 2
    plt.plot(x_midpoint, y_midpoint, 'ro')

# x軸とy軸のスケールを等しくする
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('Local X')
plt.ylabel('Local Y')
plt.title('OSM Map (L184 and L185 Resampled with Midpoints)')
plt.grid()
plt.show()
