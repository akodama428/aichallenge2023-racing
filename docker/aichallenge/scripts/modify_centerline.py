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
N = 200

# ds184とds185を計算
ds184 = length184 / (N - 1)
ds185 = length185 / (N - 1)

# ライン上でds184/ds185分ずつ進んだ点をリサンプル点として生成
resampled_coords184 = [coords184[0]]
resampled_coords185 = [coords185[0]]

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

# L184の各リサンプル点から最も近いL185のリサンプル点を探索し、
# 最近傍のリサンプル点と線分の中点を緑点でプロット
for i in range(1, N):
    x0, y0 = resampled_coords184[i]
    min_distance = float('inf')
    closest_point = (0, 0)
    
    for j in range(1, N):
        x1, y1 = resampled_coords185[j]
        distance = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x1, y1)
    
    x_midpoint = (x0 + closest_point[0]) / 2
    y_midpoint = (y0 + closest_point[1]) / 2
    plt.plot(x_midpoint, y_midpoint, 'ko')

    # L185からL184に向かって線分を引く
    x_line = [resampled_coords184[i][0], x_midpoint]
    y_line = [resampled_coords184[i][1], y_midpoint]
    plt.plot(x_line, y_line, 'k--')

# L185の各リサンプル点から最も近いL184のリサンプル点を探索し、
# 最近傍のリサンプル点と線分の中点を緑点でプロット
for i in range(1, N):
    x0, y0 = resampled_coords185[i]
    min_distance = float('inf')
    closest_point = (0, 0)
    
    for j in range(1, N):
        x1, y1 = resampled_coords184[j]
        distance = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x1, y1)
    
    x_midpoint = (x0 + closest_point[0]) / 2
    y_midpoint = (y0 + closest_point[1]) / 2
    plt.plot(x_midpoint, y_midpoint, 'go')

    # L185からL184に向かって線分を引く
    x_line = [resampled_coords185[i][0], x_midpoint]
    y_line = [resampled_coords185[i][1], y_midpoint]
    plt.plot(x_line, y_line, 'g--')

# L184の元のラインを描画
local_x_coords184, local_y_coords184 = zip(*coords184)
plt.plot(local_x_coords184, local_y_coords184, marker='o', linestyle='-', color='blue')

# L185の元のラインを描画
local_x_coords185, local_y_coords185 = zip(*coords185)
plt.plot(local_x_coords185, local_y_coords185, marker='o', linestyle='-', color='red')

# x軸とy軸のスケールを等しくする
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('Local X')
plt.ylabel('Local Y')
plt.title('OSM Map (L184 and L185 Resampled with Midpoints and Vertical Lines)')
plt.grid()
plt.show()
