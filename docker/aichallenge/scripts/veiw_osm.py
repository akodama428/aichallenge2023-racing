import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# OSMデータをファイルから読み込む
osm_file = "lanelet2_map.osm"

# OSMデータを解析
tree = ET.parse(osm_file)
root = tree.getroot()

# ノードの座標と番号を格納するリスト
node_coordinates184 = []
node_numbers184 = []

node_coordinates185 = []
node_numbers185 = []


# ノードを取得し、座標をリストに追加する関数
def noderef_append(node_ref, node_coordinates, node_numbers):
    for node in root.findall('.//node'):
        if node.get('id') == node_ref:
            local_x = float(node.find(".//tag[@k='local_x']").get('v'))
            local_y = float(node.find(".//tag[@k='local_y']").get('v'))
            node_coordinates.append((local_x, local_y))
            node_id = int(node.get('id'))
            node_numbers.append(node_id)

for way in root.findall('.//way'):
    way_id = way.get('id')
    for nd in way.findall('nd'):
        node_ref = nd.get('ref')
        if way_id == "184":
            noderef_append(node_ref, node_coordinates184, node_numbers184)
        if way_id == "185":
            noderef_append(node_ref, node_coordinates185, node_numbers185)

# L184地図を描画
local_x_coords, local_y_coords = zip(*node_coordinates184)
# 各ノードをプロット
plt.plot(local_x_coords, local_y_coords, marker='o', linestyle='-', color='blue', label='L184')
# 各ノードの番号をテキストとして描画
for i, node_number in enumerate(node_numbers184):
    plt.text(local_x_coords[i], local_y_coords[i], str(node_number), fontsize=8, ha='center', va='bottom')

# L185地図を描画
local_x_coords, local_y_coords = zip(*node_coordinates185)
# 各ノードをプロット
plt.plot(local_x_coords, local_y_coords, marker='o', linestyle='-', color='red', label='L185')
# 各ノードの番号をテキストとして描画
for i, node_number in enumerate(node_numbers185):
    plt.text(local_x_coords[i], local_y_coords[i], str(node_number), fontsize=8, ha='center', va='bottom')

# x軸とy軸のスケールを等しくする
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('Local X')
plt.ylabel('Local Y')
plt.title('OSM Map')
plt.legend()
plt.grid()
plt.show()
