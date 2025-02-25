# 頂点ID が連続でない場合に ID を振り直すのに使用するコード

import networkx as nx

# /usr/bin/python3 /Users/tyama/tyama_code/apps/update_index.py

# ノードの id を振り直すするためにマッピング
def rerabel_mapping(node_list):
    node_num = len(node_list)   # ノード数

    l = []
    for i in range(node_num):
        l.append(i)

    # 元グラフのノード id をキーに, [0, 1, 2, ...] をバリューに持つ辞書型を生成
    mapping_dict = dict(zip(node_list, l))

    return mapping_dict

# データセット PATH
PATH = './datasets/facebook.txt'

# 無向グラフの準備
G = nx.read_edgelist(PATH, nodetype=int)
G = nx.relabel_nodes(G, rerabel_mapping(list(G.nodes)))   # ノード id の付け直し

# これで id 0から振り直してるの確認できる.
#print(list(G.nodes))

nx.write_edgelist(G, "update.txt", data=False)




