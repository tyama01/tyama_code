# 提案手法のコミュニティ抽出 {com_id : Node_id}

from utils import *
import networkx as nx
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import matplotlib as mpl
from matplotlib import rcParams as rcp
from scipy.stats import kendalltau

import japanize_matplotlib

from sklearn.preprocessing import normalize

import pandas as pd

import statistics


# /usr/bin/python3 /Users/tyama/tyama_code/apps/output_com.py

# -------------------------- データ読み込み -------------------------

dataset_name = "facebook"



data_loader = DataLoader(dataset_name, is_directed=False)
G = data_loader.get_graph()
print(G) # グラフのノード数、エッジ数出力

# focus_id = 107
# print(f"ID {focus_id} : degree {G.degree[focus_id]}")
#------------------------------------------------------------------

#------------------------------------------------------------------
# ノード還流度読み込み
# self PPR 値を取得 {ノードID : Self PPR 値}
alpha = 15

print(f"alpha : {alpha}")

node_selfppr = {}
path = '../alpha_dir/' + dataset_name + '/selfppr_'+ str(alpha) + '_01_n.txt' # n の場合は正規化されている
with open(path) as f:
    for line in f:
        (id, val) = line.split()
        node_selfppr[int(id)] = float(val)
        
#print(node_selfppr)


#------------------------------------------------------------------

#------------------------------------------------------------------
# FLOW の結果読み込み


# self_ppr {src_node : {node_id : ppr 値}}

path = '../alpha_dir/' + dataset_name + '/flow_' + str(alpha) + '.pkl'
with open(path, 'rb') as f:
    flow_dic = pickle.load(f)
    
    
#------------------------------------------------------------------

#------------------------------------------------------------------
# エッジ還流度計算

eppr_obj = EPPR(G)
edge_selfppr = eppr_obj.calc_flow_edge_selfppr(node_selfppr=node_selfppr, flow=flow_dic)
print("End Calc Edge_selfPPR")
print("--------------------------------")


#print(edge_selfppr)

#------------------------------------------------------------------

#------------------------------------------------------------------
# コミュニティ抽出
bfs_obj = BFS(G, edge_selfppr=edge_selfppr)

# community = bfs_obj.find_single_community(G, node_selfppr=node_selfppr, tolerance_ratio=0.2)

# print(f"community size : {len(community)}")
# print(community)


communities = bfs_obj.extract_all_communities(G, alpha=alpha,node_selfppr=node_selfppr, tolerance_ratio=0.5)
print(f"com num : {len(communities)}")

# txt ファイルに出力
# {コミュニティID : 頂点ID}
output_file = "../com_dir/" + dataset_name + "/proposed_com_" + str(alpha) + ".txt"

with open(output_file, "w") as f:
    for com_id in range(len(communities)):
        for node_id in communities[com_id]:
            f.write("{}\t{}\n".format(com_id, node_id))
            
print("End")
    