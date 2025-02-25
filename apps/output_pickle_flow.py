# FLOW を pickle ファイルで出力

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

from sklearn.preprocessing import normalize

import pandas as pd

# /usr/bin/python3 /Users/tyama/tyama_code/apps/output_pickle_flow.py

# -------------------------- データ読み込み -------------------------
dataset_name = "facebook"
data_loader = DataLoader(dataset_name, is_directed=False)
G = data_loader.get_graph()
print(G) # グラフのノード数、エッジ数出力

Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
print(f"component num : {len(Gcc)}")
print("-----------------------------------")

node_list = list(G.nodes)
n = len(node_list)

edge_list = list(G.edges())

#print(edge_list)

#------------------------------------------------------------------

#------------------------------------------------------------------
flow_obj = FLOW(G)

# {node : {(エッジ) : 通過回数}}
flow_dic = {}

for v in node_list:
    flow_times = flow_obj.get_flow_times(src_node=v, count = 10000, alpha=0.15)
    flow_dic[v] = flow_times 
    
    

#print(flow_times)

#------------------------------------------------------------------

# -----------------------------------------------
# 結果出力

with open('../alpha_dir/' + dataset_name + '/flow_15.pkl', 'wb') as f:
    pickle.dump(flow_dic, f)
    
print("End")

# -----------------------------------------------
