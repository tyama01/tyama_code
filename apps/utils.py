import networkx as nx 
import random
from queue import Queue
import numpy as np
import itertools
import math
from collections import deque



#------------------------------------------------------------------

# データセット読み込みのクラス
class DataLoader:
    def __init__(self, dataset_name, is_directed):
        self.dataset_name = dataset_name
        self.is_directed = is_directed
        self.c_id = {}
        self.id_c = {}
        
        # 有向グラフ
        if is_directed:
            self.G = nx.DiGraph()
            dataset_path = "../datasets/" + self.dataset_name + ".txt"
            self.G = nx.read_edgelist(dataset_path, nodetype=int, create_using=nx.DiGraph)
            
            
        # 無向グラフ    
        else:
            self.G = nx.Graph()
            dataset_path = "../datasets/" + self.dataset_name + ".txt"
            self.G = nx.read_edgelist(dataset_path, nodetype=int)
            
        self.node_list = list(self.G.nodes())
        self.edge_list = list(self.G.edges())
            
    def get_graph(self):
        return self.G
    
    # Louvain のコミュニティを読み込む時に使用
    # def load_community(self):
    #     community_path = "../datasets/" + self.dataset_name + "_louvain.txt"
    #     with open(community_path, 'r') as f:
    #         lines = f.readlines()
    #     for line in lines:
    #         data = line.split()
    #         self.c_id.setdefault(int(data[0]), []).append(int(data[1]))
    #         self.id_c[int(data[1])] = int(data[0])
            
    # def get_communities(self):
    #     return self.c_id, self.id_c
    
   
    
#------------------------------------------------------------------

#------------------------------------------------------------------
# 還流度によるエッジ RW 流量比を計算
class FLOW:
    def __init__(self, G):
        self.G = G 
    
    # PPR した際の経路
    def get_paths(self, source_node, count, alpha):
        paths = list()
        #node_list = list(self.G.nodes)         
            
        for _ in range(count):
            current_node = source_node
            path = [source_node]
            while True:
                if random.random() < alpha:
                    break
                neighbors = list(self.G.neighbors(current_node))
                
                if(len(neighbors) == 0): # 有向エッジがない場合はランダムジャンプ
                    current_node = source_node
                    break
                else:   
                    random_index = random.randrange(len(neighbors))
                    current_node = neighbors[random_index]
                    path.append(current_node)
            paths.append(path)
            
        return paths
    
    
            
        
    def get_flow_times(self, src_node, count, alpha):
        
        # ソースノードの隣接ノード
        adj_node_list = list(self.G.neighbors(src_node))
        
        # 隣接ノード -> ソースノードに流入した回数を記録
        flow_times = {(adj_node, src_node) : 0 for adj_node in adj_node_list}
        
        paths = self.get_paths(src_node, count, alpha)
        
        for path in paths:
            for hop_num in range(len(path)):
                if (hop_num > 1 and path[hop_num] == src_node):
                    flow_times[(path[hop_num - 1], src_node)] += 1
                    
        return flow_times
                
        
#------------------------------------------------------------------

#------------------------------------------------------------------

# エッジ還流度 演算

class EPPR:
    def __init__(self, G):
        self.G = G
        self.edge_list = list(self.G.edges())    
        
    # 無向エッジ還流度
    def calc_edge_selfppr(self, node_selfppr, flow):
        
        edge_selfppr = {}
        
        for edge in self.edge_list:
            
            edge_selfppr_val = 0
            
                
            
            # edge (0)
            
            total_flow_val_0 = sum(flow[edge[0]].values())
            flow_ratio_0 = flow[edge[0]][(edge[1], edge[0])] / total_flow_val_0
            edge_selfppr_val += node_selfppr[edge[0]] * flow_ratio_0
            
            # edge (1)
            
            total_flow_val_1 = sum(flow[edge[1]].values())
            flow_ratio_1 = flow[edge[1]][(edge[0], edge[1])] / total_flow_val_1
            edge_selfppr_val += node_selfppr[edge[1]] * flow_ratio_1
                
            
            
           
            edge_selfppr[edge] = edge_selfppr_val
            
            
        return edge_selfppr            
    
   
    # 重み付き無向エッジ還流度
    def calc_flow_edge_selfppr(self, node_selfppr, flow):
        
        edge_selfppr = {}
        
        for edge in self.edge_list:
            
            edge_selfppr_val = 0
            
                
            
            # edge (0)
            
            total_flow_val_0 = sum(flow[edge[0]].values())
            if(total_flow_val_0 != 0):
                flow_ratio_0 = flow[edge[0]][(edge[1], edge[0])] / total_flow_val_0
                edge_selfppr_val += node_selfppr[edge[0]] * flow_ratio_0 * self.G.degree(edge[1])
            else:
                edge_selfppr_val += 0
            #edge_selfppr_val += node_selfppr[edge[0]] * flow_ratio_0 * node_selfppr[edge[0]]
            
            
            # edge (1)
            
            total_flow_val_1 = sum(flow[edge[1]].values())
            if(total_flow_val_1 != 0):
                flow_ratio_1 = flow[edge[1]][(edge[0], edge[1])] / total_flow_val_1
                edge_selfppr_val += node_selfppr[edge[1]] * flow_ratio_1 * self.G.degree(edge[0])
                
            else:
                edge_selfppr_val += 0
            #edge_selfppr_val += node_selfppr[edge[1]] * flow_ratio_1 * node_selfppr[edge[1]]
            
                
            
            
           
            edge_selfppr[edge] = edge_selfppr_val
            
            
        return edge_selfppr
      
#------------------------------------------------------------------

#------------------------------------------------------------------
# 幅優先探索 BFS
class BFS:
    def __init__(self, G, edge_selfppr):
        self.G = G.copy() # 元グラフのコピー
        self.node_list = list(self.G.nodes)
        self.edge_selfppr = edge_selfppr.copy()
        
    def get_edge_selfppr_val(self, edge):
        
        try:
            return self.edge_selfppr[edge]
        except KeyError:
            return self.edge_selfppr[(edge[1], edge[0])] 
        
    def delete_edge_from_edge_selfppr(self, community):
        
        for edge in list(self.edge_selfppr.keys()):
            if (edge[0] in community or edge[1] in community):
                self.edge_selfppr.pop(edge)
                
        return None
                
        
        
    def calc_simple_bfs(self, src_node):
        
        # BFS のためのデータ構造
        dist_dict = {node : -1 for node in self.node_list} # 全ノードを「未訪問」に初期化
        que = Queue()
        
        # 初期化条件 (ソースノードを初期ノードとする)
        dist_dict[src_node] = 0
        que.put(src_node)
        
        # BFS 開始 (キューがからになるまで探索を行う)
        while not que.empty():
            node = que.get() # キューから先頭ノードを取り出す
            
            # グラフにノードが存在するか確認
            if node not in self.G:
                continue
            
            # node から辿れるノードを全て辿る
            adj_list = list(self.G.neighbors(node))
            if(len(adj_list) != 0):
                for adj_node in adj_list:
                    if (dist_dict[adj_node] != -1):
                        continue # 既に発見済みのノードは探索しない
                    
                    # 新たに発見した頂点について距離情報を更新してキューに追加
                    dist_dict[adj_node] = dist_dict[node] + 1
                    que.put(adj_node)
                    
        return dist_dict
    
     # BFS で探索したノードと同様にエッジの距離層を決める
    def find_single_community(self, G, alpha, node_selfppr, tolerance_ratio):
        
        if not self.edge_selfppr: # エッジ還流度テーブルが空の場合は終了
            return None
        
        # エッジ還流度が最大のエッジを取得
        edge_selfppr_sort_list = []
        for tmp in sorted(self.edge_selfppr.items(), key=lambda x:x[1], reverse=True):
            edge_selfppr_sort_list.append(tmp[0])
        
        max_edge = edge_selfppr_sort_list[0]
        #max_edge = max(self.edge_selfppr, key=self.edge_selfppr.get)
        
        # 最大還流度を持つエッジの端のうち、次数が高い方を開始ノードとする
        start_node = max_edge[0] if self.G.degree[max_edge[0]] >= self.G.degree[max_edge[1]] else max_edge[1]
        
        
        # グラフ内にノードが存在しない場合のエラーを回避
        if start_node not in self.G:
            return None
        
        print(f"start_node : {start_node}")
        
        
        # BFSで各ノードの距離を取得
        distance_from_start = self.calc_simple_bfs(start_node)
        max_distance = max(distance_from_start.values())
        
        # {距離k層: [ノードリスト]}形式でノードを層別に分類
        nodes_by_layer = {dist: [node for node, dist_val in distance_from_start.items() if dist_val == dist] for dist in range(max_distance + 1)}
        
        # 各距離層ごとのエッジを保存する辞書
        edges_by_layer = {k: [] for k in range(max_distance + 1)}
        
        # 初期層(距離0)に接続するエッジを追加
        for neighbor in self.G.neighbors(start_node):
            edges_by_layer[0].append((start_node, neighbor))
            
            
        # 距離層間を比較して極小値をとるエッジを削除するアルゴリズム    
        for layer in range(max_distance):
            
            # k+1層のエッジ (この時はk層のエッジも含まれうる)
            next_layer_edges = []
            for node in nodes_by_layer[layer+1]:
                for neighbor in self.G.neighbors(node):
                    next_layer_edges.append((node, neighbor))
                    
            # 重複しないエッジのみを次層に追加
            for edge in next_layer_edges:
                if edge not in edges_by_layer[layer] and (edge[1], edge[0]) not in edges_by_layer[layer]:
                    edges_by_layer[layer + 1].append(edge)
                    
        
            # k 層のエッジのエッジ還流度を昇順にソートしたエッジリスト 
            #リスト型 [(エッジ, edgeselfppr_val)]
            
            edges_by_layer_dic = dict()
            
            for edge in edges_by_layer[layer]:
                edges_by_layer_dic[edge] = self.get_edge_selfppr_val(edge)
                #print(edges_by_layer_dic[edge])
                 
            
            edges_by_layer_sort = sorted(edges_by_layer_dic.items(), key=lambda x:x[1])
            
            
            
            
            # エッジ還流度の減少から増加に変わるポイントを探索
            for tmp_key in edges_by_layer_sort:
                edge_in_layer = tmp_key[0]
                edge_selfppr_val = self.get_edge_selfppr_val(edge_in_layer)
                
                if edge_selfppr_val is None:
                        continue
                    
                
                allowed_increase = edge_selfppr_val * tolerance_ratio
                
                for edge_in_next_layer in edges_by_layer[layer + 1]:
                    next_edge_selfppr_val = self.get_edge_selfppr_val(edge_in_next_layer)
                    
                    if next_edge_selfppr_val is None:
                        continue
                    
                    if next_edge_selfppr_val > (edge_selfppr_val + allowed_increase):
                        
                        if self.G.degree(edge_in_layer[0]) == 1 or self.G.degree(edge_in_layer[1]) == 1:
                            continue
                        
                        # 還流度の変化が許容範囲を超えたエッジをカット
                        self.G.remove_edges_from([edge_in_layer])
                        
                        # 連結成分を探索し、src_nodeを含む成分をコミュニティとする
                        largest_components = sorted(nx.connected_components(self.G), key=len, reverse=True)
                        
                    
                        
                        community = list() # 欲しいコミュニティ
                        component_list = list() # 非連結成分
                        
                        for component in largest_components[1:]:
                            if start_node in component:
                                community = list(component)
                            else:
                                for node in list(component):
                                    component_list.append(node)
                                
                        if(len(community) != 0):
                            if(len(component_list) != 0):
                                for node in component_list:
                                    community.append(node)
                                
                                return community
                            else:   
                                return community
                        else:
                            if(len(component_list) != 0):
                                self.G.add_edges_from([edge_in_layer])

                        break
                
        return None
    
    # 全ノードに対するコミュニティを抽出し、モジュラリティを計算
    def extract_all_communities(self, G, alpha, node_selfppr, tolerance_ratio):
        communities = []
        
        
        while self.edge_selfppr:
            community = self.find_single_community(G, alpha, node_selfppr, tolerance_ratio)
            
            if community:
                communities.append(community)
                
                # 事前に削除対象のノードをリスト化してコピー
                nodes_to_remove = list(community)
                
                
                self.G.remove_nodes_from(nodes_to_remove)
                
                self.delete_edge_from_edge_selfppr(community)
            
            else: 
                break
            
        communities.append(list(self.G.nodes))           
            
        return communities
                
               
            
#------------------------------------------------------------------

#------------------------------------------------------------------
# モジュラリティ計算
                    
class MOD:
    def __init__(self, G):
        """
        初期化: グラフ全体のエッジ数を計算
        :param G: networkx グラフ
        """
        self.G = G
        self.edge_num = self.G.size()  # グラフのエッジ数
        self.total_degree = 2 * self.edge_num  # グラフ全体の合計次数 (2 * エッジ数)

    def calc_mod_per_com(self, c_id):
        """
        各コミュニティのモジュラリティを計算する。
        :param c_id: {コミュニティID: ノードリスト} の辞書
        :return: {コミュニティID: モジュラリティ} の辞書
        """
        q_per_com_dic = dict()  # 結果を格納する辞書

        for com_id, nodes in c_id.items():
            # コミュニティ内のサブグラフを取得
            H = self.G.subgraph(nodes)

            # コミュニティ内エッジ数 (L_c)
            intra_edges = H.size()  # コミュニティ内のエッジ数

            # コミュニティ内ノードの次数合計 (k_c)
            total_degree_in_com = sum(self.G.degree(node) for node in nodes)

            # モジュラリティ計算
            Q_per_com = (intra_edges / self.edge_num) - (total_degree_in_com ** 2) / (self.total_degree ** 2)

            # モジュラリティスコアを格納
            q_per_com_dic[com_id] = Q_per_com

        return q_per_com_dic
                


#------------------------------------------------------------------

              
                
            