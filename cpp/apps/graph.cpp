#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <bits/stdc++.h>
#include "../include/graph.h"

using namespace std;

/* getter 関数 */


// const 参照の隣接リストを返す
const unordered_map<int, vector<int>>& Graph::get_adj_list(){
    return this->adj_list;
}

// ノード数を返す
const int Graph::get_number_of_nodes(){
    return this->node_list_set.size();
}

// エッジ数を返す
const int Graph::get_number_of_edges(){
    int number_of_edges = 0;

    // ノードが持つエッジ数を合計する
    for(const pair<int, vector<int>>& item : this->adj_list){
        number_of_edges += item.second.size();
    }

    return number_of_edges;
}

/* グラフの操作 */

// エッジ生成 有向グラフ
void Graph::d_add_edge(int n1, int n2){
    // adjacency_list に n1 -> n2 を追加(有向のため)
    this->adj_list[n1].push_back(n2);
    this->adj_list[n2];

    // 頂点リスト生成 (set)
    this->node_list_set.insert(n1);
    this->node_list_set.insert(n2);
}

// エッジ生成 無向グラフ
void Graph::u_add_edge(int n1, int n2){
    // adjacency_list に n1 -> n2 を追加(無向のため)
    this->adj_list[n1].push_back(n2);
    this->adj_list[n2].push_back(n1);

    // 頂点リスト生成 (set)
    this->node_list_set.insert(n1);
    this->node_list_set.insert(n2);
}

 // 着目ノードの次数
const int Graph::get_degree(int node_id){

    return this->adj_list[node_id].size();
}

// 着目ノードのランダムな隣接ノードを一つ取得
const int Graph::get_random_adjacent(int node_id){

     int degree = this->adj_list[node_id].size();
     if(degree == 0) return -1;
     else return this->adj_list[node_id].at((int)(rand() % degree));

}

// ノードリスト(vector)を取得
const vector<int> Graph::get_node_list(){

    vector<int> node_list_vector(this->node_list_set.begin(), this->node_list_set.end());

    return node_list_vector;
}


 /* FORA 実装 */

// RW した際の終点を記録 
const vector<int> Graph::get_random_walk_end_nodes(int src_id, long long walk_count, double alpha){
    vector<int> end_node_id_list;
    for (int i = 0; i < walk_count; i++){
        int current_node_id = src_id;
        while((double)rand()/RAND_MAX > alpha){
            if(current_node_id == -1) break;
            current_node_id = get_random_adjacent(current_node_id);
        }
        end_node_id_list.push_back(current_node_id);
    }

    return end_node_id_list;
}




// Forward Push
const pair<unordered_map<int, double>, unordered_map<int, double>> Graph::calc_ppr_by_fp(int src_id, long long walk_count, double alpha, double r_max_coef){
    unordered_map<int, double> ppr, residue;
    unordered_set<int> active_node_set;
    queue<int> active_node_queue;

    int src_degree = get_degree(src_id);
    active_node_set.insert(src_id);
    active_node_queue.push(src_id);
    residue.emplace(src_id, 1);
    while(active_node_queue.size() > 0){
        int node_id = active_node_queue.front();
        int node_degree = get_degree(node_id);
        active_node_queue.pop();
        active_node_set.erase(node_id);

        if (node_degree == 0){
            ppr[node_id] += alpha * residue.at(node_id);

            ppr[-1] += (1 - alpha) * residue.at(node_id);
        } else {
            vector<int> neigh_list = this->adj_list[node_id];
            for(int i = 0; i < node_degree; i++){
                int neigh_id = neigh_list[i];
                int neigh_degree = get_degree(neigh_id);
                residue[neigh_id] += (1 - alpha) * residue.at(node_id) / node_degree;
                if((residue.at(neigh_id) > r_max_func(neigh_degree, alpha, walk_count, r_max_coef)) && (active_node_set.count(neigh_id) == 0)){
                    active_node_set.insert(neigh_id);
                    active_node_queue.push(neigh_id);
                }
            }
            ppr[node_id] += alpha * residue.at(node_id);
        }
        residue[node_id] = 0;
    }

    return make_pair(ppr, residue);
}



// FORA
const unordered_map<int, double> Graph::calc_ppr_by_fora(int src_id, long long walk_count, double alpha, double r_max_coef=1){
    pair<unordered_map<int, double>, unordered_map<int, double>> tmp = calc_ppr_by_fp(src_id, walk_count, alpha, r_max_coef);
    unordered_map<int, double> ppr = tmp.first;
    unordered_map<int, double> residure = tmp.second;

    for (auto itr = residure.begin(); itr != residure.end(); itr++){
        int node_id = itr->first;
        double r_val = itr->second;
        if(r_val == 0) continue;

        int walk_count_i = (int)ceil(r_val * walk_count);
        vector<int> end_node_id_list = get_random_walk_end_nodes(node_id, walk_count_i, alpha);
        for (int end_node_id : end_node_id_list){
            if(ppr.count(end_node_id) == 0) ppr[end_node_id] = 0;
            ppr[end_node_id] += (double)r_val / walk_count_i;
        }
    }
    return ppr;
}

 // RWer 数を決める関数
long long Graph::calc_omega(double delta, double eps){

    int n = get_number_of_nodes();
    double Pf = 1.0/ static_cast<double>(n);
    //double eps = 0.1;

    double omega = (4 * log(1/Pf)) / ((eps * eps) * delta);

    return ceil(omega);
}

// delta を求める (α 以上)
double Graph::determine_delta(int src_id, double alpha){

    // source node の隣接ノード
    vector<int> neigh_list = this->adj_list[src_id];

    // source node の次数
    int src_id_degree = get_degree(src_id);

    // hamonic centrality の計算 ： source node の隣接ノード次数の逆数の総和
    double hamonic_c = 0;

    for (int neigh_id : neigh_list){
        int neigh_id_degree = get_degree(neigh_id);
        hamonic_c += 1.0/static_cast<double>(neigh_id_degree); 
    }

    // 公比
    double r = (((1 - alpha) * (1 - alpha)) / src_id_degree) * hamonic_c;

    double deno = 1 - r;

    double delta = alpha / deno;

    return delta;

}

 /* top k 還流度順位の精度保証のために用いる関数たち */
pair<double, double> Graph::calc_upper_and_lower_ppr(double ppr_val, double eps){

    double upper_ppr = ppr_val / (1 - eps);
    double lower_ppr = ppr_val / (1 + eps);

    return make_pair(upper_ppr, lower_ppr);
}

double Graph::determine_new_eps(double ppr_val, double eps, double d){

    double new_eps = (d - ppr_val * eps) / (ppr_val + d);
    //double new_eps = (d - ppr_val*eps*(1 - eps)/2)  / (ppr_val + d);


    return new_eps;
}

double Graph::calc_new_eps(double ppr_i, double ppr_i_1, double eps_i, double eps_i_1){

    double ppr_ratio = ppr_i_1 / ppr_i;
    double eps_ratio = (1 + eps_i) / (1 + eps_i_1);
    double new_eps = 1 - (ppr_ratio * eps_ratio);
    //double new_eps = 1 - ppr_ratio;

    return new_eps;
}



// 還流度 PR を計算
const unordered_map<int, double> Graph::calc_selfpr_by_fora(double alpha, double eps, double r_max_coef){

    int n = get_number_of_nodes();

    vector<int> node_list_vec = get_node_list();

    unordered_map<int, double> selfpr;

    for (int id : node_list_vec){
        selfpr[id] = 0;
    }

    for (int src_id : node_list_vec){
        double delta = 1/static_cast<double>(n);
        int walk_count = calc_omega(delta, eps);
        unordered_map<int, double> ppr = calc_ppr_by_fora(src_id, walk_count, alpha, r_max_coef);
        ppr[src_id] -= alpha;

        for (auto [id, ppr_val] : ppr){
            ppr[id] /= (1 - alpha);
            selfpr[id] += ppr[id] / static_cast<double>(n);
        }
    }



    return selfpr;
}
