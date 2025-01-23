// 還流度 正規化した値で出力

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <utility>
#include <bits/stdc++.h> // iota
#include <time.h> // time
#include <chrono>
#include <thread>
#include "../include/graph.h"
#include "../include/read.h"

// Files to compile
// c++ output_selfppr_n.cpp graph.cpp read.cpp -std=gnu++17 -O3 -march=native -o a.out 

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char* argv[]){

    /* グラフ選択 */
    string graph_name;
    cout << "Enter graph name : ";
    cin >> graph_name;

    /* グラフのデータセットがあるか確認 */
    string dataset_path = "../../datasets/" + graph_name + ".txt"; 
    if(!fs::is_regular_file(dataset_path)){ // なければ異常終了
        cout << "There are no such datasets" << endl;
        return 1;
    }

    /* グラフ読み込み */
    Graph graph;

    // 無向グラフ
    read_u_graph_from_text_file(dataset_path, graph);
    cout << "Compleate reading graph" << endl;

    /* グラフ情報(ノード数, エッジ数) */
    int N = graph.get_number_of_nodes();

    // 無向グラフなのでエッジ数は半分
    int E = graph.get_number_of_edges()/2;

    cout << "Nodes : " << N << endl;
    cout << "Edges : " << E << endl;

    // パラメータ

    //int src_id = 0;
    //int walk_count = 10000;
    double alpha = 0.15;
    double r_max_coef = 1;
    double eps = 0.1;

    double self_ppr_sum = 0;

    vector<int> node_list_vector = graph.get_node_list(); 

    // self_ppr 値
    unordered_map<int, double> self_ppr;

    // Proposed 2
    auto start2 = chrono::system_clock::now();


    for (int src_id : node_list_vector){
        double delta = graph.determine_delta(src_id, alpha);
        int walk_count = graph.calc_omega(delta, eps);
        unordered_map<int, double> ppr = graph.calc_ppr_by_fora(src_id, walk_count, alpha, r_max_coef);
        self_ppr[src_id] = (ppr[src_id] - alpha) / (1 - alpha);
        self_ppr_sum += self_ppr[src_id];
    }

    auto end2 = chrono::system_clock::now();

    chrono::duration<double> elapsed2 = end2 - start2;

    cout << "Proposed_2 : " << elapsed2.count() << "sec" << endl;

    // map valueソート して出力
    typedef pair<int, double> pair;
    vector<pair> vec;

    copy(self_ppr.begin(), self_ppr.end(), back_inserter<vector<pair> >(vec));

    sort(vec.rbegin(), vec.rend(), [](const pair &l, const pair &r)
    {
        if(l.second != r.second){
            return l.second < r.second;
        }
        return l.first < r.first;
    });

    string result_path = "../../alpha_dir/" + graph_name + "/selfppr_15_01_n.txt";
    ofstream ofs;
    ofs.open(result_path);

    for(auto const &pair: vec){
        ofs << pair.first << " " << pair.second / self_ppr_sum << endl;
    }


    return 0;

}