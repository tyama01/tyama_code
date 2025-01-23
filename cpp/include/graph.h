#ifndef GRAPH_H_
#define GRAPH_H_

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <time.h>
#include <queue>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h>
// #define NDEBUG
#include <cassert>

using namespace std;
using namespace std;

class Graph{
    private:
        unordered_set<int> node_list_set;
        unordered_map<int, vector<int> > adj_list; // ノードid -> 隣接ノードのvector
        double r_max_func(int degree, double alpha, long long walk_count, double r_max_coef) const {
            return (double)degree * r_max_coef / (alpha * (long double)walk_count);
        }

    public:
        // getter 関数
        const unordered_map<int, vector<int> >& get_adj_list();
        const int get_number_of_nodes();
        const int get_number_of_edges();

        /* グラフの操作 */

        // エッジ追加　有向グラフ
        void d_add_edge(int n1, int n2);

        // エッジ追加 無向グラフ
        void u_add_edge(int n1, int n2);

        // 着目ノードの次数
        const int get_degree(int node_id);

        // 着目ノードのランダムな隣接ノードを一つ取得
        const int get_random_adjacent(int node_id);


        // ノードリスト(vector)を取得
        const vector<int> get_node_list();


        // FORA 実装
        const vector<int> get_random_walk_end_nodes(int src_id, long long walk_count, double alpha);
        const pair<unordered_map<int, double>, unordered_map<int, double> > calc_ppr_by_fp(int src_id, long long walk_count, double alpha, double r_max_coef);
        const unordered_map<int, double> calc_ppr_by_fora(int src_id, long long walk_count, double alpha, double r_max_coef);

        // RWer 数を決める関数
        long long calc_omega(double delta, double eps);

        // delta を求める (α 以上)
        double determine_delta(int src_id, double alpha);

        /* top k 還流度順位の精度保証のために用いる関数たち */

        // ppr値 の上限と下限を求める関数
        pair<double, double> calc_upper_and_lower_ppr(double ppr_val, double eps);

        // new_eps を決定する関数
        double determine_new_eps(double ppr_val, double eps, double d);

        // new_eps を計算する関数 パート2
        double calc_new_eps(double ppr_i, double ppr_i_1, double eps_i, double eps_i_1);




        // 還流度 PR を計算
        const unordered_map<int, double> calc_selfpr_by_fora(double alpha, double eps, double r_max_coef);


};

#endif // GURAD_GRAPH_H