#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <sstream>
#include "../include/graph.h"


using namespace std;

// 指定したファイルの読み込み 有向グラフ
void read_d_graph_from_text_file(string file_path, Graph& graph){
    ifstream ifs(file_path);
    if(!ifs){
        cout << "Failed to open file" << endl;
        exit(-1);
    } else {
        for(int n1, n2; ifs >> n1 >> n2;){
            graph.d_add_edge(n1, n2);
        }
    }
}

// 指定したファイルの読み込み　無向グラフ
void read_u_graph_from_text_file(string file_path, Graph& graph){
    ifstream ifs(file_path);
    if(!ifs){
        cout << "Failed to open file" << endl;
        exit(-1);
    } else {
        for(int n1, n2; ifs >> n1 >> n2;){
            graph.u_add_edge(n1, n2);
        }
    }
}
