#ifndef GUARD_READ_H
#define GUARD_READ_H

#include <string>
#include <unordered_map>

using namespace std;
class Graph;

// 有向グラフ読み込み
void read_d_graph_from_text_file(string file_path, Graph& graph);

// 無向グラフ読み込み
void read_u_graph_from_text_file(string file_path, Graph& graph);

#endif // GUAD_READ_H