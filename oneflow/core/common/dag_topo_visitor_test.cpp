#include "oneflow/core/common/dag_topo_visitor.h"
namespace oneflow {

namespace test {

namespace {

void TestTopo(const std::list<int>& start_nodes,
              const std::unordered_map<int, std::list<int>>& next_nodes,
              const std::list<int>& expected_topo_order,
              const std::function<bool(int)>& is_final) {
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    if (prev_nodes.find(id) == prev_nodes.end()) return;
    for (int x : prev_nodes.at(id)) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    if (next_nodes.find(id) == next_nodes.end()) return;
    for (int x : next_nodes.at(id)) { cb(x); }
  };

  DAGTopoVisitor<int> topo_for_each(foreach_next, foreach_prev);

  std::list<int> topo_list;
  topo_for_each(start_nodes, is_final, [&](int x) { topo_list.push_back(x); });
  ASSERT_TRUE(topo_list == expected_topo_order);
}

void TestTopo(const std::list<int>& start_nodes,
              const std::unordered_map<int, std::list<int>>& next_nodes,
              const std::list<int>& expected_topo_order) {
  TestTopo(start_nodes, next_nodes, expected_topo_order,
           [](int) { return false; });
}

}  // namespace

TEST(DAGTopoVisitor, one_vertix) { TestTopo({1}, {{1, {}}}, {1}); }

TEST(DAGTopoVisitor, diamond) {
  TestTopo({1}, {{1, {2, 3}}, {2, {4}}, {3, {4}}}, {1, 2, 3, 4});
}

TEST(DAGTopoVisitor, linked_list) {
  TestTopo({1}, {{1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}},
           {1, 2, 3, 4, 5, 6});
}

TEST(DAGTopoVisitor, linked_list_break_in_while) {
  int count = 0;
  TestTopo({1}, {{1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}}, {1, 2, 3},
           [&](int) { return ++count == 3; });
}

TEST(DAGTopoVisitor, multiple_linked_list) {
  TestTopo({1, 7},
           {
               {1, {2}},
               {2, {3}},
               {3, {4}},
               {4, {5}},
               {5, {6}},
               {7, {8}},
               {8, {9}},
               {9, {10}},
               {10, {11}},
               {11, {12}},
           },
           {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12});
}

TEST(DAGTopoVisitor, binary_tree) {
  TestTopo({1},
           {{1, {2, 3}}, {2, {4, 5}}, {3, {6, 7}}, {4, {8, 9}}, {5, {10, 11}}},
           {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

}  // namespace test

}  // namespace oneflow
