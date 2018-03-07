#include "oneflow/core/graph/graph_node_visitor_util.h"

namespace oneflow {

namespace test {

namespace {

using TestNodeVisitor = GraphNodeVisitorUtil<int>;
using TestNodeHandler = TestNodeVisitor::HandlerType;

void TestTopoForEach(const std::list<int>& start_nodes,
                     const std::unordered_map<int, std::list<int>>& next_nodes,
                     const std::list<int>& expected_topo_order) {
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }
  auto ForEachIn = [&](int id, const TestNodeHandler& Handler) {
    if (prev_nodes.find(id) == prev_nodes.end()) return;
    for (int x : prev_nodes.at(id)) { Handler(x); }
  };
  auto ForEachOut = [&](int id, const TestNodeHandler& Handler) {
    if (next_nodes.find(id) == next_nodes.end()) return;
    for (int x : next_nodes.at(id)) { Handler(x); }
  };
  std::list<int> topo_list;
  TestNodeVisitor::TopoForEach(start_nodes, ForEachIn, ForEachOut,
                               [&](int x) { topo_list.push_back(x); });
  ASSERT_TRUE(topo_list == expected_topo_order);
}

void TestBfsForEach(const std::list<int>& start_nodes,
                    const std::unordered_map<int, std::list<int>>& next_nodes,
                    const std::list<int>& expected_order) {
  auto ForEachNext = [&](int id, const TestNodeHandler& Handler) {
    if (next_nodes.find(id) == next_nodes.end()) return;
    for (int x : next_nodes.at(id)) { Handler(x); }
  };
  std::list<int> visit_order;
  TestNodeVisitor::BfsForEach(start_nodes, ForEachNext,
                              [&](int x) { visit_order.push_back(x); });
  ASSERT_TRUE(visit_order == expected_order);
}

}  // namespace

TEST(GraphNodeVisitorUtil_TopoForEach, one_vertix) {
  TestTopoForEach({1}, {{1, {}}}, {1});
}

TEST(GraphNodeVisitorUtil_TopoForEach, diamond) {
  TestTopoForEach({1}, {{1, {2, 3}}, {2, {4}}, {3, {4}}}, {1, 2, 3, 4});
}

TEST(GraphNodeVisitorUtil_TopoForEach, weak_connected_non_symmetric) {
  TestTopoForEach({1}, {{1, {2, 3}}, {2, {4}}, {3, {6}}, {4, {5}}, {5, {6}}},
                  {1, 2, 3, 4, 5, 6});
}

TEST(GraphNodeVisitorUtil_TopoForEach, linked_list) {
  TestTopoForEach({1}, {{1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}},
                  {1, 2, 3, 4, 5, 6});
}

TEST(GraphNodeVisitorUtil_TopoForEach, multiple_linked_list) {
  TestTopoForEach({1, 7},
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

TEST(GraphNodeVisitorUtil_TopoForEach, binary_tree) {
  TestTopoForEach(
      {1}, {{1, {2, 3}}, {2, {4, 5}}, {3, {6, 7}}, {4, {8, 9}}, {5, {10, 11}}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

TEST(GraphNodeVisitorUtil_BfsForEach, one_vertix) {
  TestBfsForEach({1}, {{1, {}}}, {1});
}

TEST(GraphNodeVisitorUtil_BfsForEach, diamond) {
  TestBfsForEach({1}, {{1, {2, 3}}, {2, {4}}, {3, {4}}}, {1, 2, 3, 4});
}

TEST(GraphNodeVisitorUtil_BfsForEach, weak_connected_non_symmetric) {
  TestBfsForEach({1}, {{1, {2, 3}}, {2, {4}}, {3, {6}}, {4, {5}}, {5, {6}}},
                 {1, 2, 3, 4, 6, 5});
}

TEST(GraphNodeVisitorUtil_BfsForEach, linked_list) {
  TestBfsForEach({1}, {{1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}},
                 {1, 2, 3, 4, 5, 6});
}

TEST(GraphNodeVisitorUtil_BfsForEach, multiple_linked_list) {
  TestBfsForEach({1, 7},
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

TEST(GraphNodeVisitorUtil_BfsForEach, binary_tree) {
  TestBfsForEach(
      {1}, {{1, {2, 3}}, {2, {4, 5}}, {3, {6, 7}}, {4, {8, 9}}, {5, {10, 11}}},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

}  // namespace test

}  // namespace oneflow
