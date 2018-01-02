#include "oneflow/core/common/dfs_visitor.h"
namespace oneflow {
namespace test {

void TestBackEdgeDetection(const HashMap<int, std::list<int>>& graph,
                           const std::list<std::pair<int, int>>& expected) {
  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    if (graph.find(id) == graph.end()) { return; }
    for (int x : graph.at(id)) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  HashMap<int, bool> entered;
  auto OnEnter = [&](int x) { entered[x] = true; };
  HashMap<int, bool> exited;
  auto OnExit = [&](int x) { exited[x] = true; };

  std::list<std::pair<int, int>> back_edges;

  dfs_foreach_node({1}, OnExit, [&](int x) {
    OnEnter(x);
    ForEachNext(x, [&](int child) {
      if (entered[child] && !exited[child]) {
        back_edges.push_back(std::make_pair(x, child));
      }
    });
  });
  auto Compare = [](const std::pair<int, int>& a,
                    const std::pair<int, int>& b) {
    return (a.first < b.first)
           || ((a.first == b.first) && (a.second < b.second));
  };
  back_edges.sort(Compare);
  auto expected_back_edges = expected;
  expected_back_edges.sort(Compare);
  ASSERT_TRUE(back_edges == expected_back_edges);
}

TEST(DfsVisitor, one_vertix) {
  auto ForEachNext = [&](int, const std::function<void(int)>&) {};
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  dfs_foreach_node({1}, [&](int node) { ASSERT_TRUE(node == 1); });
}

TEST(DfsVisitor, diamond) {
  HashMap<int, std::list<int>> next_nodes{{1, {2, 3}}, {2, {4}}, {3, {4}}};

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int index = 0;
  std::vector<int> order{1, 2, 4, 3};

  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });
}

TEST(DfsVisitor, no_back_edge_on_diamond) {
  HashMap<int, std::list<int>> next_nodes{{1, {2, 3}}, {2, {4}}, {3, {4}}};
  std::list<std::pair<int, int>> expected_back_edges{};
  TestBackEdgeDetection(next_nodes, expected_back_edges);
}

TEST(DfsVisitor, linked_list) {
  HashMap<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}};

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int counter = 1;
  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });
}

TEST(DfsVisitor, multiple_linked_list) {
  HashMap<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}},  {4, {5}},   {5, {6}},
      {7, {8}}, {8, {9}}, {9, {10}}, {10, {11}}, {11, {12}},
  };

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int index = 0;
  std::vector<int> order{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  std::list<int> starts{1, 7};
  dfs_foreach_node(starts, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });
}

TEST(DfsVisitor, binary_tree) {
  HashMap<int, std::list<int>> next_nodes{
      {1, {2, 3}}, {2, {4, 5}}, {3, {6, 7}}, {4, {8, 9}}, {5, {10, 11}}};

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int index = 0;
  std::vector<int> order{1, 2, 4, 8, 9, 5, 10, 11, 3, 6, 7};
  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });
}

TEST(DfsVisitor, cycle_linked_list) {
  HashMap<int, std::list<int>> next_nodes{{1, {2}}, {2, {3}}, {3, {4}},
                                          {4, {5}}, {5, {6}}, {6, {2}}};

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int counter = 1;
  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });
}

TEST(DfsVisitor, back_edge_on_cycle_linked_list) {
  HashMap<int, std::list<int>> next_nodes{{1, {2}}, {2, {3}}, {3, {4}},
                                          {4, {5}}, {5, {6}}, {6, {2}}};
  std::list<std::pair<int, int>> expected_back_edges{{6, 2}};
  TestBackEdgeDetection(next_nodes, expected_back_edges);
}

TEST(DfsVisitor, mutiple_cycles) {
  HashMap<int, std::list<int>> next_nodes{{1, {2}}, {2, {3}},       {3, {4}},
                                          {4, {5}}, {5, {6, 2, 3}}, {6, {2}}};

  auto ForEachNext = [&](int id, const std::function<void(int)>& Handler) {
    for (int x : next_nodes[id]) { Handler(x); }
  };
  DfsVisitor<int> dfs_foreach_node(ForEachNext);

  int counter = 1;
  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });
}

TEST(DfsVisitor, back_edge_on_multiple_cycles) {
  HashMap<int, std::list<int>> next_nodes{{1, {2}}, {2, {3}},       {3, {4}},
                                          {4, {5}}, {5, {6, 2, 3}}, {6, {2}}};
  std::list<std::pair<int, int>> expected_back_edges{{6, 2}, {5, 3}, {5, 2}};
  TestBackEdgeDetection(next_nodes, expected_back_edges);
}

}  // namespace test

}  // namespace oneflow
