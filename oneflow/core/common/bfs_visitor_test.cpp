#include "oneflow/core/common/bfs_visitor.h"
namespace oneflow {

TEST(BfsVisitor, one_vertix) {
  auto foreach_next = [&](int, const std::function<void(int)>&) {};
  auto foreach_prev = [&](int, const std::function<void(int)>&) {};
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  uint32_t cnt = bfs_foreach_node(1, [&](int node) { ASSERT_TRUE(node == 1); });
  ASSERT_TRUE(cnt == 1);
}

TEST(BfsVisitor, diamond) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2, 3}}, {2, {4}}, {3, {4}}};
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  int index = 0;
  std::vector<int> order{1, 2, 3, 4};

  uint32_t cnt = bfs_foreach_node(1, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });

  ASSERT_TRUE(cnt == 4);
}

TEST(BfsVisitor, linked_list) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}};
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  int counter = 1;
  uint32_t cnt = bfs_foreach_node(1, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });

  ASSERT_TRUE(cnt == 6);
}

TEST(BfsVisitor, linked_list_break_in_while) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}};
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  auto is_final = [](int) { return true; };

  int counter = 1;
  uint32_t cnt = bfs_foreach_node.Walk({1}, is_final, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });

  ASSERT_TRUE(cnt == 1);
}

TEST(BfsVisitor, multiple_linked_list) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}},  {4, {5}},   {5, {6}},
      {7, {8}}, {8, {9}}, {9, {10}}, {10, {11}}, {11, {12}},
  };
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  int index = 0;
  std::vector<int> order{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};

  std::list<int> starts{1, 7};
  uint32_t cnt = bfs_foreach_node(starts, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });

  ASSERT_TRUE(cnt == 12);
}

TEST(BfsVisitor, binary_tree) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2, 3}}, {2, {4, 5}}, {3, {6, 7}}, {4, {8, 9}}, {5, {10, 11}}};
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  int counter = 1;
  uint32_t cnt = bfs_foreach_node(1, [&](int x) {
    ASSERT_TRUE(x == counter);
    ++counter;
  });

  ASSERT_TRUE(cnt == 11);
}

TEST(BfsVisitor, start_from_middle) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2, 6}},  {2, {3, 5}},  {3, {4}},  {4, {5}},   {5, {6}},
      {7, {8, 12}}, {8, {9, 11}}, {9, {10}}, {10, {11}}, {11, {12}},
  };
  std::unordered_map<int, std::list<int>> prev_nodes;
  for (const auto& pair : next_nodes) {
    int prev = pair.first;
    for (int next : pair.second) { prev_nodes[next].push_back(prev); }
  }

  auto foreach_prev = [&](int id, const std::function<void(int)>& cb) {
    for (int x : prev_nodes[id]) { cb(x); }
  };

  auto foreach_next = [&](int id, const std::function<void(int)>& cb) {
    for (int x : next_nodes[id]) { cb(x); }
  };
  BfsVisitor<int> bfs_foreach_node(foreach_next, foreach_prev);

  int index = 0;
  std::vector<int> order{3, 9, 4, 10, 5, 11, 6, 12};

  std::list<int> starts{3, 9};
  uint32_t cnt = bfs_foreach_node(starts, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });

  ASSERT_TRUE(cnt == 8);
}

}  // namespace oneflow
