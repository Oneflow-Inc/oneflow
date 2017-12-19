#include "oneflow/core/common/longest_path_visitor.h"
namespace oneflow {

namespace {

typedef std::function<void(int, const std::function<void(int)>&)> ForEachNode;
typedef std::unordered_map<int, std::unordered_set<int>> AscendantNodes;

void GetAscendant(const std::list<int>& starts, const ForEachNode& foreach_next,
                  const ForEachNode& foreach_prev, AscendantNodes* ascendants) {
  BfsVisitor<int> bfs_foreach(foreach_next, foreach_prev);
  bfs_foreach(starts, [&](int node) {
    foreach_prev(node, [&](int prev) {
      std::list<int> prev_asc((*ascendants)[prev].begin(),
                              (*ascendants)[prev].end());
      for (int asc : prev_asc) { (*ascendants)[node].insert(asc); }
      (*ascendants)[node].insert(prev);
    });
  });
}

}  // namespace

TEST(LongestPathVisitor, one_vertix) {
  auto foreach_next = [&](int, const std::function<void(int)>&) {};
  auto foreach_prev = [&](int, const std::function<void(int)>&) {};
  auto is_ascendant = [](int, int) { return false; };
  auto get_node_weight = [](int) { return static_cast<double>(1); };
  auto path_visitor = [](const std::list<int>&) {};
  LongestPathVisitor<int> path(foreach_next, foreach_prev, is_ascendant);
  uint32_t cnt = path(1, 1, get_node_weight, path_visitor);
  ASSERT_TRUE(cnt == 1);
}

TEST(LongestPathVisitor, linked_list) {
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

  AscendantNodes ascendants;
  GetAscendant({1}, foreach_next, foreach_prev, &ascendants);

  auto is_ascendant = [&](int x, int y) {
    return ascendants[y].find(x) != ascendants[y].end();
  };
  auto get_node_weight = [](int) { return static_cast<double>(1); };

  LongestPathVisitor<int> lpath(foreach_next, foreach_prev, is_ascendant);

  int counter = 1;
  auto path_handler = [&](const std::list<int>& l) {
    int i = 1;
    for (auto itt = l.begin(); itt != l.end(); ++itt, ++i) {
      ASSERT_TRUE(i == *itt);
    }
    ++counter;
  };
  uint32_t cnt = lpath(1, 6, get_node_weight, path_handler);

  ASSERT_TRUE(cnt == 6);
}

TEST(LongestPathVisitor, linked_list_limited_path) {
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

  AscendantNodes ascendants;
  GetAscendant({1}, foreach_next, foreach_prev, &ascendants);

  auto is_ascendant = [&](int x, int y) {
    return ascendants[y].find(x) != ascendants[y].end();
  };
  auto get_node_weight = [](int) { return static_cast<double>(1); };

  LongestPathVisitor<int> lpath(foreach_next, foreach_prev, is_ascendant);

  int counter = 1;
  auto path_handler = [&](const std::list<int>& l) {
    int i = 1;
    for (auto itt = l.begin(); itt != l.end(); ++itt, ++i) {
      ASSERT_TRUE(i == *itt);
    }
    ++counter;
  };
  uint32_t cnt = lpath(1, 4, get_node_weight, path_handler);

  ASSERT_TRUE(cnt == 4);
}

TEST(LongestPathVisitor, multi_ascendences) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {-1, {3}}, {1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}};
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

  AscendantNodes ascendants;
  GetAscendant({1, -1}, foreach_next, foreach_prev, &ascendants);

  auto is_ascendant = [&](int x, int y) {
    return ascendants[y].find(x) != ascendants[y].end();
  };
  auto get_node_weight = [](int) { return static_cast<double>(1); };

  LongestPathVisitor<int> lpath(foreach_next, foreach_prev, is_ascendant);

  int counter = 1;
  auto path_handler = [&](const std::list<int>& l) {
    int i = 2;
    for (auto itt = l.begin(); itt != l.end(); ++itt, ++i) {
      ASSERT_TRUE(i == *itt);
    }
    ++counter;
  };
  uint32_t cnt = lpath(2, 6, get_node_weight, path_handler);

  ASSERT_TRUE(cnt == 5);
}

TEST(LongestPathVisitor, multi_descendants) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}}, {4, {5, 7}}, {5, {6, 7}}};
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

  AscendantNodes ascendants;
  GetAscendant({1}, foreach_next, foreach_prev, &ascendants);

  auto is_ascendant = [&](int x, int y) {
    return ascendants[y].find(x) != ascendants[y].end();
  };
  auto get_node_weight = [](int) { return static_cast<double>(1); };

  LongestPathVisitor<int> lpath(foreach_next, foreach_prev, is_ascendant);

  int counter = 1;
  auto path_handler = [&](const std::list<int>& l) {
    int i = 1;
    for (auto itt = l.begin(); itt != l.end(); ++itt, ++i) {
      ASSERT_TRUE(i == *itt);
    }
    ++counter;
  };
  uint32_t cnt = lpath(1, 6, get_node_weight, path_handler);

  ASSERT_TRUE(cnt == 6);
}

TEST(LongestPathVisitor, multi_path) {
  std::unordered_map<int, std::list<int>> next_nodes{
      {-1, {1}}, {1, {2, 6}}, {2, {3, 6}}, {3, {4}}, {4, {5, 7}}, {5, {6, 7}}};
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

  AscendantNodes ascendants;
  GetAscendant({-1}, foreach_next, foreach_prev, &ascendants);

  auto is_ascendant = [&](int x, int y) {
    return ascendants[y].find(x) != ascendants[y].end();
  };
  auto get_node_weight = [](int) { return static_cast<double>(1); };

  LongestPathVisitor<int> lpath(foreach_next, foreach_prev, is_ascendant);

  int counter = 1;
  auto path_handler = [&](const std::list<int>& l) {
    int i = 1;
    for (auto itt = l.begin(); itt != l.end(); ++itt, ++i) {
      ASSERT_TRUE(i == *itt);
    }
    ++counter;
  };
  uint32_t cnt = lpath(1, 6, get_node_weight, path_handler);

  ASSERT_TRUE(cnt == 6);
}

}  // namespace oneflow
