#include "oneflow/core/common/dfs_visitor.h"
namespace oneflow {
namespace test {

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
  std::vector<int> order{1, 3, 4, 2};

  dfs_foreach_node({1}, [&](int x) {
    ASSERT_TRUE(x == order[index]);
    ++index;
  });
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
  std::vector<int> order{7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6};

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
  std::vector<int> order{1, 3, 7, 6, 2, 5, 11, 10, 4, 9, 8};
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

TEST(DfsVisitor, mutli_cycle) {
  HashMap<int, std::list<int>> next_nodes{
      {1, {2}}, {2, {3}}, {3, {4}}, {4, {5}}, {5, {6}}, {6, {2}}, {5, {3, 2}}};

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

}  // namespace test

}  // namespace oneflow
