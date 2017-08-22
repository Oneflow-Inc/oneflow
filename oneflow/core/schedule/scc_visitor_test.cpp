#include "oneflow/core/schedule/scc_visitor.h"
namespace oneflow {
namespace schedule {

TEST(SccVisitor, none) {
  std::unordered_map<int, int> data{
      {1, 2}, {2, 3}, {3, 4},
  };
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    if (data[x]) { cb(data[x]); }
  };
  SccVisitor<int> scc(foreach_next);
  ASSERT_TRUE(scc(1) == 0);
}

TEST(SccVisitor, weakly_connected) {
  std::unordered_map<int, std::list<int>> data{
      {1, {2, 3}}, {2, {4}}, {3, {4}},
  };
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    for (int n : data[x]) { cb(n); }
  };
  SccVisitor<int> scc(foreach_next);
  ASSERT_TRUE(scc(1) == 0);
}

TEST(SccVisitor, self_loop) {
  std::unordered_map<int, int> data{
      {1, 1},
  };
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    if (data[x]) { cb(data[x]); }
  };
  SccVisitor<int> scc(foreach_next);
  uint32_t cnt = scc(1, [&](const std::list<int>& l) {
    ASSERT_TRUE(l.size() == 1);
    ASSERT_TRUE(l.front() == 1);
  });
  ASSERT_TRUE(cnt == 1);
}

TEST(SccVisitor, loop) {
  std::unordered_map<int, int> data{
      {1, 2}, {2, 3}, {3, 1},
  };
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    if (data[x]) { cb(data[x]); }
  };
  SccVisitor<int> scc(foreach_next);
  uint32_t cnt = scc(1, [&](const std::list<int>& l) {
    ASSERT_TRUE(l.size() == 3);
    ASSERT_TRUE(l.front() == 1);
  });
  ASSERT_TRUE(cnt == 1);
}

TEST(SccVisitor, nested_loop) {
  std::unordered_map<int, std::list<int>> data{{1, {2}},    {2, {3}}, {3, {4}},
                                               {4, {2, 5}}, {5, {6}}, {6, {1}}};
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    for (auto itt = data[x].begin(); itt != data[x].end(); itt++) {
      int n = *itt;
      cb(n);
    }
  };
  SccVisitor<int> scc(foreach_next);
  uint32_t cnt = scc(1, [&](const std::list<int>& l) {
    ASSERT_TRUE(l.size() == 6);
    ASSERT_TRUE(l.front() == 1);
  });
  ASSERT_TRUE(cnt == 1);
}

TEST(SccVisitor, forest) {
  std::unordered_map<int, int> data{
      {1, 2}, {2, 3}, {3, 1}, {4, 5}, {5, 6}, {6, 4},
  };
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    if (data[x]) { cb(data[x]); }
  };
  SccVisitor<int> scc(foreach_next);
  std::list<int> starts{1, 4};
  uint32_t cnt = scc(starts, [&](const std::list<int>& l) {
    ASSERT_TRUE(l.size() == 3);
    ASSERT_TRUE(l.front() == 1 || l.front() == 4);
  });
  ASSERT_TRUE(cnt == 2);
}

TEST(SccVisitor, wikipedia_tarjan_demo) {
  std::unordered_map<int, std::list<int>> data{
      {1, {2}},    {2, {3}},    {3, {1}}, {4, {2, 3, 5}},
      {5, {4, 6}}, {6, {3, 7}}, {7, {6}}, {8, {5, 7, 8}}};
  auto foreach_next = [&](int x, const std::function<void(int)>& cb) {
    for (auto itt = data[x].begin(); itt != data[x].end(); itt++) {
      int n = *itt;
      cb(n);
    }
  };
  SccVisitor<int> scc(foreach_next);
  std::list<int> nodes{1, 2, 3, 4, 5, 6, 7, 8};
  uint32_t cnt = scc(nodes, [&](const std::list<int>& l) {
    if (l.front() == 1) { ASSERT_TRUE(l.size() == 3); }
    if (l.front() == 4) { ASSERT_TRUE(l.size() == 2); }
    if (l.front() == 6) { ASSERT_TRUE(l.size() == 2); }
    if (l.front() == 8) { ASSERT_TRUE(l.size() == 1); }
  });
  ASSERT_TRUE(cnt == 4);
}

}  // namespace schedule
}  // namespace oneflow
