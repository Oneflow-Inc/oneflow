#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

TEST(Optional, copy_constructor) {
  Optional<int64_t> a(0);
  std::vector<Optional<int64_t>> vec;
  vec.push_back(a);
  ASSERT_TRUE(vec[0].has_value());
  int64_t val = CHECK_JUST(vec[0].value());
  ASSERT_EQ(val, 0);
}

TEST(Optional, move_constructor) {
  Optional<int64_t> a(0);
  std::map<int64_t, Optional<int64_t>> map;
  map.emplace(0, a);
  ASSERT_TRUE(map.at(0).has_value());
  int64_t val = CHECK_JUST(map.at(0).value());
  ASSERT_EQ(val, 0);
}

}
}
