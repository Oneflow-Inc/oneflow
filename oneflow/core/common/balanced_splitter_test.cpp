#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

TEST(BalancedSplitter, split_20_to_6_part) {
  BalancedSplitter splitter(20, 6);
  ASSERT_TRUE(splitter.At(0) == Range(0, 4));
  ASSERT_TRUE(splitter.At(1) == Range(4, 8));
  ASSERT_TRUE(splitter.At(2) == Range(8, 11));
  ASSERT_TRUE(splitter.At(3) == Range(11, 14));
  ASSERT_TRUE(splitter.At(4) == Range(14, 17));
  ASSERT_TRUE(splitter.At(5) == Range(17, 20));
}

TEST(BalancedSplitter, split_2_to_3_part) {
  BalancedSplitter splitter(2, 3);
  ASSERT_TRUE(splitter.At(0) == Range(0, 1));
  ASSERT_TRUE(splitter.At(1) == Range(1, 2));
  ASSERT_TRUE(splitter.At(2) == Range(2, 2));
}

} // namespace oneflow
