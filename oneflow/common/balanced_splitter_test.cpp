#include "balanced_splitter.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(BalancedSplitter, split_20_to_6_part) {
  BalancedSplitter splitter;
  splitter.Init(20, 6);
  ASSERT_EQ(splitter.At(0), 4);
  ASSERT_EQ(splitter.At(1), 4);
  ASSERT_EQ(splitter.At(2), 3);
  ASSERT_EQ(splitter.At(3), 3);
  ASSERT_EQ(splitter.At(4), 3);
  ASSERT_EQ(splitter.At(5), 3);
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
