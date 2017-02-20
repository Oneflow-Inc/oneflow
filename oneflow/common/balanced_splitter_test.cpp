#include "balanced_splitter.h"
#include "gtest/gtest.h"

namespace oneflow {

TEST(BalancedSplitter, split_20_to_6_part) {
  BalancedSplitter splitter;
  splitter.Init(20, 6);
  ASSERT_EQ(splitter.at(0), 4);
  ASSERT_EQ(splitter.at(1), 4);
  ASSERT_EQ(splitter.at(2), 3);
  ASSERT_EQ(splitter.at(3), 3);
  ASSERT_EQ(splitter.at(4), 3);
  ASSERT_EQ(splitter.at(5), 3);
}

} // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
