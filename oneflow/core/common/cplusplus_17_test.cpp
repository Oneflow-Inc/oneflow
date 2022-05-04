#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

#include "oneflow/core/common/cplusplus_17.h"

namespace oneflow {
namespace test {

TEST(Scan, scan) {
  std::vector<int> data{3, 1, 4, 1, 5, 9, 2, 6};
  std::vector<int> output;

  std::exclusive_scan(data.begin(), data.end(), std::back_insert_iterator<std::vector<int>>(output), 0);
  std::vector<int> ref_output = {0, 3, 4, 8, 9, 14, 23, 25};
  EXPECT_EQ(output, ref_output);
  output.clear();

  std::inclusive_scan(data.begin(), data.end(), std::back_insert_iterator<std::vector<int>>(output));
  ref_output = {3, 4, 8, 9, 14, 23, 25, 31};
  EXPECT_EQ(output, ref_output);
  output.clear();

  std::exclusive_scan(data.begin(), data.end(), std::back_insert_iterator<std::vector<int>>(output), 1, std::multiplies<>{});
  ref_output = {1, 3, 3, 12, 12, 60, 540, 1080};
  EXPECT_EQ(output, ref_output);
  output.clear();

  std::inclusive_scan(data.begin(), data.end(), std::back_insert_iterator<std::vector<int>>(output), std::multiplies<>{});
  ref_output = {3, 3, 12, 12, 60, 540, 1080, 6480};
  EXPECT_EQ(output, ref_output);
  output.clear();

  std::exclusive_scan(data.rbegin(), data.rend(), std::back_insert_iterator<std::vector<int>>(output), 1, std::multiplies<>{});
  ref_output = {1, 6, 12, 108, 540, 540, 2160, 2160};
  EXPECT_EQ(output, ref_output);
  output.clear();
}

}  // namespace test
}  // namespace oneflow
