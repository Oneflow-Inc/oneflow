/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/ops/reshape_user_op_util.h"

#include <gtest/gtest.h>

namespace oneflow {
namespace test {

namespace {

void TestEnumerateNdSplitInAxis2OutAxis(
    const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
    const std::vector<std::vector<std::pair<int, int>>>& expected_in2out_axis) {
  std::vector<std::vector<std::pair<int, int>>> actual_in2out_axis;
  CHECK_JUST(ReshapeUserOpUtil::EnumerateNdSplitIn2OutAxis(in_shape, out_shape, rank_mesh,
                                                           &actual_in2out_axis));
  ASSERT_EQ(expected_in2out_axis.size(), actual_in2out_axis.size());
  for (size_t i = 0; i < expected_in2out_axis.size(); ++i) {
    const auto& exp_nd_split_axis = expected_in2out_axis[i];
    const auto& act_nd_split_axis = actual_in2out_axis[i];
    ASSERT_EQ(exp_nd_split_axis.size(), act_nd_split_axis.size());
    for (size_t j = 0; j < exp_nd_split_axis.size(); ++j) {
      ASSERT_EQ(exp_nd_split_axis[j].first, act_nd_split_axis[j].first);
      ASSERT_EQ(exp_nd_split_axis[j].second, act_nd_split_axis[j].second);
    }
  }
}

}  // namespace

TEST(ReshapeUserOpUtil, EnumerateNdSplitIn2OutAxis) {
  // 2D-split
  TestEnumerateNdSplitInAxis2OutAxis({4}, {2, 2}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({12}, {2, 2, 3}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({2, 4}, {8}, {2, 2}, {{{0, 0}, {1, 0}}});
  TestEnumerateNdSplitInAxis2OutAxis({2, 1, 4}, {8}, {2, 2}, {{{0, 0}, {2, 0}}});
  TestEnumerateNdSplitInAxis2OutAxis({8, 2}, {2, 4, 2}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({8, 1, 2}, {2, 1, 4, 2}, {2, 2}, {{{0, 0}, {0, 2}}});
  TestEnumerateNdSplitInAxis2OutAxis({3, 2, 3, 5}, {3, 30}, {2, 3}, {{{1, 1}, {2, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({2, 4}, {4, 2}, {2, 2}, {{{0, 0}, {1, 0}}});
  TestEnumerateNdSplitInAxis2OutAxis({4, 2}, {2, 4}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({4, 3}, {3, 4}, {2, 3}, {});
  TestEnumerateNdSplitInAxis2OutAxis({2, 6}, {4, 3}, {2, 3}, {});
  TestEnumerateNdSplitInAxis2OutAxis({2, 2, 5, 4}, {4, 5, 2, 2}, {2, 2},
                                     {{{0, 0}, {1, 0}}, {{3, 2}, {3, 3}}});
  // 3D-split
  TestEnumerateNdSplitInAxis2OutAxis({24}, {2, 4, 3}, {2, 2, 2}, {{{0, 0}, {0, 1}, {0, 1}}});
  TestEnumerateNdSplitInAxis2OutAxis({3, 24}, {3, 2, 2, 6}, {2, 2, 2}, {{{1, 1}, {1, 2}, {1, 3}}});
  TestEnumerateNdSplitInAxis2OutAxis({2, 3, 2, 5}, {12, 5}, {2, 3, 2}, {{{0, 0}, {1, 0}, {2, 0}}});
  TestEnumerateNdSplitInAxis2OutAxis({2, 1, 3, 2, 5}, {12, 1, 5}, {2, 3, 2},
                                     {{{0, 0}, {2, 0}, {3, 0}}});
  TestEnumerateNdSplitInAxis2OutAxis({8, 4}, {2, 2, 8}, {2, 2, 2}, {{{0, 0}, {0, 1}, {0, 2}}});
  TestEnumerateNdSplitInAxis2OutAxis({8, 2, 2}, {2, 2, 4, 2}, {2, 2, 2},
                                     {{{0, 0}, {0, 1}, {0, 2}}});
  TestEnumerateNdSplitInAxis2OutAxis({8, 2, 1, 2}, {2, 2, 1, 4, 2}, {2, 2, 2},
                                     {{{0, 0}, {0, 1}, {0, 3}}});
  TestEnumerateNdSplitInAxis2OutAxis({6, 4}, {2, 3, 2, 2}, {2, 3, 2}, {{{0, 0}, {0, 1}, {1, 2}}});
  TestEnumerateNdSplitInAxis2OutAxis({6, 4, 1}, {2, 1, 3, 2, 2}, {2, 3, 2},
                                     {{{0, 0}, {0, 2}, {1, 3}}});
  TestEnumerateNdSplitInAxis2OutAxis({6, 5, 4}, {2, 3, 5, 2, 2}, {2, 3, 2},
                                     {{{0, 0}, {0, 1}, {2, 3}}});
}

}  // namespace test
}  // namespace oneflow
