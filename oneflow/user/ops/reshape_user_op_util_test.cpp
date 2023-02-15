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

void TestEnumerateNdSbpIn2OutSignatures(
    const Shape& in_shape, const Shape& out_shape, const Shape& rank_mesh,
    const std::vector<std::vector<std::pair<int, int>>>& expected_nd_sbp_in2out_sig_list) {
  std::vector<std::vector<std::pair<int, int>>> actual_nd_sbp_in2out_sig_list;
  CHECK_JUST(ReshapeUserOpUtil::EnumerateNdSbpIn2OutSignatures(in_shape, out_shape, rank_mesh,
                                                               &actual_nd_sbp_in2out_sig_list));
  ASSERT_EQ(expected_nd_sbp_in2out_sig_list.size(), actual_nd_sbp_in2out_sig_list.size());
  for (size_t i = 0; i < actual_nd_sbp_in2out_sig_list.size(); ++i) {
    const auto& exp_nd_sbp_sig = expected_nd_sbp_in2out_sig_list[i];
    const auto& act_nd_sbp_sig = actual_nd_sbp_in2out_sig_list[i];
    ASSERT_EQ(exp_nd_sbp_sig.size(), act_nd_sbp_sig.size());
    for (size_t j = 0; j < exp_nd_sbp_sig.size(); ++j) {
      ASSERT_EQ(exp_nd_sbp_sig[j].first, act_nd_sbp_sig[j].first);
      ASSERT_EQ(exp_nd_sbp_sig[j].second, act_nd_sbp_sig[j].second);
    }
  }
}

}  // namespace

TEST(ReshapeUserOpUtil, EnumerateNdSbpIn2OutSignatures) {
  // 2D-split
  TestEnumerateNdSbpIn2OutSignatures({4}, {2, 2}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({12}, {2, 2, 3}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 4}, {8}, {2, 2}, {{{0, 0}, {1, 0}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 1, 4}, {8}, {2, 2}, {{{0, 0}, {2, 0}}});
  TestEnumerateNdSbpIn2OutSignatures({8, 2}, {2, 4, 2}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({8, 1, 2}, {2, 1, 4, 2}, {2, 2}, {{{0, 0}, {0, 2}}});
  TestEnumerateNdSbpIn2OutSignatures({3, 2, 3, 5}, {3, 30}, {2, 3}, {{{1, 1}, {2, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 4}, {4, 2}, {2, 2}, {{{0, 0}, {1, 0}}});
  TestEnumerateNdSbpIn2OutSignatures({4, 2}, {2, 4}, {2, 2}, {{{0, 0}, {0, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({4, 3}, {3, 4}, {2, 3}, {});
  TestEnumerateNdSbpIn2OutSignatures({2, 6}, {4, 3}, {2, 3}, {});
  TestEnumerateNdSbpIn2OutSignatures({2, 2, 5, 4}, {4, 5, 2, 2}, {2, 2},
                                     {{{0, 0}, {1, 0}}, {{3, 2}, {3, 3}}});
  // 3D-split
  TestEnumerateNdSbpIn2OutSignatures({24}, {2, 4, 3}, {2, 2, 2}, {{{0, 0}, {0, 1}, {0, 1}}});
  TestEnumerateNdSbpIn2OutSignatures({3, 24}, {3, 2, 2, 6}, {2, 2, 2}, {{{1, 1}, {1, 2}, {1, 3}}});
  // TestEnumerateNdSbpIn2OutSignatures({4, 77, 3}, {2, 2, 77, 3}, {2, 2, 3},
  //                                    {{{0, 0}, {0, 1}, {2, 3}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 3, 2, 5}, {12, 5}, {2, 3, 2}, {{{0, 0}, {1, 0}, {2, 0}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 1, 3, 2, 5}, {12, 1, 5}, {2, 3, 2},
                                     {{{0, 0}, {2, 0}, {3, 0}}});
  TestEnumerateNdSbpIn2OutSignatures({8, 4}, {2, 2, 8}, {2, 2, 2}, {{{0, 0}, {0, 1}, {0, 2}}});
  TestEnumerateNdSbpIn2OutSignatures({8, 2, 2}, {2, 2, 4, 2}, {2, 2, 2},
                                     {{{0, 0}, {0, 1}, {0, 2}}});
  TestEnumerateNdSbpIn2OutSignatures({8, 2, 1, 2}, {2, 2, 1, 4, 2}, {2, 2, 2},
                                     {{{0, 0}, {0, 1}, {0, 3}}});
  TestEnumerateNdSbpIn2OutSignatures({6, 4}, {2, 3, 2, 2}, {2, 3, 2}, {{{0, 0}, {0, 1}, {1, 2}}});
  TestEnumerateNdSbpIn2OutSignatures({6, 4, 1}, {2, 1, 3, 2, 2}, {2, 3, 2},
                                     {{{0, 0}, {0, 2}, {1, 3}}});
  // TestEnumerateNdSbpIn2OutSignatures({6, 5, 4}, {2, 3, 5, 2, 2}, {2, 3, 2},
  //                                    {{{0, 0}, {0, 1}, {2, 3}}});
  TestEnumerateNdSbpIn2OutSignatures({2, 8}, {2, 2, 2, 2}, {2, 2, 2}, {{{0, 0}, {1, 1}, {1, 2}}});
  // 4D-split
  // TestEnumerateNdSbpIn2OutSignatures({4, 77, 8}, {2, 2, 77, 2, 4}, {2, 2, 2, 2},
  //                                    {{{0, 0}, {0, 1}, {2, 3}, {2, 4}}});
}

}  // namespace test
}  // namespace oneflow
