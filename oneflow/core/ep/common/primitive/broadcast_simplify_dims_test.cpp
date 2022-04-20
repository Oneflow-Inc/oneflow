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
#include "oneflow/core/ep/common/primitive/util.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

template<size_t max_num_dims>
void TestSimplifyBroadcastDims(size_t num_src0_dims, const int64_t* src0_dims, size_t num_src1_dims,
                               const int64_t* src1_dims, size_t expected_num_dims,
                               const int64_t* expected_src0_dims, const int64_t* expected_src1_dims,
                               const int64_t* expected_dst_dims) {
  size_t simplified_num_dims = 0;
  int64_t simplified_src0_dims[max_num_dims]{};
  int64_t simplified_src1_dims[max_num_dims]{};
  int64_t simplified_dst_dims[max_num_dims]{};
  SimplifyBroadcastDims<max_num_dims>(num_src0_dims, src0_dims, num_src1_dims, src1_dims,
                                      &simplified_num_dims, simplified_src0_dims,
                                      simplified_src1_dims, simplified_dst_dims);
  ASSERT_EQ(simplified_num_dims, expected_num_dims);
  for (size_t i = 0; i < simplified_num_dims; ++i) {
    ASSERT_EQ(simplified_src0_dims[i], expected_src0_dims[i]);
    ASSERT_EQ(simplified_src1_dims[i], expected_src1_dims[i]);
    ASSERT_EQ(simplified_dst_dims[i], expected_dst_dims[i]);
  }
}

TEST(Broadcast, SimplifyBroadcastDims) {
  constexpr size_t max_num_dims = 8;

  const size_t num_src0_dims_1 = 4;
  const size_t num_src1_dims_1 = 5;
  int64_t src0_dims_1[max_num_dims]{2, 5, 10, 5};
  int64_t src1_dims_1[max_num_dims]{5, 1, 5, 10, 1};
  const size_t simplified_num_dims_1 = 4;
  int64_t simplified_src0_dims_1[max_num_dims]{1, 2, 50, 5};
  int64_t simplified_src1_dims_1[max_num_dims]{5, 1, 50, 1};
  int64_t simplified_dst_dims_1[max_num_dims]{5, 2, 50, 5};
  TestSimplifyBroadcastDims<max_num_dims>(
      num_src0_dims_1, src0_dims_1, num_src1_dims_1, src1_dims_1, simplified_num_dims_1,
      simplified_src0_dims_1, simplified_src1_dims_1, simplified_dst_dims_1);

  const size_t num_src0_dims_2 = 4;
  const size_t num_src1_dims_2 = 1;
  int64_t src0_dims_2[max_num_dims]{10, 5, 1, 5};
  int64_t src1_dims_2[max_num_dims]{5};
  const size_t simplified_num_dims_2 = 2;
  int64_t simplified_src0_dims_2[max_num_dims]{50, 5};
  int64_t simplified_src1_dims_2[max_num_dims]{1, 5};
  int64_t simplified_dst_dims_2[max_num_dims]{50, 5};
  TestSimplifyBroadcastDims<max_num_dims>(
      num_src0_dims_2, src0_dims_2, num_src1_dims_2, src1_dims_2, simplified_num_dims_2,
      simplified_src0_dims_2, simplified_src1_dims_2, simplified_dst_dims_2);

  const size_t num_src0_dims_3 = 4;
  const size_t num_src1_dims_3 = 1;
  int64_t src0_dims_3[max_num_dims]{2, 5, 10, 5};
  int64_t src1_dims_3[max_num_dims]{1};
  const size_t simplified_num_dims_3 = 1;
  int64_t simplified_src0_dims_3[max_num_dims]{500};
  int64_t simplified_src1_dims_3[max_num_dims]{1};
  int64_t simplified_dst_dims_3[max_num_dims]{500};
  TestSimplifyBroadcastDims<max_num_dims>(
      num_src0_dims_3, src0_dims_3, num_src1_dims_3, src1_dims_3, simplified_num_dims_3,
      simplified_src0_dims_3, simplified_src1_dims_3, simplified_dst_dims_3);
}

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
