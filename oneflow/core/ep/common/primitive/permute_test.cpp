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
#include "oneflow/core/ep/common/primitive/permute.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace ep {
namespace primitive {

namespace permute {

namespace {

template<size_t max_num_dims>
void TestSimplifyPermutation(size_t num_dims, const int64_t* src_dims, const int* permutation,
                             size_t expected_num_dims, const int64_t* expected_src_dims,
                             const int* expected_permutation) {
  size_t simplified_num_dims = 0;
  int64_t simplified_src_dims[max_num_dims]{};
  int simplified_permutation[max_num_dims]{};
  SimplifyPermutation<max_num_dims>(num_dims, src_dims, permutation, &simplified_num_dims,
                                    simplified_src_dims, simplified_permutation);
  ASSERT_EQ(simplified_num_dims, expected_num_dims);
  for (size_t i = 0; i < simplified_num_dims; ++i) {
    ASSERT_EQ(simplified_src_dims[i], expected_src_dims[i]);
    ASSERT_EQ(simplified_permutation[i], expected_permutation[i]);
  }
}

TEST(Permute, SimplifyPermutation) {
  constexpr size_t max_num_dims = 8;

  const size_t num_dims_1 = 5;
  int64_t src_dims_1[max_num_dims]{1, 2, 2, 1, 2};
  int permutation_1[max_num_dims]{0, 1, 3, 4, 2};
  const size_t simplified_num_dims_1 = 3;
  int64_t simplified_src_dims_1[max_num_dims]{2, 2, 2};
  int simplified_permutation_1[max_num_dims]{0, 2, 1};
  TestSimplifyPermutation<max_num_dims>(num_dims_1, src_dims_1, permutation_1,
                                        simplified_num_dims_1, simplified_src_dims_1,
                                        simplified_permutation_1);

  const size_t num_dims_2 = 4;
  int64_t src_dims_2[max_num_dims]{5, 6, 7, 8};
  int permutation_2[max_num_dims]{2, 3, 0, 1};
  const size_t simplified_num_dims_2 = 2;
  int64_t simplified_src_dims_2[max_num_dims]{5 * 6, 7 * 8};
  int simplified_permutation_2[max_num_dims]{1, 0};
  TestSimplifyPermutation<max_num_dims>(num_dims_2, src_dims_2, permutation_2,
                                        simplified_num_dims_2, simplified_src_dims_2,
                                        simplified_permutation_2);

  const size_t num_dims_3 = 4;
  int64_t src_dims_3[max_num_dims]{5, 6, 7, 8};
  int permutation_3[max_num_dims]{0, 1, 2, 3};
  const size_t simplified_num_dims_3 = 1;
  int64_t simplified_src_dims_3[max_num_dims]{5 * 6 * 7 * 8};
  int simplified_permutation_3[max_num_dims]{0};
  TestSimplifyPermutation<max_num_dims>(num_dims_3, src_dims_3, permutation_3,
                                        simplified_num_dims_3, simplified_src_dims_3,
                                        simplified_permutation_3);
}

}  // namespace

}  // namespace permute

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
