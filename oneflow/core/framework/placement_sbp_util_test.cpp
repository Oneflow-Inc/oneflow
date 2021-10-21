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
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace test {

TEST(GetBroadcastParallelIds, 1d_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size});
  std::vector<bool> dim2is_broadcast{true};
  const auto& expected = std::vector<int64_t>{0, 1, 2, 3};
  for (int i = 0; i < parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, i));
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetBroadcastParallelIds, 1d_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size});
  std::vector<bool> dim2is_broadcast{false};
  for (int i = 0; i < parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, i));
    const auto& expected = std::vector<int64_t>{i};
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetBroadcastParallelIds, 2d_broadcast_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<bool> dim2is_broadcast{true, true};
  std::vector<int64_t> expected{};
  for (int i = 0; i < parallel_size * parallel_size; ++i) { expected.push_back(i); }
  for (int i = 0; i < parallel_size * parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, i));
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetBroadcastParallelIds, 2d_nonbroadcast_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<bool> dim2is_broadcast{false, false};
  for (int i = 0; i < parallel_size * parallel_size; ++i) {
    const auto& broadcast_parallel_ids =
        CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, i));
    const auto& expected = std::vector<int64_t>{i};
    ASSERT_TRUE(*broadcast_parallel_ids == expected);
  }
}

TEST(GetBroadcastParallelIds, 2d_broadcast_nonbroadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<bool> dim2is_broadcast{true, false};
  for (int i = 0; i < parallel_size; ++i) {
    for (int j = 0; j < parallel_size; ++j) {
      std::vector<int64_t> expected{};
      for (int k = 0; k < parallel_size; ++k) { expected.push_back(k * parallel_size + j); }
      int64_t parallel_id = i * parallel_size + j;
      const auto& broadcast_parallel_ids =
          CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, parallel_id));
      ASSERT_TRUE(*broadcast_parallel_ids == expected);
    }
  }
}

TEST(GetBroadcastParallelIds, 2d_nonbroadcast_broadcast) {
  int64_t parallel_size = 4;
  Shape hierarchy_shape(DimVector{parallel_size, parallel_size});
  std::vector<bool> dim2is_broadcast{false, true};
  for (int i = 0; i < parallel_size; ++i) {
    std::vector<int64_t> expected{};
    for (int j = 0; j < parallel_size; ++j) { expected.push_back(i * parallel_size + j); }
    for (int j = 0; j < parallel_size; ++j) {
      int64_t parallel_id = i * parallel_size + j;
      const auto& broadcast_parallel_ids =
          CHECK_JUST(GetBroadcastParallelIds(hierarchy_shape, dim2is_broadcast, parallel_id));
      ASSERT_TRUE(*broadcast_parallel_ids == expected);
    }
  }
}

}  // namespace test
}  // namespace oneflow
