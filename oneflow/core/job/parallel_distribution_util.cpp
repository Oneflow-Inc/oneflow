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

#include "oneflow/core/job/parallel_distribution_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

std::vector<TensorSliceView> GetTensorSliceView(const int64_t parallel_num,
                                                const SbpParallel& sbp_parallel,
                                                const BlobDesc& blob_desc) {
  std::vector<Range> ranges(blob_desc.shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, blob_desc.shape().NumAxes()) {
    ranges[i].mut_begin() = 0;
    ranges[i].mut_end() = blob_desc.shape().At(i);
  }
  std::vector<TensorSliceView> views;
  if (sbp_parallel.has_partial_sum_parallel() || sbp_parallel.has_broadcast_parallel()) {
    FOR_RANGE(int64_t, i, 0, parallel_num) { views.emplace_back(ranges); }
  } else if (sbp_parallel.has_split_parallel()) {
    const int64_t axis = sbp_parallel.split_parallel().axis();
    const BalancedSplitter bs(blob_desc.shape().At(axis), parallel_num);
    FOR_RANGE(int64_t, i, 0, parallel_num) {
      if (bs.At(i).size() == 0) {
        views.emplace_back();
      } else {
        ranges[axis] = bs.At(i);
        views.emplace_back(ranges);
      }
    }
  } else {
    UNIMPLEMENTED();
  }
  return views;
}

TensorSliceView GetTensorSliceView4ParallelRank(const Shape& parallel_hierarchy,
                                                const ParallelDistribution& parallel_distribution,
                                                const Shape& logical_shape,
                                                const std::vector<int64_t>& parallel_rank) {
  std::vector<Range> ranges(logical_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, logical_shape.NumAxes()) {
    ranges[i].mut_begin() = 0;
    ranges[i].mut_end() = logical_shape.At(i);
  }
  if (parallel_hierarchy.elem_cnt() == 1) { return TensorSliceView(ranges); }
  if (parallel_hierarchy.NumAxes() == 1) {
    const SbpParallel& sbp_parallel = parallel_distribution.sbp_parallel(0);
    if (sbp_parallel.has_split_parallel()) {
      const int64_t split_axis = sbp_parallel.split_parallel().axis();
      CHECK_GE(split_axis, 0);
      CHECK_LT(split_axis, ranges.size());
      const int64_t id = parallel_rank.front();
      CHECK_GE(id, 0);
      CHECK_LT(id, parallel_hierarchy.elem_cnt());
      const BalancedSplitter bs(logical_shape.At(split_axis), parallel_hierarchy.elem_cnt());
      CHECK_GT(bs.At(id).size(), 0);
      ranges[split_axis] = bs.At(id);
    }
  } else {
    FOR_RANGE(int64_t, i, 0, parallel_hierarchy.NumAxes()) {
      const SbpParallel& sbp_parallel = parallel_distribution.sbp_parallel(i);
      if (sbp_parallel.has_split_parallel()) {
        const int64_t split_axis = sbp_parallel.split_parallel().axis();
        CHECK_GE(split_axis, 0);
        CHECK_LT(split_axis, ranges.size());
        CHECK_EQ(ranges[split_axis].size() % parallel_hierarchy.At(i), 0);
        const int64_t range_size = ranges[split_axis].size() / parallel_hierarchy.At(i);
        const int64_t dim_start = ranges[split_axis].begin() + parallel_rank.at(i) * range_size;
        ranges[split_axis].mut_begin() = dim_start;
        ranges[split_axis].mut_end() = dim_start + range_size;
      }
    }
  }
  return TensorSliceView(ranges);
}

TensorSliceView GetTensorSliceView4ParallelId(const Shape& parallel_hierarchy,
                                              const ParallelDistribution& parallel_distribution,
                                              const Shape& logical_shape, int64_t parallel_id) {
  NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE> hierarchy_index_helper(
      parallel_hierarchy.dim_vec().data(), parallel_hierarchy.NumAxes());
  std::vector<int64_t> parallel_rank(SHAPE_MAX_AXIS_SIZE);
  hierarchy_index_helper.OffsetToNdIndex(parallel_id, parallel_rank.data());
  return GetTensorSliceView4ParallelRank(parallel_hierarchy, parallel_distribution, logical_shape,
                                         parallel_rank);
}

std::vector<TensorSliceView> GetTensorSliceView(const Shape& parallel_hierarchy,
                                                const ParallelDistribution& parallel_distribution,
                                                const Shape& logical_shape) {
  std::vector<TensorSliceView> views;
  FOR_RANGE(int64_t, i, 0, parallel_hierarchy.elem_cnt()) {
    views.emplace_back(
        GetTensorSliceView4ParallelId(parallel_hierarchy, parallel_distribution, logical_shape, i));
  }
  return views;
}

TensorSliceView GetBroadcastTensorSliceView(const BlobDesc& blob_desc) {
  return TensorSliceView(blob_desc.shape());
}

}  // namespace oneflow
