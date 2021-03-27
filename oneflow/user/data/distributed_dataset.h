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
#ifndef ONEFLOW_USER_DATA_DISTRIBUTED_DATASET_H_
#define ONEFLOW_USER_DATA_DISTRIBUTED_DATASET_H_

#include <cstddef>
#include "oneflow/user/data/dataset.h"
#include "oneflow/core/job/parallel_distribution_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace data {

template<typename LoadTarget>
class DistributedDataset final : public Dataset<LoadTarget> {
 public:
  using BaseDataset = RandomAccessDataset<LoadTarget>;
  using BaseDatasetUnqPtr = std::unique_ptr<BaseDataset>;
  using LoadTargetShdPtr = std::shared_ptr<LoadTarget>;
  using LoadTargetShdPtrVec = std::vector<LoadTargetShdPtr>;

  DistributedDataset(const Shape& hierarchy, const ParallelDistribution& parallel_dist, size_t rank,
                     BaseDatasetUnqPtr&& base_dataset)
      : base_dataset_(std::move(base_dataset)) {
    CHECK_EQ(hierarchy.NumAxes(), parallel_dist.sbp_parallel_size());
    num_shards_ = GetNumShards(hierarchy, parallel_dist);
    CHECK_EQ(base_dataset_->Size() % num_shards_, 0);
    index_counter_ = GetShardIndex(hierarchy, parallel_dist, rank);
    CHECK_LT(index_counter_, num_shards_);
  }
  ~DistributedDataset() = default;

  virtual LoadTargetShdPtrVec Next() override {
    LoadTargetShdPtrVec ret = base_dataset_->At(index_counter_);
    index_counter_ += num_shards_;
    return ret;
  }

 private:
  size_t GetNumShards(const Shape& hierarchy, const ParallelDistribution& parallel_dist) const {
    size_t num_shards = 1;
    FOR_RANGE(size_t, i, 0, parallel_dist.sbp_parallel_size()) {
      const auto& sbp_parallel = parallel_dist.sbp_parallel(i);
      if (sbp_parallel.has_split_parallel()) {
        num_shards *= hierarchy.At(sbp_parallel.split_parallel().axis());
      }
    }
    return num_shards;
  }

  size_t GetShardIndex(const Shape& hierarchy, const ParallelDistribution& parallel_dist,
                       size_t rank) const {
    using index_helper_t = NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>;
    size_t ndim = hierarchy.NumAxes();
    index_helper_t index_helper(hierarchy.dim_vec().data(), ndim);
    int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
    index_helper.OffsetToNdIndex(rank, nd_index);
    size_t stride = 1;
    size_t idx = 0;
    for (size_t i = ndim - 1; i >= 0; --i) {
      const auto& sbp_parallel = parallel_dist.sbp_parallel(i);
      if (sbp_parallel.has_split_parallel()) {
        idx += nd_index[i] * stride;
        stride *= hierarchy.At(i);
      }
    }
    return idx;
  }

  BaseDatasetUnqPtr base_dataset_;
  size_t num_shards_;
  size_t index_counter_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_DISTRIBUTED_DATASET_H_
