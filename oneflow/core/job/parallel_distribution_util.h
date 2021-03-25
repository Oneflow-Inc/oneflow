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
#ifndef ONEFLOW_CORE_JOB_PARALLEL_DISTRIBUTION_UTIL_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DISTRIBUTION_UTIL_H_

#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

std::vector<TensorSliceView> GetTensorSliceView(int64_t parallel_num,
                                                const SbpParallel& sbp_parallel,
                                                const BlobDesc& blob_desc);
std::vector<TensorSliceView> GetTensorSliceView(const Shape& parallel_hierarchy,
                                                const ParallelDistribution& parallel_distribution,
                                                const Shape& logical_shape);
TensorSliceView GetTensorSliceView4ParallelRank(const Shape& parallel_hierarchy,
                                                const ParallelDistribution& parallel_distribution,
                                                const Shape& logical_shape,
                                                const std::vector<int64_t>& parallel_rank);
TensorSliceView GetTensorSliceView4ParallelId(const Shape& parallel_hierarchy,
                                              const ParallelDistribution& parallel_distribution,
                                              const Shape& logical_shape, int64_t parallel_id);
TensorSliceView GetBroadcastTensorSliceView(const BlobDesc& blob_desc);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_SBP_PARALLEL_H_
