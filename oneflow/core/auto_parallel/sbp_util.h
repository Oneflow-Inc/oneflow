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
#ifndef ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_
#define ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/sbp_parallel.pb.h"

namespace oneflow {
namespace auto_parallel {

// check whether the sbp_parallel is legal
bool CheckSbpParallel(const cfg::SbpParallel& sbp_parallel);
// check whether the nd_sbp is legal
bool CheckNdSbp(const cfg::NdSbp& nd_sbp);

// compute copy cost
Maybe<double> ComputCopyCostBetweenTwoSbpParallel(const cfg::SbpParallel& producer_sbp_parallel,
                                                  const cfg::SbpParallel& consumer_sbp_parallel,
                                                  const BlobDesc& logical_blob_desc,
                                                  const ParallelDesc& producer_parallel_desc,
                                                  const ParallelDesc& consumer_parallel_desc,
                                                  bool is_same_sbp, bool allow_cpu2gpu);
Maybe<double> ComputCopyCostBetweenNdSbp(const cfg::NdSbp& producer_sbp_parallel,
                                         const cfg::NdSbp& consumer_sbp_parallel,
                                         const BlobDesc& logical_blob_desc,
                                         const ParallelDesc& producer_parallel_desc,
                                         const ParallelDesc& consumer_parallel_desc,
                                         bool is_same_sbp, bool allow_cpu2gpu);

// Judge whether we need the same SBP for both producer and consumer
bool IsSameSbp(OpNode* consumer, const std::string& ibn);

}  // namespace auto_parallel
}  // namespace oneflow
#endif  // ONEFLOW_CORE_AUTO_PARALLEL_SBP_UTIL_H_
