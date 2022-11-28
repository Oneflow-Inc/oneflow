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
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

int64_t GetMappedParallelId(const int64_t from_parallel_id, const ParallelDesc& from_parallel_desc,
                            const ParallelDesc& to_parallel_desc);

void GetRankSendRecvIntersection(int64_t parallel_id, const ParallelDesc& parallel_desc,
                                 const ParallelDesc& in_parallel_desc,
                                 const ParallelDesc& out_parallel_desc, const NdSbp& in_nd_sbp,
                                 const NdSbp& out_nd_sbp, const Shape& logical_shape,
                                 std::vector<TensorSliceView>* send_intersections,
                                 std::vector<TensorSliceView>* recv_intersections);

}  // namespace oneflow
