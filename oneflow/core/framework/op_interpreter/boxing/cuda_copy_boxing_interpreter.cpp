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
#include "oneflow/core/framework/op_interpreter/boxing/cuda_copy_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

Maybe<one::Tensor> CudaCopyBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(in_nd_sbp == out_nd_sbp);
  const auto& new_tag_in_parallel_desc =
      JUST(ReplaceDeviceType(in_parallel_desc, out_parallel_desc->device_type()));
  CHECK_OR_RETURN(new_tag_in_parallel_desc == out_parallel_desc);
  const auto& local_tensor = JUST(input->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out_nd_sbp));
  const auto& tensor =
      JUST(one::functional::ToConsistent(local_tensor, out_parallel_desc, *sbp_list, {}));
  CHECK_OR_RETURN(tensor->is_consistent());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return tensor;
}

}  // namespace oneflow
