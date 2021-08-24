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
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/device.h"
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
  std::shared_ptr<one::Tensor> local_tensor = JUST(input->cur_rank_phy_tensor());
  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
  if (!out_parallel_id->has_value()) {
    std::string device_type = Device::Type4DeviceTag(in_parallel_desc->device_tag());
    local_tensor = JUST(one::functional::Empty(
        *JUST(GetPhysicalShape(*input->shape(), *in_nd_sbp, *in_parallel_desc, 0)), input->dtype(),
        JUST(Device::New(device_type))));
  }
  const auto& sbp_list = JUST(GetSbpList(out_nd_sbp));
  const auto& tensor =
      JUST(one::functional::ToConsistent(local_tensor, out_parallel_desc, *sbp_list, {}));
  CHECK_OR_RETURN(tensor->is_consistent());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == out_parallel_desc);
  return tensor;
}

namespace {

Maybe<bool> IgnoringDeviceTypeEqual(Symbol<ParallelDesc> lhs, Symbol<ParallelDesc> rhs) {
  return lhs == JUST(ReplaceDeviceType(rhs, lhs->device_type()));
}

}  // namespace

Maybe<void> CheckCudaCopyH2D(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  bool equal = JUST(IgnoringDeviceTypeEqual(in->placement(), out->placement()));
  CHECK_OR_RETURN(equal);
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kGPU);
  CHECK_OR_RETURN(in->nd_sbp() == out->nd_sbp());
  return Maybe<void>::Ok();
}

Maybe<void> CheckCudaCopyD2H(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  bool equal = JUST(IgnoringDeviceTypeEqual(in->placement(), out->placement()));
  CHECK_OR_RETURN(equal);
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kGPU);
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kCPU);
  CHECK_OR_RETURN(in->nd_sbp() == out->nd_sbp());
  return Maybe<void>::Ok();
}

Maybe<one::Tensor> CudaCopy(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                            Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  const auto& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(), *sbp_list,
                                                 *tensor->shape(), tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("cuda-copy-h2d", &CheckCudaCopyH2D, &CudaCopy));
COMMAND(RegisterBoxingFunction("cuda-copy-d2h", &CheckCudaCopyD2H, &CudaCopy));

}  // namespace oneflow
