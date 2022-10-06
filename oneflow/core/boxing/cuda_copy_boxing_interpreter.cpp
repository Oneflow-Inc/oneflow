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
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

Maybe<bool> IgnoringDeviceTypeEqual(Symbol<ParallelDesc> lhs, Symbol<ParallelDesc> rhs) {
  return lhs == JUST(ReplaceDeviceType(rhs, lhs->device_type()));
}

}  // namespace

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> CheckCopyH2D(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                         const Shape& logical_shape) {
  bool equal = JUST(IgnoringDeviceTypeEqual(in->placement(), out->placement()));
  CHECK_OR_RETURN(equal);
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  CHECK_NE_OR_RETURN(out->placement()->device_type(), DeviceType::kCPU);
  CHECK_OR_RETURN(in->nd_sbp() == out->nd_sbp());
  return Maybe<void>::Ok();
}

Maybe<void> CheckCopyD2H(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                         const Shape& logical_shape) {
  bool equal = JUST(IgnoringDeviceTypeEqual(in->placement(), out->placement()));
  CHECK_OR_RETURN(equal);
  CHECK_NE_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  CHECK_EQ_OR_RETURN(out->placement()->device_type(), DeviceType::kCPU);
  CHECK_OR_RETURN(in->nd_sbp() == out->nd_sbp());
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

Maybe<one::Tensor> CopyBoxingFunction(const std::shared_ptr<one::Tensor>& tensor,
                                      Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp())
      << Error::RuntimeError() << "The sbp of input tensor (" << NdSbpToString(tensor_nd_sbp)
      << ") must match the input sbp (" << NdSbpToString(in->nd_sbp()) << ")";
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement())
      << Error::RuntimeError() << "The placement of input tensor ("
      << *JUST(PlacementToString(tensor_placement)) << ") must match the input placement ("
      << *JUST(PlacementToString(in->placement())) << ")";
  const std::shared_ptr<one::Tensor>& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  return JUST(one::functional::LocalToGlobal(local_tensor, out->placement(), *sbp_list,
                                             *tensor->shape(), tensor->dtype(),
                                             /* sync_data */ false, /*copy=*/false));
}

COMMAND(RegisterBoxingFunction("copy-h2d", &CheckCopyH2D, &CopyBoxingFunction));
COMMAND(RegisterBoxingFunction("copy-d2h", &CheckCopyD2H, &CopyBoxingFunction));

}  // namespace oneflow
