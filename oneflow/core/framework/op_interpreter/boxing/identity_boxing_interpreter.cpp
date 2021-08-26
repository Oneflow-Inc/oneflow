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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interpreter/boxing/identity_boxing_interpreter.h"

namespace oneflow {

Maybe<one::Tensor> IdentityBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::NdSbp> in_nd_sbp,
    Symbol<cfg::NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  CHECK_OR_RETURN(in_parallel_desc->parallel_num() == 1 || in_nd_sbp == out_nd_sbp);
  // reset sbp if parallel_num == 1 and reset ConsistentId
  std::shared_ptr<one::Tensor> tensor = JUST(input->cur_rank_phy_tensor());
  return one::functional::ToConsistent(tensor, out_parallel_desc, *JUST(GetSbpList(out_nd_sbp)),
                                       GetNoneSbpList());
}

namespace {

Maybe<void> RawCheckIdentity(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->parallel_num() == 1 || in->nd_sbp() == out->nd_sbp());
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<one::Tensor> GetIdentity(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                               Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  if (tensor_nd_sbp == out->nd_sbp()) { return tensor; }
  const auto& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(), *sbp_list,
                                                 *tensor->shape(), tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("identity", DECORATE(&RawCheckIdentity, ThreadLocal), &GetIdentity));
}  // namespace oneflow
