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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<void> RawCheckSymmetricXToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsAllBroadcastNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->device_type() == DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymmetricXToB = DECORATE(&RawCheckSymmetricXToB, ThreadLocal);

}  // namespace

Maybe<one::Tensor> SymmetricXToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                 Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  if (EagerBoxingInterpreterUtil::IsBroadcastSbp(tensor_nd_sbp->sbp_parallel(0))) { return tensor; }
  if (EagerBoxingInterpreterUtil::IsPartialSumSbp(tensor_nd_sbp->sbp_parallel(0))) {
    const auto& NcclPToBBoxingFunction = *JUST(GetBoxingFunction("nccl-p-to-b", in, out));
    return JUST(NcclPToBBoxingFunction(tensor, in, out));
  }
  if (EagerBoxingInterpreterUtil::IsSplitSbp(tensor_nd_sbp->sbp_parallel(0), 0)) {
    const auto& NcclSToBBoxingFunction = *JUST(GetBoxingFunction("nccl-s-to-b", in, out));
    return JUST(NcclSToBBoxingFunction(tensor, in, out));
  }
  UNIMPLEMENTED_THEN_RETURN();
}

COMMAND(RegisterBoxingFunction("symmetric-x-to-b", CheckSymmetricXToB, &SymmetricXToB));

}  // namespace oneflow
