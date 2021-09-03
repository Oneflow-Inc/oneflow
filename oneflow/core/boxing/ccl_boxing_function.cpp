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
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

namespace {

bool IsSplitSbp(Symbol<cfg::SbpParallel> sbp_parallel) {
  return sbp_parallel->has_split_parallel();
}

Maybe<void> RawCheckCclS2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(IsSplitSbp(in->nd_sbp()->sbp_parallel(0)));
  CHECK_OR_RETURN(IsSplitSbp(out->nd_sbp()->sbp_parallel(0)));
  CHECK_NE_OR_RETURN(in->nd_sbp()->sbp_parallel(0).split_parallel().axis(),
                     out->nd_sbp()->sbp_parallel(0).split_parallel().axis());

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclS2S = DECORATE(&RawCheckCclS2S, ThreadLocal);

}  // namespace

Maybe<one::Tensor> CclS2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  return JUST(one::functional::ConsistentS2S(tensor, *JUST(GetSbpList(out->nd_sbp()))));
}

COMMAND(RegisterBoxingFunction("ccl-s-to-s", CheckCclS2S, &CclS2S));

}  // namespace oneflow
