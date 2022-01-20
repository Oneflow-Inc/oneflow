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

bool IsAllBroadcastNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

bool IsAllPartialSumNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_partial_sum_parallel()) { return false; }
  }
  return true;
}

bool IsAllSplitNdSbp(Symbol<cfg::NdSbp> nd_sbp, int64_t axis) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!(sbp_parallel.has_split_parallel() && sbp_parallel.split_parallel().axis() == axis)) {
      return false;
    }
  }
  return true;
}

Maybe<void> RawCheckCclP2B(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(IsAllPartialSumNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(out->nd_sbp()));

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclP2B = DECORATE(&RawCheckCclP2B, ThreadLocalCopiable);

Maybe<void> RawCheckCclP2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsAllPartialSumNdSbp(in->nd_sbp()));
  CHECK_OR_RETURN(IsAllSplitNdSbp(out->nd_sbp(), 0));

  CHECK_OR_RETURN(logical_shape.At(0) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclP2S = DECORATE(&RawCheckCclP2S, ThreadLocalCopiable);

Maybe<void> RawCheckCclS2B(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(IsAllSplitNdSbp(in->nd_sbp(), 0));
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(out->nd_sbp()));

  CHECK_OR_RETURN(logical_shape.At(0) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclS2B = DECORATE(&RawCheckCclS2B, ThreadLocalCopiable);

Maybe<void> RawCheckCclS2S(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                           const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);

  CHECK_OR_RETURN(in->nd_sbp()->sbp_parallel(0).has_split_parallel());
  CHECK_OR_RETURN(out->nd_sbp()->sbp_parallel(0).has_split_parallel());
  CHECK_NE_OR_RETURN(in->nd_sbp()->sbp_parallel(0).split_parallel().axis(),
                     out->nd_sbp()->sbp_parallel(0).split_parallel().axis());

  int64_t in_split_axis = in->nd_sbp()->sbp_parallel(0).split_parallel().axis();
  int64_t out_split_axis = out->nd_sbp()->sbp_parallel(0).split_parallel().axis();
  CHECK_OR_RETURN(logical_shape.At(in_split_axis) % in->placement()->parallel_num() == 0);
  CHECK_OR_RETURN(logical_shape.At(out_split_axis) % in->placement()->parallel_num() == 0);

  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), DeviceType::kCPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckCclS2S = DECORATE(&RawCheckCclS2S, ThreadLocalCopiable);

}  // namespace

Maybe<one::Tensor> CclP2B(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  return JUST(one::functional::ConsistentAllReduce(tensor));
}

Maybe<one::Tensor> CclP2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());

  return JUST(one::functional::ConsistentReduceScatter(tensor, "sum"));
}

Maybe<one::Tensor> CclS2B(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  return JUST(one::functional::ConsistentAllGather(tensor));
}

Maybe<one::Tensor> CclS2S(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                          Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  return JUST(one::functional::ConsistentS2S(tensor, *JUST(GetSbpList(out->nd_sbp()))));
}

COMMAND(RegisterBoxingFunction("ccl-p-to-b", CheckCclP2B, &CclP2B));
COMMAND(RegisterBoxingFunction("ccl-p-to-s", CheckCclP2S, &CclP2S));
COMMAND(RegisterBoxingFunction("ccl-s-to-b", CheckCclS2B, &CclS2B));
COMMAND(RegisterBoxingFunction("ccl-s-to-s", CheckCclS2S, &CclS2S));

}  // namespace oneflow
