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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckUnflattenHierarchy(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                       const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_GT_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  for (int i = 0; i < out->nd_sbp()->sbp_parallel_size(); ++i) {
    const auto& sbp_parallel = out->nd_sbp()->sbp_parallel(i);
    CHECK_OR_RETURN(sbp_parallel == out->nd_sbp()->sbp_parallel(0)) << "nd_sbp axis: " << i;
  }
  CHECK_EQ_OR_RETURN(in->placement()->device_type(), out->placement()->device_type());
  CHECK_EQ_OR_RETURN(in->placement()->parallel_num(), out->placement()->parallel_num());
  ParallelConf unflattened_parallel_conf(in->placement()->parallel_conf());
  unflattened_parallel_conf.mutable_hierarchy()->CopyFrom(
      out->placement()->parallel_conf().hierarchy());
  const auto& unflatten_placement = SymbolOf(ParallelDesc(unflattened_parallel_conf));
  CHECK_OR_RETURN(unflatten_placement == out->placement())
      << "The output placement is not a hierarch-unflattened version of the input placement";
  for (int64_t in_parallel_id = 0; in_parallel_id < in->placement()->parallel_num();
       ++in_parallel_id) {
    const auto& in_physical_shape =
        JUST(GetPhysicalShape(logical_shape, *in->nd_sbp(), *in->placement(), in_parallel_id));
    const auto& out_physical_shape =
        JUST(GetPhysicalShape(logical_shape, *out->nd_sbp(), *out->placement(), in_parallel_id));
    CHECK_EQ_OR_RETURN(*in_physical_shape, *out_physical_shape);
  }
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

}  // namespace

static constexpr auto* CheckUnflattenHierarchy =
    DECORATE(&RawCheckUnflattenHierarchy, ThreadLocalCachedCopiable);

Maybe<one::Tensor> UnflattenHierarchy(const std::shared_ptr<one::Tensor>& tensor,
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
  const auto& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  return JUST(one::functional::LocalToGlobal(local_tensor, out->placement(), *sbp_list,
                                             *tensor->shape(), tensor->dtype(),
                                             /* sync_data */ false, /*copy=*/true));
}

COMMAND(RegisterBoxingFunction("unflatten-hierarchy", CheckUnflattenHierarchy,
                               &UnflattenHierarchy));

}  // namespace oneflow
