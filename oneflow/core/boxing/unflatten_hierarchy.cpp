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

namespace oneflow {

namespace {

Maybe<void> RawCheckUnflattenHierarchy(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
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
  return Maybe<void>::Ok();
}

}  // namespace

static constexpr auto* CheckUnflattenHierarchy = DECORATE(&RawCheckUnflattenHierarchy, ThreadLocal);

Maybe<one::Tensor> UnflattenHierarchy(const std::shared_ptr<one::Tensor>& tensor,
                                      Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  const auto& local_tensor = JUST(tensor->cur_rank_phy_tensor());
  const auto& sbp_list = JUST(GetSbpList(out->nd_sbp()));
  return JUST(one::functional::LocalToConsistent(local_tensor, out->placement(), *sbp_list,
                                                 *tensor->shape(), tensor->dtype()));
}

COMMAND(RegisterBoxingFunction("unflatten-hierarchy", CheckUnflattenHierarchy,
                               &UnflattenHierarchy));

}  // namespace oneflow
