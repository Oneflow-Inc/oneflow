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
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/sbp_infer_util.h"

namespace oneflow {

namespace {

Maybe<std::tuple<Symbol<PlacedNdSbp>, Symbol<PlacedNdSbp>>> RawInOutPlacedNdSbpDimReduce(
    Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out, const Shape& logical_shape) {
  // reduce hierarchy
  ParallelDesc reduced_in_placement = *in->placement();
  ParallelDesc reduced_out_placement = *out->placement();
  NdSbp reduced_in_nd_sbp;
  NdSbp reduced_out_nd_sbp;
  InOutParallelDimReduce(*in->placement(), *out->placement(), *in->nd_sbp(), *out->nd_sbp(),
                         &reduced_in_placement, &reduced_out_placement, &reduced_in_nd_sbp,
                         &reduced_out_nd_sbp, logical_shape);
  return std::make_tuple(
      JUST(PlacedNdSbp::New(SymbolOf(reduced_in_nd_sbp), SymbolOf(reduced_in_placement))),
      JUST(PlacedNdSbp::New(SymbolOf(reduced_out_nd_sbp), SymbolOf(reduced_out_placement))));
}

constexpr auto* InOutPlacedNdSbpDimReduce =
    DECORATE(&RawInOutPlacedNdSbpDimReduce, ThreadLocalCachedCopiable);

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<void> RawCheckParallelDimReduce(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                      const Shape& logical_shape) {
  CHECK_OR_RETURN(in->nd_sbp()->sbp_parallel_size() > 1 || out->nd_sbp()->sbp_parallel_size() > 1);
  CHECK_EQ_OR_RETURN(in->placement()->device_tag(), out->placement()->device_tag());
  Symbol<PlacedNdSbp> reduced_in;
  Symbol<PlacedNdSbp> reduced_out;
  std::tie(reduced_in, reduced_out) = *JUST(InOutPlacedNdSbpDimReduce(in, out, logical_shape));

  for (int64_t in_parallel_id = 0; in_parallel_id < in->placement()->parallel_num();
       ++in_parallel_id) {
    const auto& in_physical_shape =
        JUST(GetPhysicalShape(logical_shape, *in->nd_sbp(), *in->placement(), in_parallel_id));
    const auto& reduce_in_physical_shape = JUST(GetPhysicalShape(
        logical_shape, *reduced_in->nd_sbp(), *reduced_in->placement(), in_parallel_id));
    CHECK_EQ_OR_RETURN(*in_physical_shape, *reduce_in_physical_shape);
  }

  for (int64_t out_parallel_id = 0; out_parallel_id < out->placement()->parallel_num();
       ++out_parallel_id) {
    const auto& out_physical_shape =
        JUST(GetPhysicalShape(logical_shape, *out->nd_sbp(), *out->placement(), out_parallel_id));
    const auto& reduce_out_physical_shape = JUST(GetPhysicalShape(
        logical_shape, *reduced_out->nd_sbp(), *reduced_out->placement(), out_parallel_id));
    CHECK_EQ_OR_RETURN(*out_physical_shape, *reduce_out_physical_shape);
  }

  if (reduced_in->nd_sbp()->sbp_parallel_size() == 1
      && reduced_out->nd_sbp()->sbp_parallel_size() == 1) {
    return Maybe<void>::Ok();
  }
  if ((reduced_in->placement() != in->placement() || reduced_out->placement() != out->placement())
      && reduced_in->placement() == reduced_out->placement()) {
    return Maybe<void>::Ok();
  }
  return Error::CheckFailedError();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckParallelDimReduce =
    DECORATE(&RawCheckParallelDimReduce, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> ParallelDimReduce(const std::shared_ptr<one::Tensor>& tensor,
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

  Symbol<PlacedNdSbp> reduced_in;
  Symbol<PlacedNdSbp> reduced_out;
  std::tie(reduced_in, reduced_out) = *JUST(InOutPlacedNdSbpDimReduce(in, out, *tensor->shape()));

  const std::shared_ptr<one::Tensor>& local_tensor = JUST(tensor->cur_rank_phy_tensor());

  std::shared_ptr<one::Tensor> reduced_in_tensor = JUST(one::functional::LocalToGlobal(
      local_tensor, reduced_in->placement(), *JUST(GetSbpList(reduced_in->nd_sbp())),
      *tensor->shape(), tensor->dtype(), /* sync_data */ false, /*copy=*/false));

  const auto& boxing_interpreter =
      JUST(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          reduced_in->nd_sbp(), reduced_out->nd_sbp(), reduced_in->placement(),
          reduced_out->placement(), *tensor->shape()));
  Singleton<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()),
      /* prefix */ "\t\tInternal boxing of nd-sbp-dim-reduce, ");
  std::shared_ptr<one::Tensor> reduced_out_tensor = JUST(
      boxing_interpreter->Interpret(reduced_in_tensor, reduced_in->nd_sbp(), reduced_out->nd_sbp(),
                                    reduced_in->placement(), reduced_out->placement()));

  const std::shared_ptr<one::Tensor>& reduced_out_local_tensor =
      JUST(reduced_out_tensor->cur_rank_phy_tensor());

  return JUST(one::functional::LocalToGlobal(
      reduced_out_local_tensor, out->placement(), *JUST(GetSbpList(out->nd_sbp())),
      *tensor->shape(), tensor->dtype(), /* sync_data */ false, /*copy=*/false));
}

COMMAND(RegisterBoxingFunction("nd-sbp-dim-reduce", CheckParallelDimReduce, &ParallelDimReduce));

}  // namespace oneflow
