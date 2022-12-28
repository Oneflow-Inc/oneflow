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
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/stride.h"

namespace oneflow {

namespace {

bool RawIsAllBroadcastNdSbpAfterDim(Symbol<NdSbp> nd_sbp, int dim) {
  for (int i = dim; i < nd_sbp->sbp_parallel_size(); ++i) {
    if (!nd_sbp->sbp_parallel(i).has_broadcast_parallel()) { return false; }
  }
  return true;
}

static constexpr auto* IsAllBroadcastNdSbpAfterDim =
    DECORATE(&RawIsAllBroadcastNdSbpAfterDim, ThreadLocalCached);

Maybe<Symbol<SbpParallel>> GetBroadcastSbp() {
  SbpParallel broadcast_sbp;
  broadcast_sbp.mutable_broadcast_parallel();
  return SymbolOf(broadcast_sbp);
}

auto* CachedGetBroadcastSbp = DECORATE(&GetBroadcastSbp, ThreadLocalCached);

// NOLINTBEGIN(maybe-need-error-msg)
Maybe<Shape> CalcLogicalShape4Axis(const Shape& logical_shape, int axis,
                                   Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp) {
  CHECK_LT_OR_RETURN(axis, nd_sbp->sbp_parallel_size());  // Always true
  std::shared_ptr<Shape> sub_logical_shape = std::make_shared<Shape>(logical_shape);

  const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  int64_t parallel_id = JUST(*opt_parallel_id);
  const auto& hierarchy_shape = *parallel_desc->hierarchy();
  Stride hierarchy_stride(hierarchy_shape);

  FOR_RANGE(int64_t, i, 0, axis) {
    const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      int64_t index = CalcIndex4Axis(parallel_id, hierarchy_stride, i);
      int64_t dim = hierarchy_shape.At(i);
      const int64_t split_axis = sbp_parallel.split_parallel().axis();

      if (sub_logical_shape->At(split_axis) > 0) {
        CHECK_GE_OR_RETURN(sub_logical_shape->At(split_axis), dim)
            << Error::RuntimeError() << "The size of tensor (" << sub_logical_shape->At(split_axis)
            << ") at split dimension (" << i
            << ") should be greater than or equal to parallle num (" << dim << ")";
        const BalancedSplitter bs(sub_logical_shape->At(split_axis), dim);
        sub_logical_shape->Set(split_axis, bs.At(index).size());
      }
    }
  }

  return sub_logical_shape;
}

static constexpr auto* GetLogicalShape4Axis =
    DECORATE(&CalcLogicalShape4Axis, ThreadLocalCachedCopiable);

Maybe<int> CalcTheFirstDiffAxisBetweenTwoNdSbp(Symbol<NdSbp> in_nd_sbp, Symbol<NdSbp> out_nd_sbp) {
  CHECK_EQ_OR_RETURN(in_nd_sbp->sbp_parallel_size(),
                     out_nd_sbp->sbp_parallel_size());  // Always true
  int dim = 0;
  for (; dim < in_nd_sbp->sbp_parallel_size(); ++dim) {
    if (in_nd_sbp->sbp_parallel(dim) != out_nd_sbp->sbp_parallel(dim)) { break; }
  }
  return dim;
}

Maybe<one::Tensor> Apply1DBoxing(const std::shared_ptr<one::Tensor>& input, Symbol<NdSbp> in_nd_sbp,
                                 Symbol<NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc) {
  const auto& boxing_interpreter =
      JUST(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *input->shape()));
  Singleton<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()),
      /* prefix */ "\t\tInternal boxing of generic-symmetric-nd-sbp-to-nd-sbp, ");
  return JUST(boxing_interpreter->Interpret(input, in_nd_sbp, out_nd_sbp, in_parallel_desc,
                                            out_parallel_desc));
}

Maybe<void> RawCheckGenericSymmetricNdSbpBoxing(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                                const Shape& logical_shape) {
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->nd_sbp() != out->nd_sbp());
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), out->nd_sbp()->sbp_parallel_size());
  CHECK_GT_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  return Maybe<void>::Ok();
}
// NOLINTEND(maybe-need-error-msg)

static constexpr auto* CheckGenericSymmetricNdSbpBoxing =
    DECORATE(&RawCheckGenericSymmetricNdSbpBoxing, ThreadLocalCachedCopiable);

}  // namespace

Maybe<one::Tensor> GenericSymmetricNdSbpBoxing(const std::shared_ptr<one::Tensor>& input,
                                               Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& in_parallel_desc = in->placement();
  const auto& out_nd_sbp = out->nd_sbp();
  const auto& out_parallel_desc = out->placement();
  std::shared_ptr<one::Tensor> output;

  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
  if (out_parallel_id->has_value()) {
    output = input;

    int first_diff_sbp_dim = JUST(CalcTheFirstDiffAxisBetweenTwoNdSbp(in->nd_sbp(), out_nd_sbp));
    Symbol<SbpParallel> broadcast_sbp = JUST(CachedGetBroadcastSbp());

    const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_parallel_desc));
    int64_t parallel_id = JUST(*opt_parallel_id);
    const auto& hierarchy_shape = *in_parallel_desc->hierarchy();
    Stride hierarchy_stride(hierarchy_shape);

    const auto& logical_shape = input->shape();

    // Convert input to broadcast tensor step by step
    // e.g.
    // If in_nd_sbp is (S(0), B, S(0)), (S(0), S(0), S(1))
    // Altered state of sbp is (S(0), B, S(0)) -> (S(0), B, B)
    for (int64_t i = out_nd_sbp->sbp_parallel_size() - 1; i >= first_diff_sbp_dim; --i) {
      const auto& nd_sbp = JUST(output->nd_sbp());
      const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
      if (sbp_parallel.has_broadcast_parallel()) { continue; }

      const auto& one_dim_nd_sbp = JUST(SbpToNdSbp(sbp_parallel));
      const auto& sub_logical_shape =
          *JUST(GetLogicalShape4Axis(*logical_shape, i, in_parallel_desc, nd_sbp));
      std::shared_ptr<one::Tensor> local_tensor = JUST(output->cur_rank_phy_tensor());
      const auto& sub_parallel_desc = JUST(CalcSubParallelDesc4Axis(in_parallel_desc, i));

      int64_t index = CalcIndex4Axis(parallel_id, hierarchy_stride, i);

      const auto& physical_shape =
          JUST(GetPhysicalShape(sub_logical_shape, *one_dim_nd_sbp, *sub_parallel_desc, index));
      CHECK_EQ_OR_RETURN(*physical_shape, *local_tensor->shape())
          << Error::RuntimeError() << "Invalid input tensor, size of local tensor ("
          << local_tensor->shape()->ToString() << ") does not match global tensor ("
          << logical_shape->ToString() << ")!";
      std::shared_ptr<one::Tensor> sub_global_tensor = JUST(one::functional::LocalToGlobal(
          local_tensor, sub_parallel_desc, *JUST(GetSbpList(one_dim_nd_sbp)), sub_logical_shape,
          local_tensor->dtype(), /* sync_data */ false, /*copy=*/false));

      sub_global_tensor =
          JUST(Apply1DBoxing(sub_global_tensor, one_dim_nd_sbp, JUST(SbpToNdSbp(broadcast_sbp)),
                             sub_parallel_desc, sub_parallel_desc));

      local_tensor = JUST(sub_global_tensor->cur_rank_phy_tensor());

      const auto& new_nd_sbp = JUST(SetSbpAtAxis(*nd_sbp, *broadcast_sbp, i));

      output = JUST(one::functional::LocalToGlobal(
          local_tensor, in_parallel_desc, *JUST(GetSbpList(new_nd_sbp)), *logical_shape,
          local_tensor->dtype(), /* sync_data */ false, /*copy=*/false));
    }

    CHECK_OR_RETURN(IsAllBroadcastNdSbpAfterDim(JUST(output->nd_sbp()), first_diff_sbp_dim))
        << Error::RuntimeError()
        << "Compute generic-symmetric-nd-sbp-to-nd-sbp failed. Please submit an issue in "
           "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
           "possible";

    // Convert broadcast tensor to output with out_nd_sbp data step by step
    // e.g.
    // If out_nd_sbp is (S(0), S(0), S(1))
    // Altered state of sbp is (S(0), B, B) -> (S(0), S(0), B) -> (S(0), S(0), S(1))
    std::shared_ptr<Shape> sub_logical_shape = JUST(GetLogicalShape4Axis(
        *logical_shape, first_diff_sbp_dim, in_parallel_desc, JUST(output->nd_sbp())));
    for (int64_t i = first_diff_sbp_dim; i < out_nd_sbp->sbp_parallel_size(); ++i) {
      const auto& sbp_parallel = out_nd_sbp->sbp_parallel(i);
      if (sbp_parallel.has_broadcast_parallel()) { continue; }

      const auto& nd_sbp = JUST(output->nd_sbp());

      const auto& sub_parallel_desc = JUST(CalcSubParallelDesc4Axis(in_parallel_desc, i));

      std::shared_ptr<one::Tensor> local_tensor = JUST(output->cur_rank_phy_tensor());

      std::shared_ptr<one::Tensor> sub_global_tensor = JUST(one::functional::LocalToGlobal(
          local_tensor, sub_parallel_desc, *JUST(GetSbpList(JUST(SbpToNdSbp(broadcast_sbp)))),
          *sub_logical_shape, local_tensor->dtype(), /* sync_data */ false, /*copy=*/false));

      const auto& one_dim_nd_sbp = JUST(SbpToNdSbp(sbp_parallel));
      sub_global_tensor = JUST(Apply1DBoxing(sub_global_tensor, JUST(SbpToNdSbp(broadcast_sbp)),
                                             one_dim_nd_sbp, sub_parallel_desc, sub_parallel_desc));

      local_tensor = JUST(sub_global_tensor->cur_rank_phy_tensor());

      int64_t index = CalcIndex4Axis(parallel_id, hierarchy_stride, i);
      const auto& physical_shape =
          JUST(GetPhysicalShape(*sub_logical_shape, *one_dim_nd_sbp, *sub_parallel_desc, index));
      CHECK_EQ_OR_RETURN(*physical_shape, *local_tensor->shape())
          << Error::RuntimeError()
          << "Compute generic-symmetric-nd-sbp-to-nd-sbp failed. Please submit an issue in "
             "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
             "possible";

      const auto& new_nd_sbp = JUST(SetSbpAtAxis(*nd_sbp, sbp_parallel, i));

      output = JUST(one::functional::LocalToGlobal(
          local_tensor, in_parallel_desc, *JUST(GetSbpList(new_nd_sbp)), *logical_shape,
          local_tensor->dtype(), /* sync_data */ false, /*copy=*/false));
      // physical_shape of this axis is logical shape of next axis
      sub_logical_shape = physical_shape;
    }
  } else {
    one::GlobalTensorMeta tensor_meta(*input->shape(), input->dtype()->data_type(), out_nd_sbp,
                                      out_parallel_desc);
    const auto& tensor_impl =
        JUST(one::EagerGlobalTensorImpl::New(SymbolOf(tensor_meta), input->requires_grad(), false));
    output = std::make_shared<one::GlobalTensor>(tensor_impl);
  }

  return output;
}

COMMAND(RegisterBoxingFunction("generic-symmetric-nd-sbp-to-nd-sbp",
                               CheckGenericSymmetricNdSbpBoxing, &GenericSymmetricNdSbpBoxing));

}  // namespace oneflow
