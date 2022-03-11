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
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

using IndexVector = DimVector;
using StrideVector = DimVector;

bool RawIsAllBroadcastNdSbp(Symbol<NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

static constexpr auto* IsAllBroadcastNdSbp = DECORATE(&RawIsAllBroadcastNdSbp, ThreadLocal);

Maybe<Symbol<NdSbp>> GetAllBroadcastNdSbp(int64_t ndim) {
  NdSbp broadcast_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  }
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetAllBroadcastNdSbp = DECORATE(&GetAllBroadcastNdSbp, ThreadLocal);

void GetStrideVector(const Shape& shape, StrideVector* strides) {
  strides->resize(shape.NumAxes());
  for (int i = 0; i < shape.NumAxes(); ++i) { strides->at(i) = shape.Count(i + 1); }
}

int64_t GetIndex4AxisFromOffset(int64_t offset, const StrideVector& strides, int axis) {
  CHECK_LT(axis, strides.size());
  int64_t index = -1;
  for (int i = 0; i < axis + 1; ++i) {
    index = offset / strides.at(i);
    offset = offset % strides.at(i);
  }
  CHECK_NE(index, -1);
  return index;
}

Maybe<Shape> CalLogicalShape4Axis(const Shape& logical_shape, int axis,
                                  Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp) {
  CHECK_LT_OR_RETURN(axis, nd_sbp->sbp_parallel_size());
  std::shared_ptr<Shape> sub_logical_shape = std::make_shared<Shape>(logical_shape);

  const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
  int64_t parallel_id = JUST(*opt_parallel_id);
  const auto& hierarchy_shape = *parallel_desc->hierarchy();
  StrideVector hierarchy_strides{};
  GetStrideVector(hierarchy_shape, &hierarchy_strides);

  FOR_RANGE(int64_t, i, 0, axis) {
    const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      int64_t index = GetIndex4AxisFromOffset(parallel_id, hierarchy_strides, i);
      int64_t dim = hierarchy_shape.At(i);
      const int64_t split_axis = sbp_parallel.split_parallel().axis();

      if (sub_logical_shape->At(split_axis) > 0) {
        CHECK_GE_OR_RETURN(sub_logical_shape->At(split_axis), dim);
        const BalancedSplitter bs(sub_logical_shape->At(split_axis), dim);
        sub_logical_shape->Set(split_axis, bs.At(index).size());
      }
    }
  }

  return sub_logical_shape;
}

static constexpr auto* GetLogicalShape4Axis = DECORATE(&CalLogicalShape4Axis, ThreadLocalCopiable);

Maybe<Symbol<NdSbp>> ConvertSbpParallelToNdSbp(const SbpParallel& sbp) {
  NdSbp nd_sbp;
  *nd_sbp.mutable_sbp_parallel()->Add() = sbp;
  return SymbolOf(nd_sbp);
}

static constexpr auto* CachedConvertSbpParallelToNdSbp =
    DECORATE(&ConvertSbpParallelToNdSbp, ThreadLocalCopiable);

Maybe<Symbol<NdSbp>> ConvertToBroadcastAtAxis(Symbol<NdSbp> nd_sbp, int axis) {
  CHECK_LT_OR_RETURN(axis, nd_sbp->sbp_parallel_size());
  NdSbp out_nd_sbp = *nd_sbp;
  out_nd_sbp.mutable_sbp_parallel(axis)->mutable_broadcast_parallel();
  return SymbolOf(out_nd_sbp);
}

static constexpr auto* CachedConvertToBroadcastAtAxis =
    DECORATE(&ConvertToBroadcastAtAxis, ThreadLocal);

Maybe<Symbol<NdSbp>> ReplaceSbpAtAxis(Symbol<NdSbp> nd_sbp, int axis, const SbpParallel& sbp) {
  CHECK_LT_OR_RETURN(axis, nd_sbp->sbp_parallel_size());
  NdSbp out_nd_sbp = *nd_sbp;
  *out_nd_sbp.mutable_sbp_parallel(axis) = sbp;
  return SymbolOf(out_nd_sbp);
}

auto* CachedReplaceSbpAtAxis = DECORATE(&ReplaceSbpAtAxis, ThreadLocalCopiable);

Maybe<one::Tensor> Apply1DBoxing(const std::shared_ptr<one::Tensor>& input, Symbol<NdSbp> in_nd_sbp,
                                 Symbol<NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
                                 Symbol<ParallelDesc> out_parallel_desc) {
  const auto& boxing_interpreter =
      JUST(Global<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *input->shape()));
  Global<const EagerBoxingLogger>::Get()->Log(
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

static constexpr auto* CheckGenericSymmetricNdSbpBoxing =
    DECORATE(&RawCheckGenericSymmetricNdSbpBoxing, ThreadLocalCopiable);

}  // namespace

Maybe<one::Tensor> GenericSymmetricNdSbpBoxing(const std::shared_ptr<one::Tensor>& input,
                                               Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  const auto& in_parallel_desc = in->placement();
  const auto& out_nd_sbp = out->nd_sbp();
  const auto& out_parallel_desc = out->placement();
  CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
  std::shared_ptr<one::Tensor> output;

  const auto& out_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
  if (out_parallel_id->has_value()) {
    output = input;
    Symbol<NdSbp> broadcast_nd_sbp = JUST(CachedGetAllBroadcastNdSbp(1));

    const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(in_parallel_desc));
    int64_t parallel_id = JUST(*opt_parallel_id);
    const auto& hierarchy_shape = *in_parallel_desc->hierarchy();
    StrideVector hierarchy_strides{};
    GetStrideVector(hierarchy_shape, &hierarchy_strides);

    const auto& logical_shape = *input->shape();

    // Convert input to broadcast tensor step by step
    // e.g.
    // If in_nd_sbp is (S(0), B, S(0))
    // Altered state of sbp is (S(0), B, S(0)) -> (S(0), B, B) -> (B, B, B)
    for (int64_t i = out_nd_sbp->sbp_parallel_size() - 1; i >= 0; --i) {
      const auto& nd_sbp = JUST(output->nd_sbp());
      const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
      if (sbp_parallel.has_broadcast_parallel()) { continue; }

      const auto& one_dim_nd_sbp = JUST(CachedConvertSbpParallelToNdSbp(sbp_parallel));
      const auto& sub_logical_shape =
          *JUST(GetLogicalShape4Axis(logical_shape, i, in_parallel_desc, nd_sbp));
      std::shared_ptr<one::Tensor> local_tensor = JUST(output->cur_rank_phy_tensor());
      const auto& sub_parallel_desc = JUST(CalcSubParallelDesc4Axis(in_parallel_desc, i));

      int64_t index = GetIndex4AxisFromOffset(parallel_id, hierarchy_strides, i);

      const auto& physical_shape =
          *JUST(GetPhysicalShape(sub_logical_shape, *one_dim_nd_sbp, *sub_parallel_desc, index));

      CHECK_EQ_OR_RETURN(physical_shape, *local_tensor->shape());
      std::shared_ptr<one::Tensor> sub_global_tensor = JUST(one::functional::LocalToConsistent(
          local_tensor, sub_parallel_desc, *JUST(GetSbpList(one_dim_nd_sbp)), sub_logical_shape,
          local_tensor->dtype()));

      sub_global_tensor = JUST(Apply1DBoxing(sub_global_tensor, one_dim_nd_sbp, broadcast_nd_sbp,
                                             sub_parallel_desc, sub_parallel_desc));

      local_tensor = JUST(sub_global_tensor->cur_rank_phy_tensor());

      const auto& new_nd_sbp = JUST(CachedConvertToBroadcastAtAxis(nd_sbp, i));

      output = JUST(one::functional::LocalToConsistent(local_tensor, in_parallel_desc,
                                                       *JUST(GetSbpList(new_nd_sbp)), logical_shape,
                                                       local_tensor->dtype()));
    }

    CHECK_OR_RETURN(IsAllBroadcastNdSbp(JUST(output->nd_sbp())));

    // Convert broadcast tensor to output with out_nd_sbp data step by step
    // e.g.
    // If out_nd_sbp is (S(1), S(0), S(1))
    // Altered state of sbp is (B, B, B) -> (S(1), B, B) -> (S(1), S(0), B) -> (S(1), S(0), S(1))
    for (int64_t i = 0; i < out_nd_sbp->sbp_parallel_size(); ++i) {
      const auto& sbp_parallel = out_nd_sbp->sbp_parallel(i);
      if (sbp_parallel.has_broadcast_parallel()) { continue; }

      const auto& nd_sbp = JUST(output->nd_sbp());
      const auto& sub_logical_shape =
          *JUST(GetLogicalShape4Axis(logical_shape, i, in_parallel_desc, nd_sbp));

      const auto& sub_parallel_desc = JUST(CalcSubParallelDesc4Axis(in_parallel_desc, i));

      std::shared_ptr<one::Tensor> local_tensor = JUST(output->cur_rank_phy_tensor());

      std::shared_ptr<one::Tensor> sub_global_tensor = JUST(one::functional::LocalToConsistent(
          local_tensor, sub_parallel_desc, *JUST(GetSbpList(broadcast_nd_sbp)), sub_logical_shape,
          local_tensor->dtype()));

      const auto& one_dim_nd_sbp = JUST(CachedConvertSbpParallelToNdSbp(sbp_parallel));
      sub_global_tensor = JUST(Apply1DBoxing(sub_global_tensor, broadcast_nd_sbp, one_dim_nd_sbp,
                                             sub_parallel_desc, sub_parallel_desc));

      local_tensor = JUST(sub_global_tensor->cur_rank_phy_tensor());

      int64_t index = GetIndex4AxisFromOffset(parallel_id, hierarchy_strides, i);
      const auto& physical_shape =
          *JUST(GetPhysicalShape(sub_logical_shape, *one_dim_nd_sbp, *sub_parallel_desc, index));
      CHECK_EQ_OR_RETURN(physical_shape, *local_tensor->shape());

      const auto& new_nd_sbp = JUST(CachedReplaceSbpAtAxis(nd_sbp, i, sbp_parallel));

      output = JUST(one::functional::LocalToConsistent(local_tensor, in_parallel_desc,
                                                       *JUST(GetSbpList(new_nd_sbp)), logical_shape,
                                                       local_tensor->dtype()));
    }
  } else {
    one::ConsistentTensorMeta tensor_meta(input->shape(), input->dtype()->data_type(), out_nd_sbp,
                                          out_parallel_desc);
    const auto& tensor_impl = JUST(
        one::EagerConsistentTensorImpl::New(SymbolOf(tensor_meta), input->requires_grad(), false));
    output = std::make_shared<one::ConsistentTensor>(tensor_impl);
  }

  return output;
}

COMMAND(RegisterBoxingFunction("generic-symmetric-nd-sbp-to-nd-sbp",
                               CheckGenericSymmetricNdSbpBoxing, &GenericSymmetricNdSbpBoxing));

}  // namespace oneflow
