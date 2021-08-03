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

#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

Maybe<HashMap<int64_t, std::shared_ptr<FlatShape>>> All2AllSyncShape(const Shape& shape) {
  const auto& rpc_token = JUST(RpcToken::AcquireCtrlRpcToken(kRankGroupRpcCmdAll2AllSyncShape));
  const auto& send_buffer = JUST(FlatShape::New(shape));
  const auto& map = std::make_shared<HashMap<int64_t, std::shared_ptr<FlatShape>>>();
  map->emplace(GlobalProcessCtx::Rank(), send_buffer);
  NaiveAsyncRpcCtx ctx(
      rpc_token,
      [send_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = send_buffer.get();
        *size = sizeof(FlatShape);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [map](int64_t rank, void** buffer, std::size_t* size,
            std::function<void()>* Cb) -> Maybe<void> {
        const auto& recv_buffer = std::make_shared<FlatShape>();
        recv_buffer->clear();
        *buffer = recv_buffer.get();
        *size = sizeof(FlatShape);
        *Cb = [recv_buffer] {};
        CHECK_OR_RETURN(map->emplace(rank, recv_buffer).second);
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  JUST(RpcUtil::BroadcastToAllOtherRanks(rank_group, rpc_token, &ctx));
  JUST(RpcUtil::CollectFromAllOtherRanks(rank_group, rpc_token, &ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(ctx, RpcUtil::TimeoutSeconds()));
  return map;
}

Maybe<Shape> GetConcatenatedShape(
    const HashMap<int64_t, std::shared_ptr<FlatShape>>& rank2flat_shape,
    Symbol<ParallelDesc> parallel_desc, int64_t concat_axis) {
  const auto& GetRankPhyShapeByParallelId = [&](int64_t parallel_id) -> Maybe<FlatShape> {
    int64_t machine_id = JUST(parallel_desc->MachineId4ParallelId(parallel_id));
    return JUST(MapAt(rank2flat_shape, machine_id));
  };
  const auto& first_flat_shape = JUST(GetRankPhyShapeByParallelId(0));
  CHECK_GE_OR_RETURN(concat_axis, 0);
  CHECK_LT_OR_RETURN(concat_axis, first_flat_shape->NumAxes());
  int64_t logical_concat_dim = first_flat_shape->At(concat_axis);
  for (int parallel_id = 1; parallel_id < parallel_desc->parallel_num(); ++parallel_id) {
    const auto& rank_flat_shape = JUST(GetRankPhyShapeByParallelId(parallel_id));
    CHECK_EQ_OR_RETURN(rank_flat_shape->NumAxes(), first_flat_shape->NumAxes());
    logical_concat_dim += rank_flat_shape->At(concat_axis);
  }
  BalancedSplitter bs(logical_concat_dim, parallel_desc->parallel_num());
  CHECK_EQ_OR_RETURN(first_flat_shape->At(concat_axis), bs.At(0).size());
  const auto& shape = JUST(first_flat_shape->ToShape());
  shape->Set(concat_axis, logical_concat_dim);
  for (int parallel_id = 1; parallel_id < parallel_desc->parallel_num(); ++parallel_id) {
    const auto& rank_flat_shape = JUST(GetRankPhyShapeByParallelId(parallel_id));
    for (int i = 0; i < shape->NumAxes(); ++i) {
      if (i == concat_axis) {
        CHECK_EQ_OR_RETURN(rank_flat_shape->At(i), bs.At(parallel_id).size());
      } else {
        CHECK_EQ_OR_RETURN(rank_flat_shape->At(i), shape->At(i));
      }
    }
  }
  return shape;
}

Maybe<Shape> GetConsistentShape(const Shape& physical_shape, Symbol<ParallelDesc> parallel_desc,
                                Symbol<cfg::ParallelDistribution> parallel_distribution) {
  if (parallel_distribution->sbp_parallel_size() == 1
      && parallel_distribution->sbp_parallel(0).has_split_parallel()) {
    const auto& rank2flat_shape = JUST(All2AllSyncShape(physical_shape));
    int64_t concat_axis = parallel_distribution->sbp_parallel(0).split_parallel().axis();
    return GetConcatenatedShape(*rank2flat_shape, parallel_desc, concat_axis);
  } else {
    // no need to check shape across ranks because we will do it later by checking tensor meta.
    return GetLogicalShape(physical_shape, *parallel_distribution, *parallel_desc);
  }
}

}  //  namespace

class LocalToConsistentFunctor {
 public:
  LocalToConsistentFunctor() {
    op_ =
        CHECK_JUST(one::CastToConsistentOpExpr::New(*CHECK_JUST(UniqueStr("cast_to_consistent"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> parallel_desc,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                           const Optional<Shape>& shape) const {
    CHECK_OR_RETURN(x->is_local()) << Error::Unimplemented() << "local tensors supported only";
    const auto& device = JUST(x->device());
    if (device->type() != "cpu") {
      CHECK_EQ_OR_RETURN(device->device_id(), GlobalProcessCtx::LocalRank())
          << Error::Unimplemented() << "tensor must be on default device of the current rank.";
    }
    Symbol<cfg::ParallelDistribution> parallel_distribution = JUST(GetNdSbp(sbp_parallels));
    std::shared_ptr<const Shape> shape_ptr;
    if (shape.has_value()) {
      shape_ptr = JUST(shape.value());
    } else {
      shape_ptr = JUST(GetConsistentShape(*x->shape(), parallel_desc, parallel_distribution));
    }
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<Shape>("shape", *shape_ptr));
    const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {x}, OpExprInterpContext(attrs, parallel_desc, parallel_distribution)));
    return output;
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ConsistentToLocalFunctor {
 public:
  ConsistentToLocalFunctor() {
    op_ = CHECK_JUST(
        one::CastFromConsistentOpExpr::New(*CHECK_JUST(UniqueStr("cast_to_consistent"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    CHECK_OR_RETURN(x->is_consistent()) << "consistent tensors supported only";
    return JUST(OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::LocalToConsistentFunctor>("LocalToConsistent");
  m.add_functor<impl::ConsistentToLocalFunctor>("ConsistentToLocal");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
