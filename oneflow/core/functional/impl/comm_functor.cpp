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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/eager_mirrored_op_interpreter.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

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

Maybe<one::UserOpExpr> EagerNcclAllReduce(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_reduce", *JUST(UniqueStr("eager_nccl_all_reduce")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

static constexpr auto* CachedEagerNcclAllReduceOpExpr = DECORATE(&EagerNcclAllReduce, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclReduceScatter(Symbol<ParallelDesc> parallel_desc,
                                              const std::string& op_type) {
  return one::OpBuilder("eager_nccl_reduce_scatter", *JUST(UniqueStr("eager_nccl_reduce_scatter")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<std::string>("op_type", op_type)
      .Build();
}
static constexpr auto* CachedNcclReduceScatterOpExpr =
    DECORATE(&EagerNcclReduceScatter, ThreadLocalCopiable);

Maybe<one::UserOpExpr> EagerNcclAllGather(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_gather", *JUST(UniqueStr("eager_nccl_all_gather")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

static constexpr auto* CachedEagerNcclAllGatherOpExpr = DECORATE(&EagerNcclAllGather, ThreadLocal);

}  // namespace

class BroadcastFunctor {
 public:
  BroadcastFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, bool inplace) const {
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    std::string device_type_str = JUST(x->device())->type();
    CHECK_OR_RETURN(device_type_str == "cuda" || device_type_str == "cpu");
    DeviceType device_type = device_type_str == "cuda" ? DeviceType::kGPU : DeviceType::kCPU;
    const auto& parallel_desc = JUST(RankGroup::GetDefaultParallelDesc(device_type, rank_group));
    return one::Broadcast(x, parallel_desc, inplace);
  }
};

class LocalAllReduceFunctor {
 public:
  LocalAllReduceFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    {
      const auto& device = JUST(x->device());
      CHECK_EQ_OR_RETURN(JUST(device->of_type()), "gpu");
      CHECK_EQ_OR_RETURN(device->device_id(), GlobalProcessCtx::LocalRank());
    }
    static thread_local std::unordered_map<Symbol<RankGroup>, std::shared_ptr<OpExpr>>
        rank_group2op_expr;
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    auto iter = rank_group2op_expr.find(rank_group);
    std::shared_ptr<OpExpr> op_expr;
    if (iter == rank_group2op_expr.end()) {
      ParallelConf parallel_conf;
      parallel_conf.set_device_tag("gpu");
      JUST(rank_group->ForEachRank([&parallel_conf](int64_t rank) -> Maybe<void> {
        parallel_conf.add_device_name("@" + std::to_string(rank) + ":"
                                      + std::to_string(GlobalProcessCtx::LocalRank(rank)));
        return Maybe<void>::Ok();
      }));

      op_expr = JUST(one::OpBuilder("eager_nccl_all_reduce")
                         .Input("in")
                         .Output("out")
                         .Attr("parallel_conf", PbMessage2TxtString(parallel_conf))
                         .Attr<bool>("async_launch", true)
                         .Build());
      rank_group2op_expr[rank_group] = op_expr;
    } else {
      op_expr = iter->second;
    }
    if (const auto& static_zeros_tensor = std::dynamic_pointer_cast<StaticZerosTensor>(x)) {
      return OpInterpUtil::Dispatch<Tensor>(*op_expr,
                                            {JUST(static_zeros_tensor->AsMirroredTensor())}, {});
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}, {});
    }
  }
};

class ConsistentAllReduceFunctor {
 public:
  ConsistentAllReduceFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    {
      CHECK_OR_RETURN(x->is_consistent());
      CHECK_OR_RETURN(IsAllPartialSumNdSbp(JUST(x->nd_sbp())));
      CHECK_EQ_OR_RETURN(JUST(x->parallel_desc())->device_type(), DeviceType::kGPU);
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedEagerNcclAllReduceOpExpr(JUST(x->parallel_desc())));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class ConsistentReduceScatterFunctor {
 public:
  ConsistentReduceScatterFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::string& op_type) const {
    {
      CHECK_OR_RETURN(x->is_consistent());
      if (op_type == "max") {
        CHECK_OR_RETURN(IsAllBroadcastNdSbp(JUST(x->nd_sbp())));
      } else if (op_type == "sum") {
        CHECK_OR_RETURN(IsAllPartialSumNdSbp(JUST(x->nd_sbp())));
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
      CHECK_EQ_OR_RETURN(JUST(x->parallel_desc())->device_type(), DeviceType::kGPU);
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedNcclReduceScatterOpExpr(JUST(x->parallel_desc()), op_type));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};

class ConsistentAllGatherFunctor {
 public:
  ConsistentAllGatherFunctor() = default;
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x) const {
    {
      CHECK_OR_RETURN(x->is_consistent());
      CHECK_OR_RETURN(IsAllSplitNdSbp(JUST(x->nd_sbp()), 0));
      CHECK_EQ_OR_RETURN(JUST(x->parallel_desc())->device_type(), DeviceType::kGPU);
    }
    std::shared_ptr<OpExpr> op_expr =
        JUST(CachedEagerNcclAllGatherOpExpr(JUST(x->parallel_desc())));
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_expr, {x}));
  }
};
}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::BroadcastFunctor>("Broadcast");
  m.add_functor<impl::LocalAllReduceFunctor>("LocalAllReduce");
  m.add_functor<impl::ConsistentAllReduceFunctor>("ConsistentAllReduce");
  m.add_functor<impl::ConsistentReduceScatterFunctor>("ConsistentReduceScatter");
  m.add_functor<impl::ConsistentAllGatherFunctor>("ConsistentAllGather");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
