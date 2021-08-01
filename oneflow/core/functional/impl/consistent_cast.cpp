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
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/autograd/autograd_engine.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

Maybe<one::UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_broadcast", *CHECK_JUST(UniqueStr("eager_nccl_broadcast")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

Maybe<one::UserOpExpr> FindOrCreatEagerNcclBroadcastOpExpr(Symbol<ParallelDesc> parallel_desc) {
  static thread_local HashMap<Symbol<ParallelDesc>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc2eager_nccl_broadcast;
  auto iter = parallel_desc2eager_nccl_broadcast.find(parallel_desc);
  if (iter == parallel_desc2eager_nccl_broadcast.end()) {
    int64_t root = JUST(parallel_desc->DeviceId4ParallelId(0));
    std::shared_ptr<UserOpExpr> op_expr = JUST(EagerNcclBroadcast(parallel_desc, root));
    iter = parallel_desc2eager_nccl_broadcast.emplace(parallel_desc, op_expr).first;
  }
  return iter->second;
}

Maybe<Tensor> GetSyncedTensorIfBroadcast(
    const std::shared_ptr<Tensor>& tensor, Symbol<ParallelDesc> parallel_desc,
    Symbol<cfg::ParallelDistribution> parallel_distribution) {
  Optional<int64_t> parallel_id;
  JUST(GetDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  if (!parallel_id.has_value()) { return tensor; }
  const auto& broadcast_parallel_desc =
      JUST(GetBroadcastSubParallelDesc(parallel_desc, parallel_distribution));
  if (broadcast_parallel_desc->parallel_num() == 1 /* no broadcast */) { return tensor; }
  CHECK_EQ_OR_RETURN(broadcast_parallel_desc->device_tag(), "gpu")
      << Error::Todo() << "supported cuda only now.";
  std::shared_ptr<UserOpExpr> op_expr =
      JUST(FindOrCreatEagerNcclBroadcastOpExpr(broadcast_parallel_desc));
  return JUST(OpInterpUtil::Dispatch<one::Tensor>(
      *op_expr, {tensor}, one::OpExprInterpContext(AttrMap{}, broadcast_parallel_desc)));
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
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels) const {
    CHECK_OR_RETURN(x->is_local()) << Error::Unimplemented() << "local tensors supported only";
    CHECK_OR_RETURN(x->is_eager()) << Error::Unimplemented() << "eager tensors supported only";
    const auto& device = JUST(x->device());
    if (device->type() != "cpu") {
      CHECK_EQ_OR_RETURN(device->device_id(), GlobalProcessCtx::LocalRank())
          << Error::Unimplemented() << "tensor must be on default device of the current rank.";
    }
    Symbol<cfg::ParallelDistribution> parallel_distribution = JUST(GetNdSbp(sbp_parallels));
    std::shared_ptr<Tensor> synced_tensor =
        JUST(GetSyncedTensorIfBroadcast(x, parallel_desc, parallel_distribution));
    const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(
        *op_, {synced_tensor},
        OpExprInterpContext(AttrMap{}, parallel_desc, parallel_distribution)));
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
    const auto& consistent_tensor = std::dynamic_pointer_cast<ConsistentTensor>(x);
    CHECK_NOTNULL_OR_RETURN(consistent_tensor) << "consistent tensors supported only";
    CHECK_OR_RETURN(consistent_tensor->is_eager()) << "eager tensors supported only";
    int64_t machine_id = 0;
    int64_t device_id = 0;
    const auto& parallel_desc = JUST(consistent_tensor->parallel_desc());
    GlobalProcessCtx::GetCurrentMachineIdAndDeviceId(&machine_id, &device_id);
    if (!parallel_desc->Containing(machine_id, device_id)) {
      // should return UndefinesdLocalTensor here, the impl of which need to be discussed
      return std::shared_ptr<Tensor>();
    }
    const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(*op_, {consistent_tensor}));
    return output;
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
