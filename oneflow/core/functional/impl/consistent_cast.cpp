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
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

Maybe<one::UserOpExpr> FindOrCreatEagerNcclBroadcastOpExpr(Symbol<ParallelDesc> parallel_desc) {
  thread_local HashMap<Symbol<ParallelDesc>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc2eager_nccl_broadcast;
  auto iter = parallel_desc2eager_nccl_broadcast.find(parallel_desc);
  if (iter == parallel_desc2eager_nccl_broadcast.end()) {
    int64_t root = JUST(parallel_desc->DeviceId4ParallelId(0));
    std::shared_ptr<UserOpExpr> op_expr =
        JUST(op_expr_helper::EagerNcclBroadcast(parallel_desc, root));
    iter = parallel_desc2eager_nccl_broadcast.emplace(parallel_desc, op_expr).first;
  }
  return iter->second;
}

Maybe<Tensor> SyncDataAndMetaInfo(const std::shared_ptr<Tensor>& tensor,
                                  const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                                  Symbol<ParallelDesc> parallel_desc) {
  // TODO(hanbinbin): Sync data when sync_consistent_meta_info branch merged in master
  if (sbp_parallels.size() == 1) {
    const auto& sbp_parallel = sbp_parallels.at(0);
    if (sbp_parallel->has_split_parallel()) {
      return tensor;
    } else if (sbp_parallel->has_broadcast_parallel()) {
      if (parallel_desc->device_tag() == "gpu") {
        std::shared_ptr<UserOpExpr> op_expr =
            JUST(FindOrCreatEagerNcclBroadcastOpExpr(parallel_desc));
        return JUST(OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {tensor}));
      } else {
        OF_UNIMPLEMENTED();
      }
    } else if (sbp_parallel->has_partial_sum_parallel()) {
      if (GlobalProcessCtx::Rank() == 0) {
        const auto& out_tensor = JUST(tensor->detach());
        bool requires_grad = autograd::GradMode::is_enabled() && tensor->requires_grad();
        out_tensor->set_requires_grad(requires_grad);
        out_tensor->set_is_leaf(!requires_grad);
        return out_tensor;
      } else {
        return functional::ZerosLike(tensor);
      }
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    OF_UNIMPLEMENTED();
  }
}

}  //  namespace

class ToConsistentFunctor {
 public:
  ToConsistentFunctor() { op_ = CHECK_JUST(op_expr_helper::CastToConsistentOp()); }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                           Symbol<ParallelDesc> parallel_desc) const {
    cfg::ParallelDistribution parallel_distribution;
    for (Symbol<cfg::SbpParallel> sbp_symbol : sbp_parallels) {
      *(parallel_distribution.mutable_sbp_parallel()->Add()) = *sbp_symbol;
    }
    if (x->is_consistent()) {
      UNIMPLEMENTED();
    } else {
      const auto& mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(x);
      CHECK_NOTNULL_OR_RETURN(mirrored_tensor) << "local tensors supported only";
      CHECK_OR_RETURN(mirrored_tensor->is_eager()) << "eager tensors supported only";
      if (mirrored_tensor->is_cuda()) {
        CHECK_EQ_OR_RETURN(
            JUST(mirrored_tensor->device())->device_id(),
            GlobalProcessCtx::LocalRank() % (Global<ResourceDesc, ForEnv>::Get()->GpuDeviceNum()))
            << "tensor must be on default device of rank!";
      }
      std::shared_ptr<Tensor> synced_tensor =
          JUST(SyncDataAndMetaInfo(mirrored_tensor, sbp_parallels, parallel_desc));
      const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(
          *op_, {synced_tensor},
          OpExprInterpContext(AttrMap{}, parallel_desc, SymbolOf(parallel_distribution))));
      return output;
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ToLocalFunctor {
 public:
  ToLocalFunctor() { op_ = CHECK_JUST(op_expr_helper::CastFromConsistentOp()); }

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
  m.add_functor<impl::ToConsistentFunctor>("ToConsistent");
  m.add_functor<impl::ToLocalFunctor>("ToLocal");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow