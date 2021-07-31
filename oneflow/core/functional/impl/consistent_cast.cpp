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
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/sbp_parallel.h"

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

Maybe<one::UserOpExpr> FindOrCreatEagerNcclBroadcastOpExpr(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  thread_local HashMap<std::pair<Symbol<ParallelDesc>, int64_t>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc2eager_nccl_broadcast;
  const auto& key = std::make_pair(parallel_desc, root);
  auto iter = parallel_desc2eager_nccl_broadcast.find(key);
  if (iter == parallel_desc2eager_nccl_broadcast.end()) {
    std::shared_ptr<UserOpExpr> op_expr = JUST(EagerNcclBroadcast(parallel_desc, root));
    iter = parallel_desc2eager_nccl_broadcast.emplace(key, op_expr).first;
  }
  return iter->second;
}

Maybe<Tensor> SyncMetaAndData(const std::shared_ptr<Tensor>& tensor, Symbol<ParallelDesc> parallel_desc, Symbol<cfg::ParallelDistribution> parallel_distribution) {
  // TODO(hanbinbin): Sync meta info when sync_consistent_meta_info branch merged in master
  if (parallel_distribution->sbp_parallel_size() == 1) {
    const auto& sbp_parallel = parallel_distribution->sbp_parallel(0);
    if (sbp_parallel.has_split_parallel()) {
      return tensor;
    } else if (sbp_parallel.has_broadcast_parallel()) {
      if (parallel_desc->device_tag() == "gpu") {
        MutableAttrMap mutable_attr_map;
        int64_t root = JUST(parallel_desc->DeviceId4ParallelId(0));
        mutable_attr_map.SetAttr("root", root);
        std::shared_ptr<UserOpExpr> op_expr =
            JUST(FindOrCreatEagerNcclBroadcastOpExpr(parallel_desc, root));
        return JUST(OpInterpUtil::Dispatch<one::Tensor>(
            *op_expr, {tensor},
            one::OpExprInterpContext(mutable_attr_map, parallel_desc,
                                     parallel_distribution)));
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else if (sbp_parallel.has_partial_sum_parallel()) {
      if (GlobalProcessCtx::Rank() == 0) {
        return tensor;
      } else {
        // use SliceUpdate because ZerosLike does not have backward
        int64_t num_axes = tensor->shape()->NumAxes();
        std::vector<int64_t> start(num_axes, 0);
        std::vector<int64_t> stop(num_axes);
        std::vector<int64_t> step(num_axes, 1);
        {
          for (int64_t i = 0; i < num_axes; ++i) { stop.at(i) = tensor->shape()->At(i); }
        }
        return functional::SliceUpdate(tensor, JUST(functional::ZerosLike(tensor)), start, stop,
                                       step);
      }
    }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

// used in consistent_tensor.to_consistent(sbp)
Maybe<Tensor> CastParallelDistribution(const std::shared_ptr<Tensor>& tensor,
                                       const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels,
                                       Symbol<ParallelDesc> parallel_desc) {
  const auto& consistent_tensor = std::dynamic_pointer_cast<ConsistentTensor>(tensor);
  CHECK_NOTNULL_OR_RETURN(consistent_tensor) << "local tensors supported only";
  CHECK_OR_RETURN(consistent_tensor->is_eager()) << "eager tensors supported only";
  std::vector<std::string> sbp_parallel_str_list(sbp_parallels.size());
  {
    for (int64_t i = 0; i < sbp_parallel_str_list.size(); ++i) {
      sbp_parallel_str_list.at(i) = SbpParallelToString(*sbp_parallels.at(i));
    }
  }
  const auto& parallel_distribution_cast_op_expr =
      JUST(OpBuilder("hierarchical_parallel_cast", *JUST(UniqueStr("hierarchical_parallel_cast")))
               .Input("in")
               .Output("out")
               .Attr<std::vector<std::string>>("parallel_distribution", sbp_parallel_str_list)
               .Attr<std::string>("grad_mode", "restore")
               .Attr<std::vector<std::string>>("grad_parallel_distribution", sbp_parallel_str_list)
               .Build());
  return JUST(OpInterpUtil::Dispatch<one::Tensor>(*parallel_distribution_cast_op_expr, {consistent_tensor}));
}

}  //  namespace

class ToConsistentFunctor {
 public:
  ToConsistentFunctor() {
    op_ =
        CHECK_JUST(one::CastToConsistentOpExpr::New(*CHECK_JUST(UniqueStr("cast_to_consistent"))));
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x,
                           Symbol<ParallelDesc> parallel_desc,
                           const std::vector<Symbol<cfg::SbpParallel>>& sbp_parallels) const {
    if (x->is_consistent()) {
      return JUST(CastParallelDistribution(x, sbp_parallels, parallel_desc));
    } else {
      const auto& mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(x);
      CHECK_NOTNULL_OR_RETURN(mirrored_tensor) << "local tensors supported only";
      CHECK_OR_RETURN(mirrored_tensor->is_eager()) << "eager tensors supported only";
      if (mirrored_tensor->is_cuda()) {
        CHECK_EQ_OR_RETURN(JUST(mirrored_tensor->device())->device_id(),
                           GlobalProcessCtx::LocalRank())
            << "tensor must be on default device of rank!";
      }
      Symbol<cfg::ParallelDistribution> parallel_distribution = JUST(GetNdSbp(sbp_parallels));
      std::shared_ptr<Tensor> synced_tensor =
          JUST(SyncMetaAndData(mirrored_tensor, parallel_desc, parallel_distribution));
      const auto& output = JUST(OpInterpUtil::Dispatch<one::Tensor>(
          *op_, {synced_tensor},
          OpExprInterpContext(AttrMap{}, parallel_desc, parallel_distribution)));
      return output;
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class ToLocalFunctor {
 public:
  ToLocalFunctor() {
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
  m.add_functor<impl::ToConsistentFunctor>("ToConsistent");
  m.add_functor<impl::ToLocalFunctor>("ToLocal");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
