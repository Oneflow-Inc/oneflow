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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr_grad_function.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace one {

namespace {

Maybe<one::UserOpExpr> EagerNcclReduce(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_reduce", *JUST(UniqueStr("eager_nccl_reduce")))
      .Input("in")
      .Output("out")
      .Build();
}

Maybe<one::UserOpExpr> FindOrCreatEagerNcclReduceOpExpr(Symbol<ParallelDesc> parallel_desc,
                                                        int64_t root) {
  thread_local HashMap<std::pair<Symbol<ParallelDesc>, int64_t>, std::shared_ptr<one::UserOpExpr>>
      parallel_desc_and_root_device2eager_nccl_reduce;
  const auto& key = std::make_pair(parallel_desc, root);
  auto iter = parallel_desc_and_root_device2eager_nccl_reduce.find(key);
  if (iter == parallel_desc_and_root_device2eager_nccl_reduce.end()) {
    std::shared_ptr<UserOpExpr> op_expr = JUST(EagerNcclReduce(parallel_desc, root));
    iter = parallel_desc_and_root_device2eager_nccl_reduce.emplace(key, op_expr).first;
  }
  return iter->second;
}

}  // namespace

struct EagerNcclBroadcastCaptureState : public AutoGradCaptureState {
  Symbol<ParallelDesc> parallel_desc;
  int64_t root;
};

class EagerNcclBroadcast : public OpExprGradFunction<EagerNcclBroadcastCaptureState> {
 public:
  Maybe<void> Capture(EagerNcclBroadcastCaptureState* state, const TensorTuple& inputs,
                      const TensorTuple& outputs,
                      const OpExprInterpContext& interp_ctx) const override {
    auto* op_ctx = dynamic_cast<const EagerNcclBroadcastOp*>(interp_ctx.op_ctx.get());
    state->root = op_ctx->root();
    state->parallel_desc = JUST(interp_ctx.parallel_desc);
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(const EagerNcclBroadcastCaptureState* state, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    const auto& grad_op = JUST(FindOrCreatEagerNcclReduceOpExpr(state->parallel_desc, state->root));
    auto op_ctx = std::make_shared<schema::EagerNcclReduceOp>();
    op_ctx->set_parallel_conf(PbMessage2TxtString(state->parallel_desc->parallel_conf()));
    op_ctx->set_root(state->root);
    in_grads->resize(1);
    in_grads->at(0) = JUST(
        OpInterpUtil::Dispatch<Tensor>(*grad_op, {out_grads.at(0)}, OpExprInterpContext(op_ctx)));
    return Maybe<void>::Ok();
  }
};

REGISTER_OP_EXPR_GRAD_FUNCTION("eager_nccl_broadcast", EagerNcclBroadcast);

}  // namespace one
}  // namespace oneflow
