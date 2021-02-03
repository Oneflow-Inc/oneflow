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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

namespace oneflow {
namespace one {

std::vector<TensorRef> LazyOpInterpreter::Interpret(const OpExpr& op,
                                                    const std::vector<TensorRef>& inputs,
                                                    OpInterpreterContext* ctx) {
  //   OperatorConf op_conf;
  //   *(op_conf.mutable_user_conf()) = op.proto;
  //   *(op_conf.mutable_name()) = op.op_name;
  //   int64_t symbol_id = ctx->scope->symbol_id().GetOrThrow();
  //   op_conf.set_scope_symbol_id(symbol_id);
  //   if (!op_conf.has_device_tag()) {
  //     op_conf.set_device_tag(ctx->scope->device_parallel_desc_symbol()->device_tag());
  //   }
  //   auto op_attribute = [&]() {
  //     auto infer_ctx = GetCurInferCtx().GetOrThrow();
  //     if (ctx->is_mirrored_strategy_enabled) {
  //       return infer_ctx->AddAndInferMirroredOp(op_conf).GetOrThrow();
  //     } else {
  //       return infer_ctx->AddAndInferConsistentOp(op_conf).GetOrThrow();
  //     }
  //   }();
  //   std::vector<TensorRef> return_tensors;
  //   for (const auto& it : op.proto.output()) {
  //     for (const auto& output : it.second.s()) {
  //       // TODO(): Infer tensor shape, data type and other information.
  //       return_tensors.emplace_back(new Tensor);
  //       TensorNameScope::Global()->Record(return_tensors.back(), output);
  //     }
  //   }
  //   return return_tensors;
  return std::vector<TensorRef>{};
}

std::vector<TensorRef> EagerOpInterpreter::Interpret(const OpExpr& op,
                                                     const std::vector<TensorRef>& inputs,
                                                     OpInterpreterContext* ctx) {
  //   OperatorConf op_conf;
  //   *(op_conf.mutable_name()) = op.op_name;
  //   *(op_conf.mutable_user_conf()) = op.proto;
  //   int64_t symbol_id = ctx->scope->symbol_id().GetOrThrow();
  //   op_conf.set_scope_symbol_id(symbol_id);
  //   if (!op_conf.has_device_tag()) {
  //     op_conf.set_device_tag(ctx->scope->device_parallel_desc_symbol()->device_tag());
  //   }
  //   auto op_attribute = [&]() {
  //     auto infer_ctx = GetCurInferCtx().GetOrThrow();
  //     if (ctx->is_mirrored_strategy_enabled) {
  //       return infer_ctx->AddAndInferMirroredOp(op_conf).GetOrThrow();
  //     } else {
  //       return infer_ctx->AddAndInferConsistentOp(op_conf).GetOrThrow();
  //     }
  //   }();
  //
  //   const auto& parallel_conf = ctx->scope->device_parallel_desc_symbol()->parallel_conf();
  //   if (op_attribute.op_conf().has_cast_to_mirrored_conf() ||
  //       op_attribute.op_conf().has_cast_from_mirrored_conf()) {
  //     // return MirroredCast(op_attribute);
  //   }
  //   if (op_attribute.op_conf().has_distribute_split_conf() ||
  //       op_attribute.op_conf().has_distribute_clone_conf()) {
  //     // return DistributeSplitOrClone(op_attribute, parallel_conf);
  //   }
  //   if (op_attribute.op_conf().has_distribute_concat_conf() ||
  //       op_attribute.op_conf().has_distribute_add_conf()) {
  //     // return DistributeConcatOrAdd(op_attribute, parallel_conf);
  //   }
  //   // return NaiveInterpret(op_attribute, parallel_conf);
  return std::vector<TensorRef>{};
}

}  // namespace one
}  // namespace oneflow
