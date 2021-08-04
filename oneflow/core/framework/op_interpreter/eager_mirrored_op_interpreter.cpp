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
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_expr_helper.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/id_util.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Symbol<Device>> GetDefaultDevice(const OpExprInterpContext& ctx) {
  if (ctx.device.has_value()) { return ctx.device.value(); }
  return Device::New("cpu", 0);
}

Maybe<EagerMirroredTensorImpl*> TensorImpl4Tensor(const std::shared_ptr<Tensor>& tensor) {
  CHECK_OR_RETURN(static_cast<bool>(tensor));
  return tensor->mut_eager_mirrored_tensor_impl();
}

}  // namespace

Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                           const Symbol<Device>& default_device, TensorTuple* outputs,
                           const OpExprInterpContext& ctx) {
  const auto& attrs = ctx.attrs;
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    const auto& input_device = JUST(inputs.at(i)->device());
    if (i > 0) {
      CHECK_OR_RETURN(*default_device == *input_device) << Error::InputDeviceNotMatchError();
    }
    input_eager_blob_objects->at(i) = JUST(inputs.at(i)->eager_blob_object());
  }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  for (int i = 0; i < outputs->size(); i++) {
    if (!outputs->at(i)) {
      outputs->at(i) =
          std::make_shared<MirroredTensor>(std::make_shared<EagerMirroredTensorImpl>());
    }
    if (JUST(outputs->at(i)->has_eager_blob_object())) {
      output_eager_blob_objects->at(i) = JUST(outputs->at(i)->eager_blob_object());
    }
  }
  Symbol<Device> op_device;
  std::shared_ptr<const ParallelDesc> op_parallel_desc;
  bool need_check_mem_case = true;
  bool need_event_record = false;

  // Infer devices
  if (!user_op_expr.has_device_infer_fn()) {
    op_device = default_device;
    op_parallel_desc = op_device->parallel_desc_ptr();
    for (int i = 0; i < outputs->size(); i++) {
      auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
      *JUST(tensor_impl->mut_device()) = default_device;
    }
  } else {
    need_check_mem_case = false;
    op_device = JUST(user_op_expr.InferDevices(attrs, inputs, outputs));
    for (const auto& input_tensor : inputs) {
      const auto& input_device = JUST(input_tensor->device());
      need_event_record = need_event_record || !(*op_device == *input_device);
    }
    op_parallel_desc = op_device->parallel_desc_ptr();
  }

  // Infer shapes and dtypes
  const auto& device_tag = JUST(op_device->of_type());
  JUST(user_op_expr.InferLogicalShapeAndDType(
      attrs, device_tag,
      [&](int32_t i) -> const TensorMeta* {
        return CHECK_JUST(TensorImpl4Tensor(inputs.at(i)))->mut_tensor_meta();
      },
      [&](int32_t i) -> TensorMeta* {
        return CHECK_JUST(TensorImpl4Tensor(outputs->at(i)))->mut_tensor_meta();
      }));

  for (int i = 0; i < output_eager_blob_objects->size(); i++) {
    if (!output_eager_blob_objects->at(i)) {
      auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
      JUST(tensor_impl->InitEagerBlobObject(JUST(outputs->at(i)->device())->mem_case()));
      output_eager_blob_objects->at(i) = JUST(tensor_impl->eager_blob_object());
    }
  }

  const auto& kernel = JUST(user_op_expr.MutKernel4Device(*op_device));
  kernel->set_need_check_mem_case(need_check_mem_case);

  for (int64_t index : kernel->output_tuple_indexes4mut2_obns()) {
    output_eager_blob_objects->at(index)->set_is_shape_synced(false);
  }

  const auto& instr_type_name = JUST(op_device->local_call_instruction_name());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    if (need_event_record) {
      for (const auto& input_tensor : inputs) {
        const auto& tensor = JUST(input_tensor->AsMirroredTensor());
        CHECK_OR_RETURN(static_cast<bool>(tensor));
        // Instruction `SoftSyncStream` records event which can be used to synchronize cuda
        // stream
        JUST(builder->SoftSyncStream(JUST(tensor->compute_local_dep_object()), "mut",
                                     JUST(tensor->device())->parallel_desc_ptr()));
      }
    }
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      ctx, op_parallel_desc, instr_type_name);
  }));
  return Maybe<void>::Ok();
}

Maybe<void> RunEmptyOp(TensorTuple* outputs) {
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(0)));
  const auto& shape = tensor_impl->tensor_meta()->shape_ptr();
  const auto& data_type = tensor_impl->dtype();
  const auto& device = tensor_impl->device();
  const auto empty_expr = JUST(op_expr_helper::EmptyOp(*shape, data_type));
  std::shared_ptr<TensorTuple> inputs = std::make_shared<TensorTuple>();
  JUST(NaiveInterpret(*empty_expr, *inputs, device, outputs, OpExprInterpContext(AttrMap{})));
  return Maybe<void>::Ok();
}

static Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const OpExprInterpContext& ctx) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  Symbol<Device> default_device;
  if (inputs.empty()) {
    default_device = JUST(GetDefaultDevice(ctx));
  } else {
    default_device = JUST(inputs.at(0)->device());
  }
  return NaiveInterpret(user_op_expr, inputs, default_device, outputs, ctx);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return NaiveInterpret(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

namespace {

Maybe<one::UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_broadcast", *JUST(UniqueStr("eager_nccl_broadcast")))
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

Maybe<Tensor> GetSyncedTensorIfBroadcast(const std::shared_ptr<Tensor>& tensor,
                                         Symbol<ParallelDesc> parallel_desc,
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

}  // namespace

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToConsistentOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  std::shared_ptr<MirroredTensor> input_mirrored_tensor;
  {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_OR_RETURN(!inputs.at(0)->is_consistent());
    const auto& input_tensor = JUST(inputs.at(0)->detach());
    input_mirrored_tensor = std::dynamic_pointer_cast<MirroredTensor>(input_tensor);
    CHECK_OR_RETURN(input_mirrored_tensor) << Error::ValueError("Tensor Cast Error");
    bool requires_grad = autograd::GradMode::is_enabled() && inputs.at(0)->requires_grad();
    input_mirrored_tensor->set_requires_grad(requires_grad);
    input_mirrored_tensor->set_is_leaf(!requires_grad);
  }
  std::shared_ptr<ConsistentTensor> consistent_tensor;
  {
    CHECK_OR_RETURN(ctx.parallel_desc.has_value());
    CHECK_OR_RETURN(ctx.parallel_distribution.has_value());
    const auto& parallel_distribution = JUST(ctx.parallel_distribution.value());
    const auto& parallel_desc = JUST(ctx.parallel_desc.value());
    const auto& logical_shape = JUST(ctx.attrs.GetAttr<Shape>("shape"));
    ConsistentTensorMeta tensor_meta(std::make_shared<const Shape>(logical_shape),
                                     input_mirrored_tensor->dtype(), parallel_distribution,
                                     parallel_desc);
    Optional<int64_t> parallel_id{};
    const auto& device = JUST(GetDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
    const auto& consistent_tensor_impl = JUST(EagerConsistentTensorImpl::New(
        SymbolOf(tensor_meta), device, parallel_id, input_mirrored_tensor->requires_grad(),
        !input_mirrored_tensor->requires_grad()));
    const auto& rpc_token = JUST(RpcToken::NewMetaRpcToken());
    JUST(consistent_tensor_impl->set_rpc_token(rpc_token));
    consistent_tensor = std::make_shared<ConsistentTensor>(consistent_tensor_impl);
    const auto& ctx = JUST(LaunchTensorMetaConsistencyCheck(*consistent_tensor));
    if (parallel_id.has_value()) {
      const auto& synced_tensor = JUST(
          GetSyncedTensorIfBroadcast(input_mirrored_tensor, parallel_desc, parallel_distribution));
      consistent_tensor_impl->reset_cur_rank_phy_tensor(
          std::dynamic_pointer_cast<MirroredTensor>(synced_tensor));
    }
    JUST(RpcUtil::WaitUntilDoneOrTimeout(*ctx, RpcUtil::TimeoutSeconds()));
    JUST(ctx->Check());
  }
  outputs->at(0) = consistent_tensor;
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromConsistentOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeSplitOrCloneInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeConcatAndAddInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
