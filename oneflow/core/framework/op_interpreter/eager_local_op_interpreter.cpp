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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/local_tensor_infer_cache.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_global_id.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Symbol<Device>> RawGetDefaultCpuDevice() { return Device::New("cpu"); }

constexpr auto* GetDefaultCpuDevice = DECORATE(&RawGetDefaultCpuDevice, ThreadLocal);

Maybe<Symbol<Device>> GetDefaultDevice(const TensorTuple& inputs, const OpExprInterpContext& ctx,
                                       const UserOpExpr& user_op_expr) {
  if (!inputs.empty()) {
    for (int32_t i = 0; i < inputs.size(); ++i) {
      if (!user_op_expr.IsHostMemoryInput(i)) { return JUST(inputs.at(i)->device()); }
    }
  }
  if (ctx.device.has_value()) {
    return JUST(ctx.device);
  } else {
    return GetDefaultCpuDevice();
  }
}

Maybe<EagerLocalTensorImpl*> TensorImpl4Tensor(const std::shared_ptr<Tensor>& tensor) {
  CHECK_OR_RETURN(static_cast<bool>(tensor));
  return tensor->mut_eager_local_tensor_impl();
}

}  // namespace

Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                           TensorTuple* outputs, const OpExprInterpContext& ctx) {
  OF_PROFILER_RANGE_GUARD("NaiveInterpret");
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());  // NOLINT
  Symbol<Device> default_device = JUST(GetDefaultDevice(inputs, ctx, user_op_expr));
  const std::shared_ptr<const LocalTensorInferResult> result =
      JUST([&]() -> Maybe<const LocalTensorInferResult> {
        LocalTensorMetaInferArgs infer_args;
        JUST(infer_args.Init(ctx.attrs, default_device, inputs));
        return JUST(user_op_expr.mut_local_tensor_infer_cache()->GetOrInfer(infer_args));
      }());

  vm::EagerBlobObjectList input_eager_blob_objects(inputs.size());
  // expand lifetime of host_inputs to the end of this function
  TensorTuple host_inputs;
  for (int i = 0; i < inputs.size(); i++) {
    if (user_op_expr.IsHostMemoryInput(i)) {
      const auto& host_input = JUST(functional::To(
          inputs.at(i), Optional<Symbol<Device>>(JUST(GetDefaultCpuDevice())), NullOpt, false));
      input_eager_blob_objects.at(i) = JUST(host_input->eager_blob_object());
      host_inputs.emplace_back(host_input);
    } else {
      input_eager_blob_objects.at(i) = JUST(inputs.at(i)->eager_blob_object());
    }
  }

  const auto& output_tensor_metas = result->output_tensor_metas();
  vm::EagerBlobObjectList output_eager_blob_objects(outputs->size());

  const auto& kernel = JUST(user_op_expr.MutKernel4Stream(result->stream()));

  for (int i = 0; i < outputs->size(); i++) {
    if (!outputs->at(i)) {
      // NOTE: if op support stride(non-contiguous input), then output tensor's stride
      // should be inferred in InferLogicalTensorDesc.
      // otherwise, it will be set here(according to shape).
      std::shared_ptr<MutLocalTensorMeta> mut_tensor_meta;
      {
        if (kernel->output_is_mut2_type(i)) {
          mut_tensor_meta = std::make_shared<MutLocalTensorMeta>(
              output_tensor_metas.at(i)->shape(), output_tensor_metas.at(i)->stride(),
              output_tensor_metas.at(i)->dtype(), output_tensor_metas.at(i)->device());
        }
      }
      std::shared_ptr<EagerLocalTensorImpl> tensor_impl =
          std::make_shared<EagerLocalTensorImpl>(false, false);
      const auto& dep_object = NewLocalDepObject();
      JUST(
          tensor_impl->InitEagerBlobObject(output_tensor_metas.at(i), mut_tensor_meta, dep_object));
      output_eager_blob_objects.at(i) = JUST(tensor_impl->eager_blob_object());
      (*outputs)[i] = std::make_shared<LocalTensor>(tensor_impl);
    } else {
      const auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
      // output i is inplaced.
      // check TensorMeta of infer result and TensorMeta of output i.
      CHECK_OR_RETURN(tensor_impl->tensor_meta()->shape()                                 // NOLINT
                      == output_tensor_metas.at(i)->shape())                              // NOLINT
          << Error::RuntimeError() << tensor_impl->tensor_meta()->shape().ToString()      // NOLINT
          << " .vs "                                                                      // NOLINT
          << output_tensor_metas.at(i)->shape().ToString();                               // NOLINT
      CHECK_OR_RETURN(tensor_impl->tensor_meta()->dtype()                                 // NOLINT
                      == output_tensor_metas.at(i)->dtype())                              // NOLINT
          << Error::RuntimeError() << DataType_Name(tensor_impl->tensor_meta()->dtype())  // NOLINT
          << " .vs "                                                                      // NOLINT
          << DataType_Name(output_tensor_metas.at(i)->dtype());                           // NOLINT
      bool has_eager_blob_object = JUST(outputs->at(i)->has_eager_blob_object());
      CHECK_OR_RETURN(has_eager_blob_object);  // NOLINT
      output_eager_blob_objects.at(i) = JUST(outputs->at(i)->eager_blob_object());
      // TODO(zhaoluyang):(thread_local TensorMeta set stride then check)
      // CHECK_OR_RETURN(tensor_impl->tensor_meta()->stride() ==
      // output_tensor_metas->at(i)->stride());
    }
  }

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->Call(kernel, std::move(input_eager_blob_objects),
                         std::move(output_eager_blob_objects), ctx, result->stream());
  }));
  for (int64_t index : kernel->output_tuple_indexes4mut2_obns()) {
    const auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(index)));
    auto btb = std::make_shared<BlockingThenBusy>();
    JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
      return builder->SyncAccessBlobByCallback(
          tensor_impl, btb, [](ep::Stream* stream, const std::shared_ptr<vm::EagerBlobObject>&) {},
          "const");
    }));
    JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
    const auto& mut_tensor_meta = const_cast<EagerLocalTensorImpl*>(tensor_impl)->mut_tensor_meta();
    Symbol<LocalTensorMeta> new_tensor_meta =
        SymbolOf(LocalTensorMeta(mut_tensor_meta->shape(), mut_tensor_meta->stride(),
                                 mut_tensor_meta->dtype(), mut_tensor_meta->device()));
    std::shared_ptr<EagerLocalTensorImpl> final_tensor_impl =
        std::make_shared<EagerLocalTensorImpl>(JUST(tensor_impl->tensor_storage()),
                                               JUST(tensor_impl->storage_offset()), false, false);
    JUST(final_tensor_impl->InitEagerBlobObject(
        new_tensor_meta,
        JUST(JUST(outputs->at(index)->eager_blob_object())->compute_local_dep_object())));
    JUST(JUST(outputs->at(index)->AsLocalTensor())->set_impl(final_tensor_impl));
  }

  return Maybe<void>::Ok();
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                             TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return NaiveInterpret(op_expr, inputs, outputs, ctx);
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

static Maybe<void> BuildAndRunLocalCastInstruction(const BuiltinOpExpr& op_expr,
                                                   const TensorTuple& inputs,
                                                   TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

namespace {

Maybe<one::UserOpExpr> EagerCclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root,
                                         size_t size, const std::vector<Shape>& shape_list) {
  return one::OpBuilder("eager_ccl_broadcast", *JUST(UniqueStr("eager_ccl_broadcast")))
      .Input("in", size)
      .Output("out", size)
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<std::vector<Shape>>("shape_list", shape_list)
      .Attr<int64_t>("root", root)
      .Build();
}

auto* CachedEagerCclBroadcastOpExpr = DECORATE(&EagerCclBroadcast, ThreadLocalCachedCopiable);

}  // namespace

Maybe<Tensor> Broadcast(const std::shared_ptr<Tensor>& tensor, int64_t src_rank,
                        Symbol<ParallelDesc> parallel_desc, bool inplace) {
  CHECK_OR_RETURN(parallel_desc->containing_current_rank());
  if (parallel_desc->parallel_num() == 1 /* no broadcast */) { return tensor; }
  std::shared_ptr<UserOpExpr> op_expr =
      JUST(CachedEagerCclBroadcastOpExpr(parallel_desc, src_rank, 1, {*tensor->shape()}));
  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("root");
  attrs.SetAllAttrs(src_rank);
  if (inplace) {
    TensorTuple outputs{tensor};
    JUST(OpInterpUtil::Dispatch(*op_expr, {tensor}, &outputs,
                                one::OpExprInterpContext(attrs, parallel_desc)));
    return tensor;
  } else {
    return JUST(OpInterpUtil::Dispatch<one::Tensor>(
        *op_expr, {tensor}, one::OpExprInterpContext(attrs, parallel_desc)));
  }
}

Maybe<TensorTuple> Broadcast(const TensorTuple& inputs, int64_t src_rank,
                             Symbol<ParallelDesc> parallel_desc, bool inplace) {
  CHECK_OR_RETURN(parallel_desc->containing_current_rank())
      << "Current rank are not contained in the placement arguement";
  if (parallel_desc->parallel_num() == 1 /* no broadcast */) { return inputs; }
  std::vector<Shape> shape_list;
  for (const auto& tensor : inputs) { shape_list.emplace_back(*tensor->shape()); }
  std::shared_ptr<UserOpExpr> op_expr =
      JUST(CachedEagerCclBroadcastOpExpr(parallel_desc, src_rank, inputs.size(), shape_list));
  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("root");
  attrs.SetAllAttrs(src_rank);
  if (inplace) {
    auto outputs = std::make_shared<TensorTuple>(inputs);
    JUST(OpInterpUtil::Dispatch(*op_expr, inputs, outputs.get(),
                                one::OpExprInterpContext(attrs, parallel_desc)));
    return outputs;
  } else {
    return JUST(OpInterpUtil::Dispatch<one::TensorTuple>(
        *op_expr, inputs, one::OpExprInterpContext(attrs, parallel_desc)));
  }
}

namespace {

Maybe<Tensor> GetSyncedTensorIfBroadcast(const std::shared_ptr<Tensor>& tensor,
                                         Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp,
                                         bool inplace) {
  Optional<int64_t> parallel_id;
  JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  if (!parallel_id.has_value()) { return tensor; }
  const auto& broadcast_parallel_desc = JUST(GetBroadcastSubParallelDesc(parallel_desc, nd_sbp));
  int64_t root = JUST(broadcast_parallel_desc->MachineId4ParallelId(0));
  if (broadcast_parallel_desc->parallel_num() > 1 && inplace && GlobalProcessCtx::Rank() == 0) {
    LOG_FIRST_N(WARNING, 1)
        << "Casting a local tensor to a global tensor with Broadcast sbp will modify the data of "
           "input! "
           "If you want to keep the input local tensor unchanged, please set the arg copy to True.";
  }
  return Broadcast(tensor, root, broadcast_parallel_desc, inplace);
}

Maybe<Shape> CalcPhysicalShape(Symbol<GlobalTensorMeta> global_tensor_meta) {
  const auto& opt_parallel_id =
      JUST(GetParallelId4CurrentProcessCtx(global_tensor_meta->parallel_desc()));
  int64_t parallel_id = JUST(*opt_parallel_id);
  return GetPhysicalShape(global_tensor_meta->shape(), *global_tensor_meta->nd_sbp(),
                          *global_tensor_meta->parallel_desc(), parallel_id);
}

static constexpr auto* GetPhysicalShape = DECORATE(&CalcPhysicalShape, ThreadLocal);

Maybe<Tensor> TryReshapeTensor(const std::shared_ptr<Tensor>& tensor,
                               Symbol<GlobalTensorMeta> global_tensor_meta) {
  CHECK_OR_RETURN(tensor->is_local());
  const auto& physical_shape = JUST(GetPhysicalShape(global_tensor_meta));
  if (*physical_shape == *tensor->shape()) { return tensor; }
  CHECK_EQ_OR_RETURN(physical_shape->elem_cnt(), tensor->shape()->elem_cnt());
  // TODO(lixinqi) inplace reshape.
  return tensor;
}

}  // namespace

Maybe<void> EagerLocalInterpreter::ApplyImpl(const GlobalToGlobalOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

namespace {

Maybe<void> RawLocalToGlobal(const LocalToGlobalOpExpr& op_expr, const TensorTuple& inputs,
                             TensorTuple* outputs, const OpExprInterpContext& ctx) {
  std::shared_ptr<LocalTensor> input_local_tensor;
  {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_OR_RETURN(!inputs[0]->is_global());  // NOLINT
    const auto& input_tensor = JUST(inputs.at(0)->detach());
    input_local_tensor = JUST(input_tensor->AsLocalTensor());
    CHECK_OR_RETURN(input_local_tensor)
        << Error::InvalidValueError() << "Tensor Cast Error";  // NOLINT
    bool requires_grad = autograd::GradMode::is_enabled() && inputs.at(0)->requires_grad();
    JUST(input_local_tensor->set_requires_grad(requires_grad));
    input_local_tensor->set_is_leaf(!requires_grad);
  }
  std::shared_ptr<GlobalTensor> global_tensor;
  {
    CHECK_OR_RETURN(ctx.parallel_desc.has_value());
    CHECK_OR_RETURN(ctx.nd_sbp.has_value());
    const auto& nd_sbp = JUST(ctx.nd_sbp);
    const auto& parallel_desc = JUST(ctx.parallel_desc);
    const auto& logical_shape = JUST(ctx.attrs.GetAttr<Shape>("shape"));
    DataType dtype = JUST(ctx.attrs.GetAttr<DataType>("dtype"));
    GlobalTensorMeta tensor_meta(logical_shape, dtype, nd_sbp, parallel_desc);
    Optional<int64_t> parallel_id{};
    const auto& device = JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
    const auto& global_tensor_impl = JUST(EagerGlobalTensorImpl::New(
        SymbolOf(tensor_meta), device, parallel_id, input_local_tensor->requires_grad(),
        !input_local_tensor->requires_grad()));
    global_tensor = std::make_shared<GlobalTensor>(global_tensor_impl);
    if (parallel_id.has_value()) {
      const auto& pyhsical_shape = JUST(GetPhysicalShape(tensor_meta));
      const auto& input_local_tensor_shape = input_local_tensor->shape();
      CHECK_EQ_OR_RETURN(*pyhsical_shape, *input_local_tensor_shape);      // NOLINT
      CHECK_OR_RETURN(dtype == input_local_tensor->dtype()->data_type());  // NOLINT
      global_tensor_impl->reset_cur_rank_phy_tensor(input_local_tensor);
    }
  }
  (*outputs)[0] = global_tensor;
  return Maybe<void>::Ok();
}

static constexpr auto* LocalToGlobal = DECORATE(&RawLocalToGlobal, NonRecursiveInitGlobalId);

}  // namespace

Maybe<void> EagerLocalInterpreter::ApplyImpl(const LocalToGlobalOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  bool sync_data = JUST(ctx.attrs.GetAttr<bool>("sync_data"));
  JUST(LocalToGlobal(op_expr, inputs, outputs, ctx));
  const auto& global_tensor = JUST((*outputs)[0]->AsGlobalTensor());
  JUST(WithConsistencyChecked(global_tensor, [&]() -> Maybe<void> {
    if (IsGlobalTensorMetaCheckDisabled()) { return Maybe<void>::Ok(); }
    const auto& parallel_desc = JUST(ctx.parallel_desc);
    const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
    if (!parallel_id->has_value()) { return Maybe<void>::Ok(); }
    const auto& nd_sbp = JUST(ctx.nd_sbp);
    const auto& tensor_meta = JUST(global_tensor->global_tensor_meta());
    const auto& local_tensor = JUST(global_tensor->cur_rank_phy_tensor());
    const auto& reshaped_tensor = JUST(TryReshapeTensor(local_tensor, tensor_meta));
    std::shared_ptr<Tensor> synced_tensor = reshaped_tensor;
    if (sync_data) {
      bool inplace = JUST(ctx.attrs.GetAttr<bool>("inplace_when_sync_data"));
      synced_tensor =
          JUST(GetSyncedTensorIfBroadcast(reshaped_tensor, parallel_desc, nd_sbp, inplace));
    }
    auto* global_tensor_impl = reinterpret_cast<EagerGlobalTensorImpl*>(global_tensor->mut_impl());
    CHECK_NOTNULL_OR_RETURN(global_tensor_impl);
    global_tensor_impl->reset_cur_rank_phy_tensor(JUST(synced_tensor->AsLocalTensor()));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const GlobalToLocalOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const CastToLocalOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return BuildAndRunLocalCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const CastFromLocalOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return BuildAndRunLocalCastInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeSplitOrCloneInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
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

Maybe<void> EagerLocalInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const SelectTopNOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  int top_n = JUST(ctx.attrs.GetAttr<int32_t>("top_n"));
  outputs->resize(top_n);
  for (int i = 0; i < top_n; ++i) { (*outputs)[i] = JUST(JUST(VectorAt(inputs, i))->detach()); }
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
