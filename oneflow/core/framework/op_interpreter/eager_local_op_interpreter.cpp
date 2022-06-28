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
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/tensor_consistent_id.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {
namespace one {

namespace {

Maybe<Symbol<Device>> GetDefaultDevice(const OpExprInterpContext& ctx) {
  if (ctx.device.has_value()) { return JUST(ctx.device); }
  return Device::New("cpu", 0);
}

Maybe<EagerLocalTensorImpl*> TensorImpl4Tensor(const std::shared_ptr<Tensor>& tensor) {
  CHECK_OR_RETURN(static_cast<bool>(tensor));
  return tensor->mut_eager_local_tensor_impl();
}

class MutLocalTensorMeta : public TensorMeta {  // NOLINT
 public:
  MutLocalTensorMeta()
      : TensorMeta(std::make_shared<const Shape>(), std::make_shared<const Stride>(),
                   kInvalidDataType) {}
  MutLocalTensorMeta(const MutLocalTensorMeta&) = default;
  MutLocalTensorMeta(MutLocalTensorMeta&&) = default;
  ~MutLocalTensorMeta() override = default;
};

std::vector<TensorMeta*>* ThreadLocalDefaultOutputMutTensorMetas(int64_t size) {
  static thread_local std::vector<MutLocalTensorMeta> struct_vec;
  static thread_local std::vector<TensorMeta*> ptr_vec;
  struct_vec.resize(size);
  ptr_vec.resize(size);
  if (size == 1) {
    ptr_vec.at(0) = &struct_vec.at(0);  // unfold loop
  } else if (size == 2) {
    ptr_vec.at(0) = &struct_vec.at(0);  // unfold loop
    ptr_vec.at(1) = &struct_vec.at(1);  // unfold loop
  } else {
    for (int i = 0; i < size; ++i) { ptr_vec.at(i) = &struct_vec.at(i); }
  }
  return &ptr_vec;
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
      CHECK_OR_RETURN(*default_device == *input_device)
          << Error::RuntimeError()
          << "Expected all tensors to be on the same device, but found at least two devices, "
          << default_device->ToString() << " (positional 0) and " << input_device->ToString()
          << " (positional " << i << ")!";
    }
    input_eager_blob_objects->at(i) = JUST(inputs.at(i)->eager_blob_object());
  }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  auto* output_tensor_metas = ThreadLocalDefaultOutputMutTensorMetas(outputs->size());
  for (int i = 0; i < outputs->size(); i++) {
    if (!outputs->at(i)) {
      const auto& tensor_impl = std::make_shared<EagerLocalTensorImpl>();
      (*outputs)[i] = std::make_shared<LocalTensor>(tensor_impl);
      output_tensor_metas->at(i) = tensor_impl->mut_tensor_meta();
    } else {
      bool has_eager_blob_object = JUST(outputs->at(i)->has_eager_blob_object());
      CHECK_OR_RETURN(has_eager_blob_object);
      output_eager_blob_objects->at(i) = JUST(outputs->at(i)->eager_blob_object());
    }
  }
  Symbol<Stream> stream;
  bool need_check_mem_case = true;

  // Infer devices
  if (!user_op_expr.has_device_and_stream_infer_fn()) {
    stream = JUST(GetDefaultStreamByDevice(default_device));
    for (int i = 0; i < outputs->size(); i++) {
      auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
      *JUST(tensor_impl->mut_device()) = default_device;
    }
  } else {
    need_check_mem_case = false;
    stream = JUST(user_op_expr.InferDeviceAndStream(attrs, inputs, outputs));
  }

  // Infer shapes and dtypes
  const auto& device_tag = stream->device()->type();
  JUST(user_op_expr.InferPhysicalTensorDesc(
      attrs, device_tag,
      [&](int32_t i) -> const TensorMeta* {
        return CHECK_JUST(TensorImpl4Tensor(inputs[i]))->mut_tensor_meta();
      },
      [&](int32_t i) -> TensorMeta* {
        // using thread_local TensorMeta pointer if inplace.
        // using tensor_impl TensorMeta pointer if not inplace.
        return output_tensor_metas->at(i);
      }));

  for (int i = 0; i < output_eager_blob_objects->size(); i++) {
    auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
    if (!output_eager_blob_objects->at(i)) {
      // NOTE: if op support stride(non-contiguous input), then output tensor's stride
      // should be inferred in InferLogicalTensorDesc.
      // otherwise, it will be set here(according to shape).
      if (!JUST(user_op_expr.SupportNonContiguous())) {
        std::shared_ptr<Stride> stride(new Stride(*tensor_impl->shape()));
        tensor_impl->mut_tensor_meta()->set_stride(stride);
      }
      const auto& dep_object = NewLocalDepObject();
      JUST(tensor_impl->InitEagerBlobObject(dep_object));
      output_eager_blob_objects->at(i) = JUST(tensor_impl->eager_blob_object());
    } else {
      // output i is inplaced.
      // check thread_local TensorMeta and tensor_impl TensorMeta.
      CHECK_OR_RETURN(tensor_impl->tensor_meta()->shape() == output_tensor_metas->at(i)->shape());
      // TODO:(thread_local TensorMeta set stride then check)
      // CHECK_OR_RETURN(tensor_impl->tensor_meta()->stride() ==
      // output_tensor_metas->at(i)->stride());
      CHECK_OR_RETURN(tensor_impl->tensor_meta()->dtype() == output_tensor_metas->at(i)->dtype());
    }
  }

  const auto& kernel = JUST(user_op_expr.MutKernel4Stream(stream));
  kernel->set_need_check_mem_case(need_check_mem_case);

  for (int64_t index : kernel->output_tuple_indexes4mut2_obns()) {
    output_eager_blob_objects->at(index)->set_is_shape_synced(false);
  }

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->Call(kernel, input_eager_blob_objects, output_eager_blob_objects, ctx, stream);
  }));
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

Maybe<one::UserOpExpr> EagerNcclBroadcast(Symbol<ParallelDesc> parallel_desc, int64_t root) {
  return one::OpBuilder("eager_nccl_broadcast", *JUST(UniqueStr("eager_nccl_broadcast")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Attr<int64_t>("root", root)
      .Build();
}

auto* CachedEagerNcclBroadcastOpExpr = DECORATE(&EagerNcclBroadcast, ThreadLocal);

}  // namespace

Maybe<Tensor> Broadcast(const std::shared_ptr<Tensor>& tensor, int64_t src_rank,
                        Symbol<ParallelDesc> parallel_desc, bool inplace) {
  CHECK_OR_RETURN(parallel_desc->containing_current_rank());
  if (parallel_desc->parallel_num() == 1 /* no broadcast */) { return tensor; }
  std::shared_ptr<UserOpExpr> op_expr =
      JUST(CachedEagerNcclBroadcastOpExpr(parallel_desc, src_rank));
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<int64_t>("root", src_rank));
  if (src_rank == GlobalProcessCtx::Rank() || inplace) {
    TensorTuple outputs{tensor};
    JUST(OpInterpUtil::Dispatch(*op_expr, {tensor}, &outputs,
                                one::OpExprInterpContext(attrs, parallel_desc)));
    return tensor;
  } else {
    return JUST(OpInterpUtil::Dispatch<one::Tensor>(
        *op_expr, {tensor}, one::OpExprInterpContext(attrs, parallel_desc)));
  }
}

namespace {

Maybe<Tensor> GetSyncedTensorIfBroadcast(const std::shared_ptr<Tensor>& tensor,
                                         Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp) {
  Optional<int64_t> parallel_id;
  JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  if (!parallel_id.has_value()) { return tensor; }
  const auto& broadcast_parallel_desc = JUST(GetBroadcastSubParallelDesc(parallel_desc, nd_sbp));
  int64_t root = JUST(broadcast_parallel_desc->MachineId4ParallelId(0));
  return Broadcast(tensor, root, broadcast_parallel_desc, false);
}

Maybe<Shape> CalcPhysicalShape(Symbol<ConsistentTensorMeta> consistent_tensor_meta) {
  const auto& opt_parallel_id =
      JUST(GetParallelId4CurrentProcessCtx(consistent_tensor_meta->parallel_desc()));
  int64_t parallel_id = JUST(*opt_parallel_id);
  return GetPhysicalShape(consistent_tensor_meta->shape(), *consistent_tensor_meta->nd_sbp(),
                          *consistent_tensor_meta->parallel_desc(), parallel_id);
}

static constexpr auto* GetPhysicalShape = DECORATE(&CalcPhysicalShape, ThreadLocal);

Maybe<Tensor> TryReshapeTensor(const std::shared_ptr<Tensor>& tensor,
                               Symbol<ConsistentTensorMeta> consistent_tensor_meta) {
  CHECK_OR_RETURN(tensor->is_local());
  const auto& physical_shape = JUST(GetPhysicalShape(consistent_tensor_meta));
  if (*physical_shape == *tensor->shape()) { return tensor; }
  CHECK_EQ_OR_RETURN(physical_shape->elem_cnt(), tensor->shape()->elem_cnt());
  // TODO(lixinqi) inplace reshape.
  return tensor;
}

}  // namespace

Maybe<void> EagerLocalInterpreter::ApplyImpl(const ConsistentToConsistentOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  OF_UNIMPLEMENTED();
}

namespace {

Maybe<void> RawLocalToConsistent(const CastToConsistentOpExpr& op_expr, const TensorTuple& inputs,
                                 TensorTuple* outputs, const OpExprInterpContext& ctx) {
  std::shared_ptr<LocalTensor> input_local_tensor;
  {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);
    CHECK_OR_RETURN(!inputs.at(0)->is_consistent());
    const auto& input_tensor = JUST(inputs.at(0)->detach());
    input_local_tensor = JUST(input_tensor->AsLocalTensor());
    CHECK_OR_RETURN(input_local_tensor) << Error::InvalidValueError("Tensor Cast Error");  // NOLINT
    bool requires_grad = autograd::GradMode::is_enabled() && inputs.at(0)->requires_grad();
    JUST(input_local_tensor->set_requires_grad(requires_grad));
    input_local_tensor->set_is_leaf(!requires_grad);
  }
  std::shared_ptr<ConsistentTensor> consistent_tensor;
  {
    CHECK_OR_RETURN(ctx.parallel_desc.has_value());
    CHECK_OR_RETURN(ctx.nd_sbp.has_value());
    const auto& nd_sbp = JUST(ctx.nd_sbp);
    const auto& parallel_desc = JUST(ctx.parallel_desc);
    const auto& logical_shape = JUST(ctx.attrs.GetAttr<Shape>("shape"));
    DataType dtype = JUST(ctx.attrs.GetAttr<DataType>("dtype"));
    ConsistentTensorMeta tensor_meta(std::make_shared<const Shape>(logical_shape), dtype, nd_sbp,
                                     parallel_desc);
    Optional<int64_t> parallel_id{};
    const auto& device = JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
    const auto& consistent_tensor_impl = JUST(EagerConsistentTensorImpl::New(
        SymbolOf(tensor_meta), device, parallel_id, input_local_tensor->requires_grad(),
        !input_local_tensor->requires_grad()));
    consistent_tensor = std::make_shared<ConsistentTensor>(consistent_tensor_impl);
    if (parallel_id.has_value()) {
      const auto& pyhsical_shape = JUST(GetPhysicalShape(tensor_meta));
      const auto& input_local_tensor_shape = input_local_tensor->shape();
      CHECK_EQ_OR_RETURN(*pyhsical_shape, *input_local_tensor_shape);      // NOLINT
      CHECK_OR_RETURN(dtype == input_local_tensor->dtype()->data_type());  // NOLINT
      consistent_tensor_impl->reset_cur_rank_phy_tensor(input_local_tensor);
    }
  }
  outputs->at(0) = consistent_tensor;
  return Maybe<void>::Ok();
}

static constexpr auto* LocalToConsistent =
    DECORATE(&RawLocalToConsistent, NonRecursiveInitConsistentId);

}  // namespace

Maybe<void> EagerLocalInterpreter::ApplyImpl(const CastToConsistentOpExpr& op_expr,
                                             const TensorTuple& inputs, TensorTuple* outputs,
                                             const OpExprInterpContext& ctx) const {
  JUST(LocalToConsistent(op_expr, inputs, outputs, ctx));
  const auto& consistent_tensor = JUST(outputs->at(0)->AsConsistentTensor());
  JUST(WithConsistencyChecked(consistent_tensor, [&]() -> Maybe<void> {
    if (IsConsistentTensorMetaCheckDisabled()) { return Maybe<void>::Ok(); }
    const auto& parallel_desc = JUST(ctx.parallel_desc);
    const auto& parallel_id = JUST(GetParallelId4CurrentProcessCtx(parallel_desc));
    if (!parallel_id->has_value()) { return Maybe<void>::Ok(); }
    const auto& nd_sbp = JUST(ctx.nd_sbp);
    const auto& tensor_meta = JUST(consistent_tensor->consistent_tensor_meta());
    const auto& local_tensor = JUST(consistent_tensor->cur_rank_phy_tensor());
    const auto& reshaped_tensor = JUST(TryReshapeTensor(local_tensor, tensor_meta));
    const auto& synced_tensor =
        JUST(GetSyncedTensorIfBroadcast(reshaped_tensor, parallel_desc, nd_sbp));
    auto* consistent_tensor_impl =
        reinterpret_cast<EagerConsistentTensorImpl*>(consistent_tensor->mut_impl());
    CHECK_NOTNULL_OR_RETURN(consistent_tensor_impl);
    consistent_tensor_impl->reset_cur_rank_phy_tensor(JUST(synced_tensor->AsLocalTensor()));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<void> EagerLocalInterpreter::ApplyImpl(const CastFromConsistentOpExpr& op_expr,
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
