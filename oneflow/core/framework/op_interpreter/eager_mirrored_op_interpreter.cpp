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

namespace oneflow {
namespace one {

namespace {
Maybe<const Device> GetDefaultDevice() { return Device::New("cpu", 0); }
}  // namespace

Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                           const std::shared_ptr<EagerBlobObjectList>& output_eager_blob_objects,
                           const AttrMap& attrs,
                           std::vector<std::shared_ptr<const Device>>* out_devices) {
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  std::shared_ptr<const Device> default_device;
  if (inputs.empty()) {
    default_device = JUST(GetDefaultDevice());
  } else {
    default_device = inputs.at(0)->device();
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (i > 0) { CHECK_OR_RETURN(*default_device == *inputs.at(i)->device()); }
    input_eager_blob_objects->at(i) = JUST(inputs.at(i)->eager_blob_object());
  }
  std::shared_ptr<const Device> op_device;
  std::shared_ptr<const ParallelDesc> op_parallel_desc;
  CHECK_EQ(out_devices->size(), output_eager_blob_objects->size());
  if (!user_op_expr.has_device_infer_fn()) {
    op_device = default_device;
    op_parallel_desc = op_device->parallel_desc_ptr();
    for (int i = 0; i < output_eager_blob_objects->size(); i++) {
      const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
          op_device->mem_case(), std::make_shared<Shape>(), DataType::kInvalidDataType,
          std::make_shared<vm::TensorBuffer>(), op_parallel_desc);
      output_eager_blob_objects->at(i) = eager_blob_object;
      out_devices->at(i) = default_device;
    }
  } else {
    op_device = JUST(user_op_expr.InferDevices(attrs, inputs, out_devices));
    op_parallel_desc = op_device->parallel_desc_ptr();
    for (int i = 0; i < output_eager_blob_objects->size(); i++) {
      const auto& tensor_device = out_devices->at(i);
      CHECK_OR_RETURN(static_cast<bool>(tensor_device));
      const auto& tensor_parallel_desc = op_device->parallel_desc_ptr();
      const auto& eager_blob_object = std::make_shared<vm::EagerBlobObject>(
          tensor_device->mem_case(), std::make_shared<Shape>(), DataType::kInvalidDataType,
          std::make_shared<vm::TensorBuffer>(), tensor_parallel_desc);
      output_eager_blob_objects->at(i) = eager_blob_object;
    }
  }

  const auto kernel = JUST(user_op_expr.MutKernel4Device(*op_device));

  for (int64_t index : kernel->output_tuple_indexes4mut2_obns()) {
    output_eager_blob_objects->at(index)->set_is_shape_synced(false);
  }

  kernel->ResetDynamicOpAttrs(attrs);
  JUST(kernel->InferDataType(input_eager_blob_objects, output_eager_blob_objects,
                             kernel->op_infer_ctx_for_thread_b()));
  JUST(kernel->InferTensorDesc(input_eager_blob_objects, output_eager_blob_objects,
                               kernel->op_infer_ctx_for_thread_b()));

  const auto& instr_type_name = JUST(op_device->local_call_instruction_name());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      attrs, op_parallel_desc, instr_type_name);
  }));
  return Maybe<void>::Ok();
}

Maybe<vm::EagerBlobObject> GenerateAllocatedEagerBlobObject(DataType data_type,
                                                            const Shape& shape) {
  const auto zeros_expr = JUST(op_expr_helper::ZerosOp(shape, data_type));
  std::shared_ptr<TensorTuple> inputs = std::make_shared<TensorTuple>();
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(1);
  std::vector<std::shared_ptr<const Device>> out_devices(1);
  JUST(NaiveInterpret(*zeros_expr, *inputs, output_eager_blob_objects, AttrMap{}, &out_devices));
  return output_eager_blob_objects->at(0);
}

static Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const AttrMap& attrs) {
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  std::vector<std::shared_ptr<const Device>> out_devices(outputs->size());
  NaiveInterpret(user_op_expr, inputs, output_eager_blob_objects, attrs, &out_devices);
  for (int i = 0; i < outputs->size(); ++i) {
    outputs->at(i) = JUST(OpInterpUtil::BuildEagerMirroredTensorFromEagerBlobObject(
        output_eager_blob_objects->at(i), out_devices.at(i)));
  }
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
  return NaiveInterpret(op_expr, inputs, outputs, attrs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
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
                                                const AttrMap& attrs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
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
                                                const AttrMap& attrs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrMap& attrs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
