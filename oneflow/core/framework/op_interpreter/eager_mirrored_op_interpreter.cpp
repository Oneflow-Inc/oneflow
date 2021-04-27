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
std::shared_ptr<Device> GetDefaultDevice() {
  // TODO: align with pytorch (default cpu) when tensor.to() is ready
  return std::make_shared<Device>("cuda", 0);
}
}  // namespace

Maybe<void> NaiveInterpret(
    const UserOpExpr& user_op_expr,
    const std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>>&
        input_eager_blob_objects,
    const std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>>&
        output_eager_blob_objects,
    const AttrValueMap& attrs, const std::shared_ptr<const Device> device,
    std::shared_ptr<const ParallelDesc> parallel_desc) {
  const auto kernel = JUST(user_op_expr.MutKernel4Device(*device));
  const auto mem_case = kernel->mem_case();
  for (int i = 0; i < output_eager_blob_objects->size(); i++) {
    auto eager_blob_object = std::make_shared<eager::EagerBlobObject>(
        mem_case, std::make_shared<Shape>(), DataType::kInvalidDataType,
        std::make_shared<eager::TensorBuffer>(), parallel_desc);
    output_eager_blob_objects->at(i) = eager_blob_object;
  }
  kernel->InferDataType(input_eager_blob_objects, output_eager_blob_objects);

  auto build_instruction = [&](InstructionsBuilder* builder) -> Maybe<void> {
    JUST(builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                    attrs, parallel_desc));
    return Maybe<void>::Ok();
  };
  JUST(PhysicalRun(build_instruction));
  return Maybe<void>::Ok();
}

Maybe<eager::EagerBlobObject> GenerateAllocatedEagerBlobObject(DataType data_type,
                                                               const Shape& shape) {
  const auto zeros_expr = JUST(op_expr_helper::ZerosOp(shape, data_type));
  std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>> input_eager_blob_objects =
      std::make_shared<std::vector<std::shared_ptr<eager::EagerBlobObject>>>(0);
  std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>> output_eager_blob_objects =
      std::make_shared<std::vector<std::shared_ptr<eager::EagerBlobObject>>>(1);

  const auto device = GetDefaultDevice();
  std::shared_ptr<const ParallelDesc> parallel_desc =
      JUST(Device::MakeParallelDescByDevice(*device));

  JUST(NaiveInterpret(*zeros_expr, input_eager_blob_objects, output_eager_blob_objects,
                      AttrValueMap{}, device, parallel_desc));
  return output_eager_blob_objects->at(0);
}

static Maybe<void> NaiveInterpret(const BuiltinOpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const AttrValueMap& attrs) {
  std::shared_ptr<const Device> device;
  if (inputs.empty()) {
    device = GetDefaultDevice();
  } else {
    device = inputs.at(0)->device();
    for (int i = 1; i < inputs.size(); i++) { CHECK(*device == *inputs.at(i)->device()); }
  }
  std::shared_ptr<const ParallelDesc> parallel_desc =
      JUST(Device::MakeParallelDescByDevice(*device));
  const auto& user_op_expr = dynamic_cast<const UserOpExpr&>(op_expr);
  std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>> input_eager_blob_objects =
      std::make_shared<std::vector<std::shared_ptr<eager::EagerBlobObject>>>(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    input_eager_blob_objects->at(i) = JUST(inputs.at(i)->eager_blob_object());
  }
  std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>> output_eager_blob_objects =
      std::make_shared<std::vector<std::shared_ptr<eager::EagerBlobObject>>>(outputs->size());
  NaiveInterpret(user_op_expr, input_eager_blob_objects, output_eager_blob_objects, attrs, device,
                 parallel_desc);
  for (int i = 0; i < outputs->size(); ++i) {
    outputs->at(i) = JUST(OpInterpUtil::BuildEagerMirroredTensorFromEagerBlobObject(
        output_eager_blob_objects->at(i), device));
  }
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
  return NaiveInterpret(op_expr, inputs, outputs, attrs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 0);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  return NaiveInterpret(op_expr, inputs, outputs, attrs);
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
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
                                                const AttrValueMap& attrs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
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
                                                const AttrValueMap& attrs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                const TensorTuple& inputs, TensorTuple* outputs,
                                                const AttrValueMap& attrs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
