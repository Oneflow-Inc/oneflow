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

Maybe<EagerMirroredTensorImpl*> TensorImpl4Tensor(const std::shared_ptr<Tensor>& tensor) {
  CHECK_OR_RETURN(static_cast<bool>(tensor));
  auto* tensor_ptr = dynamic_cast<MirroredTensor*>(tensor.get());
  CHECK_NOTNULL_OR_RETURN(tensor_ptr);
  CHECK_NOTNULL_OR_RETURN(tensor_ptr->mut_impl());
  auto* tensor_impl = dynamic_cast<EagerMirroredTensorImpl*>(tensor_ptr->mut_impl());
  CHECK_NOTNULL_OR_RETURN(tensor_impl);
  return tensor_impl;
}

}  // namespace

Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                           const std::shared_ptr<const Device>& default_device,
                           TensorTuple* outputs, const AttrMap& attrs) {
  std::shared_ptr<EagerBlobObjectList> input_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    const auto& input_device = JUST(inputs.at(i)->device());
    if (i > 0) {
      CHECK_OR_RETURN(*default_device == *input_device) << Error::InputDeviceNotMatchError();
    }
    input_eager_blob_objects->at(i) = JUST(inputs.at(i)->eager_blob_object());
  }
  for (int i = 0; i < outputs->size(); i++) {
    if (!outputs->at(i)) {
      outputs->at(i) =
          std::make_shared<MirroredTensor>(std::make_shared<EagerMirroredTensorImpl>());
    }
  }
  std::shared_ptr<EagerBlobObjectList> output_eager_blob_objects =
      std::make_shared<EagerBlobObjectList>(outputs->size());
  std::shared_ptr<const Device> op_device;
  std::shared_ptr<const ParallelDesc> op_parallel_desc;
  bool need_check_mem_case = true;
  bool need_event_record = false;

  // Infer devices
  if (!user_op_expr.has_device_infer_fn()) {
    op_device = default_device;
    op_parallel_desc = op_device->parallel_desc_ptr();
    for (int i = 0; i < outputs->size(); i++) {
      auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
      *tensor_impl->mut_device() = default_device;
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
        return CHECK_JUST(TensorImpl4Tensor(inputs.at(i)))->tensor_meta().get();
      },
      [&](int32_t i) -> TensorMeta* {
        return CHECK_JUST(TensorImpl4Tensor(outputs->at(i)))->mut_tensor_meta();
      }));

  for (int i = 0; i < output_eager_blob_objects->size(); i++) {
    auto* tensor_impl = JUST(TensorImpl4Tensor(outputs->at(i)));
    JUST(tensor_impl->InitEagerBlobObject(JUST(outputs->at(i)->device())->mem_case()));
    output_eager_blob_objects->at(i) = JUST(tensor_impl->eager_blob_object());
  }

  const auto kernel = JUST(user_op_expr.MutKernel4Device(*op_device));
  kernel->set_need_check_mem_case(need_check_mem_case);

  for (int64_t index : kernel->output_tuple_indexes4mut2_obns()) {
    output_eager_blob_objects->at(index)->set_is_shape_synced(false);
  }

  const auto& instr_type_name = JUST(op_device->local_call_instruction_name());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    if (need_event_record) {
      for (const auto& input_tensor : inputs) {
        const auto& tensor = std::dynamic_pointer_cast<one::MirroredTensor>(input_tensor);
        CHECK_OR_RETURN(static_cast<bool>(tensor));
        // Instruction `SoftSyncStream` records event which can be used to synchronize cuda
        // stream
        JUST(builder->SoftSyncStream(JUST(tensor->compute_local_dep_object()), "mut",
                                     JUST(tensor->device())->parallel_desc_ptr()));
      }
    }
    return builder->LocalCallOpKernel(kernel, input_eager_blob_objects, output_eager_blob_objects,
                                      attrs, op_parallel_desc, instr_type_name);
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
  JUST(NaiveInterpret(*empty_expr, *inputs, device, outputs, AttrMap{}));
  return Maybe<void>::Ok();
}

static Maybe<void> NaiveInterpret(const UserOpExpr& user_op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const AttrMap& attrs) {
  CHECK_EQ_OR_RETURN(outputs->size(), user_op_expr.output_size());
  std::shared_ptr<const Device> default_device;
  if (inputs.empty()) {
    default_device = JUST(GetDefaultDevice());
  } else {
    default_device = JUST(inputs.at(0)->device());
  }
  return NaiveInterpret(user_op_expr, inputs, default_device, outputs, attrs);
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
