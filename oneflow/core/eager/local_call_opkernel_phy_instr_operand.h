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
#ifndef ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace one {

class StatefulLocalOpKernel;

using EagerBlobObjectList = std::vector<std::shared_ptr<vm::EagerBlobObject>>;
using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;
// using DTREagerBlobObjectListPtr =
//     std::shared_ptr<const std::vector<std::shared_ptr<vm::DTREagerBlobObject>>>;
}  // namespace one

namespace user_op {

class OpKernel;

}  // namespace user_op

namespace vm {

template<typename ObjectListPtr>
class LocalCallOpKernelPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  LocalCallOpKernelPhyInstrOperand(const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
                                   const ObjectListPtr& inputs,
                                   const ObjectListPtr& outputs,
                                   const one::OpExprInterpContext& op_interp_ctx_)
      : opkernel_(opkernel), inputs_(inputs), outputs_(outputs), op_interp_ctx_(op_interp_ctx_) {}

  const one::StatefulLocalOpKernel& opkernel() const { return *opkernel_; }
  const ObjectListPtr& inputs() const { return inputs_; }
  const ObjectListPtr& outputs() const { return outputs_; }
  const AttrMap& attrs() const { return op_interp_ctx_.attrs; }
  const one::OpExprInterpContext& op_interp_ctx() const { return op_interp_ctx_; }

  one::StatefulLocalOpKernel* mut_opkernel() { return opkernel_.get(); }

  template<typename DoEachT>
  Maybe<void> ForEachOutputTensor(const DoEachT& DoEach) {
    for (const auto& output : *outputs()) { JUST(DoEach(output.get())); }
    return Maybe<void>::Ok();
  }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  const user_op::OpKernel* user_opkernel() const { return user_opkernel_; }

  void set_user_opkernel(const user_op::OpKernel* user_opkernel) { user_opkernel_ = user_opkernel; }

 private:
  std::shared_ptr<one::StatefulLocalOpKernel> opkernel_;
  ObjectListPtr inputs_;
  ObjectListPtr outputs_;
  const one::OpExprInterpContext op_interp_ctx_;
  const user_op::OpKernel* user_opkernel_;
};

template <typename ObjectListPtr>
void LocalCallOpKernelPhyInstrOperand<ObjectListPtr>::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4const_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(nullptr, CHECK_JUST(input->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

template <typename ObjectListPtr>
void LocalCallOpKernelPhyInstrOperand<ObjectListPtr>::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  // Sequantialize instructions in the same stream by consuming `compute_local_dep_object` of the
  // same device.
  auto* device_dep_object = opkernel().device()->mut_compute_local_dep_object();
  DoEach(nullptr, device_dep_object->mut_local_dep_object()->mut_mirrored_object());

  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4mut_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(nullptr, CHECK_JUST(input->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut_obns()) {
    const auto& output = output_list->at(index);
    DoEach(nullptr, CHECK_JUST(output->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

template <typename ObjectListPtr>
void LocalCallOpKernelPhyInstrOperand<ObjectListPtr>::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut2_obns()) {
    const auto& output = output_list->at(index);
    DoEach(nullptr, CHECK_JUST(output->compute_local_dep_object())
                        ->mut_local_dep_object()
                        ->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
