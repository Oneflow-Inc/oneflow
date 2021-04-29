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
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {

namespace one {

class StatefulOpKernel;

using EagerBlobObjectList =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}  // namespace one

namespace user_op {

class OpKernel;

}  // namespace user_op

namespace vm {

class LocalCallOpKernelPhyInstrOperand final : public vm::PhyInstrOperand {
 public:
  LocalCallOpKernelPhyInstrOperand(const LocalCallOpKernelPhyInstrOperand&) = delete;
  LocalCallOpKernelPhyInstrOperand(LocalCallOpKernelPhyInstrOperand&&) = delete;
  ~LocalCallOpKernelPhyInstrOperand() override = default;

  LocalCallOpKernelPhyInstrOperand(const std::shared_ptr<one::StatefulOpKernel>& opkernel,
                                   const one::EagerBlobObjectList inputs,
                                   const one::EagerBlobObjectList outputs, const AttrMap& attrs)
      : opkernel_(opkernel), inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  const one::StatefulOpKernel& opkernel() const { return *opkernel_; }
  const one::EagerBlobObjectList& inputs() const { return inputs_; }
  const one::EagerBlobObjectList& outputs() const { return outputs_; }
  const AttrMap& attrs() const { return attrs_; }

  one::StatefulOpKernel* mut_opkernel() { return opkernel_.get(); }

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
  std::shared_ptr<one::StatefulOpKernel> opkernel_;
  one::EagerBlobObjectList inputs_;
  one::EagerBlobObjectList outputs_;
  const AttrMap attrs_;
  const user_op::OpKernel* user_opkernel_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LOCAL_CALL_OPKERNEL_PHY_INSTR_OPERAND_H_
