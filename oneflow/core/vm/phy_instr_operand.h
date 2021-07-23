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
#ifndef ONEFLOW_CORE_VM_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_PHY_INSTR_OPERAND_H_

#include <functional>
#include "oneflow/core/object_msg/object_msg_core.h"

namespace oneflow {
namespace vm {

class MirroredObject;

// physical instruction operand
class PhyInstrOperand {
 public:
  virtual ~PhyInstrOperand() = default;

  virtual void ForEachConstMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const = 0;

  virtual void ForEachMutMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const = 0;

  virtual void ForEachMut2MirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const = 0;

 protected:
  PhyInstrOperand() = default;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_PHY_INSTR_OPERAND_H_
