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
#ifndef ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H
#define ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {

class LocalDepObject;

namespace vm {

class ConsumeLocalDepObjectPhyInstrOperand : public PhyInstrOperand {
 public:
  ConsumeLocalDepObjectPhyInstrOperand(LocalDepObject* compute_local_dep_object,
                                       const std::string& modifier)
      : compute_local_dep_object_(compute_local_dep_object), modifier_(modifier) {}
  ~ConsumeLocalDepObjectPhyInstrOperand() = default;

  void ForEachConstMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMutMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMut2MirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

 private:
  LocalDepObject* compute_local_dep_object_;
  const std::string modifier_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H
