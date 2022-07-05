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
#include <set>
#include <vector>
#include <memory>
#include "oneflow/core/intrusive/intrusive.h"

namespace oneflow {
namespace vm {

class MirroredObject;
class EagerBlobObject;

using DependenceVector = std::vector<MirroredObject*>;

// physical instruction operand
class PhyInstrOperand {
 public:
  virtual ~PhyInstrOperand() = default;

  virtual const DependenceVector& input_dependences() const = 0;
  virtual const DependenceVector& output_dependences() const = 0;
  virtual MirroredObject* stream_sequential_dependence() const {
    return stream_sequential_dependence_;
  }

  static std::function<void(MirroredObject*)> SetInserter(DependenceVector* dependences) {
    auto existed =
        std::make_shared<std::set<MirroredObject*>>(dependences->begin(), dependences->end());
    return [dependences, existed](MirroredObject* object) {
      if (existed->insert(object).second) { dependences->push_back(object); }
    };
  }

  virtual void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const = 0;

 protected:
  PhyInstrOperand() : stream_sequential_dependence_(nullptr) {}

  MirroredObject* stream_sequential_dependence_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_PHY_INSTR_OPERAND_H_
