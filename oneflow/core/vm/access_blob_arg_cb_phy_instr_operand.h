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
#ifndef ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_PHY_INSTR_OPERAND_H_

#include <functional>
#include <memory>
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {

namespace one {

class TensorStorage;
}

namespace vm {

class EagerBlobObject;

// access blob arg callback physical instruction operand
class AccessBlobArgCbPhyInstrOperand : public PhyInstrOperand {
 public:
  AccessBlobArgCbPhyInstrOperand(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                                 const std::function<void(uint64_t)>& callback,
                                 const std::string& modifier)
      : eager_blob_object_(eager_blob_object),
        callback_(callback),
        modifier_(modifier),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
    stream_sequential_dependence_ = nullptr;
  }
  ~AccessBlobArgCbPhyInstrOperand() = default;

  const std::function<void(uint64_t)>& callback() const { return callback_; }
  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstMirroredObject(const std::function<void(MirroredObject* compute)>&) const;

  void ForEachMutMirroredObject(const std::function<void(MirroredObject* compute)>&) const;

  void ForEachMut2MirroredObject(const std::function<void(MirroredObject* compute)>&) const;

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    DoEach(eager_blob_object_.get());
  }

 private:
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
  std::function<void(uint64_t)> callback_;
  const std::string modifier_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_PHY_INSTR_OPERAND_H_
