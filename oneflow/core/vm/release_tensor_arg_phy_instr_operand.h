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
#ifndef ONEFLOW_CORE_VM_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_

#include <functional>
#include <memory>
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {

class LocalDepObject;

namespace vm {

class EagerBlobObject;

class ReleaseTensorArgPhyInstrOperand : public PhyInstrOperand {
 public:
  ReleaseTensorArgPhyInstrOperand(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                                  LocalDepObject* compute_local_dep_object)
      : eager_blob_object_(eager_blob_object),
        compute_local_dep_object_(compute_local_dep_object) {}
  ~ReleaseTensorArgPhyInstrOperand() override = default;

  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }

  void ForEachConstMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMutMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMut2MirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

 private:
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
  LocalDepObject* compute_local_dep_object_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_RELEASE_TENSOR_ARG_PHY_INSTR_OPERAND_H_
