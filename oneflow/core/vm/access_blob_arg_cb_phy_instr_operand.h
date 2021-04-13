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
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {

class VmLocalDepObject;

namespace eager {

class EagerBlobObject;
}

namespace one {

class TensorStorage;
}

namespace vm {

// access blob arg callback physical instruction operand
class WriteBlobArgCbPhyInstrOperand : public PhyInstrOperand {
 public:
  WriteBlobArgCbPhyInstrOperand(const std::shared_ptr<eager::EagerBlobObject>& eager_blob_object,
                                const std::shared_ptr<VmLocalDepObject>& infer_local_dep_object,
                                const std::shared_ptr<VmLocalDepObject>& compute_local_dep_object, 
                                const std::function<void(uint64_t)>& callback,
                                bool write_shape)
      : eager_blob_object_(eager_blob_object),
        callback_(callback),
        infer_local_dep_object_(infer_local_dep_object), 
        compute_local_dep_object_(compute_local_dep_object),
        write_shape_(write_shape) {}
  ~WriteBlobArgCbPhyInstrOperand() = default;

  const std::function<void(uint64_t)>& callback() const { return callback_; }
  const std::shared_ptr<eager::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }

  void ForEachInferMutMirroredObject(const std::function<void(MirroredObject*)>&) const override;
  void ForEachInferConstMirroredObject(const std::function<void(MirroredObject*)>&) const override;
  void ForEachComputeMutMirroredObject(const std::function<void(MirroredObject*)>&) const override;
  void ForEachComputeConstMirroredObject(
      const std::function<void(MirroredObject*)>&) const override;

 private:
  std::shared_ptr<eager::EagerBlobObject> eager_blob_object_;
  std::function<void(uint64_t)> callback_;
  std::shared_ptr<one::TensorStorage> tensor_storage_;
  std::shared_ptr<VmLocalDepObject> infer_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object_;
  const bool write_shape_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_PHY_INSTR_OPERAND_H_
