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
#ifndef ONEFLOW_CORE_VM_COPY_BLOB_TO_OTHER_DEVICE_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_COPY_BLOB_TO_OTHER_DEVICE_PHY_INSTR_OPERAND_H_

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {

class VmLocalDepObject;

namespace eager {

class EagerBlobObject;
}

namespace one {

class MirroredTensor;
}

namespace vm {

class CopyBlobToOtherDevicePhyInstrOperand final : public PhyInstrOperand {
 public:
  CopyBlobToOtherDevicePhyInstrOperand(const std::shared_ptr<one::MirroredTensor>& tensor,
                                       const std::shared_ptr<one::MirroredTensor>& dest_tensor)
      : tensor_(tensor), dest_tensor_(dest_tensor) {}
  ~CopyBlobToOtherDevicePhyInstrOperand() override = default;

  const std::shared_ptr<one::MirroredTensor>& src_tensor() const { return tensor_; }
  const std::shared_ptr<one::MirroredTensor>& dest_tensor() const { return dest_tensor_; }

  void ForEachConstMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMutMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMut2MirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

 private:
  std::shared_ptr<one::MirroredTensor> tensor_;
  std::shared_ptr<one::MirroredTensor> dest_tensor_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_COPY_BLOB_TO_OTHER_DEVICE_PHY_INSTR_OPERAND_H_
