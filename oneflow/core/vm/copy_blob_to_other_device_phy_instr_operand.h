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

namespace vm {

class CopyBlobToOtherDevicePhyInstrOperand final : public PhyInstrOperand {
 public:
  CopyBlobToOtherDevicePhyInstrOperand(
      const std::shared_ptr<eager::EagerBlobObject>& src_eager_blob_object,
      const std::shared_ptr<eager::EagerBlobObject>& dst_eager_blob_object,
      const std::shared_ptr<VmLocalDepObject>& src_infer_local_dep_object,
      const std::shared_ptr<VmLocalDepObject>& dst_infer_local_dep_object,
      const std::shared_ptr<VmLocalDepObject>& src_compute_local_dep_object,
      const std::shared_ptr<VmLocalDepObject>& dst_compute_local_dep_object)
      : src_eager_blob_object_(src_eager_blob_object),
        dst_eager_blob_object_(dst_eager_blob_object),
        src_infer_local_dep_object_(src_infer_local_dep_object),
        dst_infer_local_dep_object_(dst_infer_local_dep_object),
        src_compute_local_dep_object_(src_compute_local_dep_object),
        dst_compute_local_dep_object_(dst_compute_local_dep_object) {}
  ~CopyBlobToOtherDevicePhyInstrOperand() override = default;

  const std::shared_ptr<eager::EagerBlobObject>& src_eager_blob_object() const {
    return src_eager_blob_object_;
  }
  const std::shared_ptr<eager::EagerBlobObject>& dst_eager_blob_object() const {
    return dst_eager_blob_object_;
  }

  void ForEachConstMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMutMirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

  void ForEachMut2MirroredObject(
      const std::function<void(MirroredObject* infer, MirroredObject* compute)>&) const override;

 private:
  std::shared_ptr<eager::EagerBlobObject> src_eager_blob_object_;
  std::shared_ptr<eager::EagerBlobObject> dst_eager_blob_object_;
  std::shared_ptr<VmLocalDepObject> src_infer_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> dst_infer_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> src_compute_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> dst_compute_local_dep_object_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_COPY_BLOB_TO_OTHER_DEVICE_PHY_INSTR_OPERAND_H_
