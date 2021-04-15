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
#include "oneflow/core/vm/copy_blob_to_other_device_phy_instr_operand.h"
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/object_msg/object_msg_list.h"

namespace oneflow {

namespace vm {

void CopyBlobToOtherDevicePhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  auto* src_infer_local_dep_object = src_infer_local_dep_object_->mut_local_dep_object();
  auto* src_compute_local_dep_object = src_compute_local_dep_object_->mut_local_dep_object();
  DoEach(src_infer_local_dep_object->mut_mirrored_object(),
         src_compute_local_dep_object->mut_mirrored_object());
}

void CopyBlobToOtherDevicePhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  auto* dst_infer_local_dep_object = dst_infer_local_dep_object_->mut_local_dep_object();
  auto* dst_compute_local_dep_object = dst_compute_local_dep_object_->mut_local_dep_object();
  DoEach(dst_infer_local_dep_object->mut_mirrored_object(),
         dst_compute_local_dep_object->mut_mirrored_object());
}

void CopyBlobToOtherDevicePhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  // Do nothing
}

}  // namespace vm
}  // namespace oneflow
