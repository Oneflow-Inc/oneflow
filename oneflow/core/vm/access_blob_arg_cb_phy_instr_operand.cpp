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
#include "oneflow/core/vm/access_blob_arg_cb_phy_instr_operand.h"
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/object_msg/object_msg_list.h"

namespace oneflow {

namespace vm {

void AccessBlobArgCbPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "const") {
    auto* compute_local_dep_object = compute_local_dep_object_->mut_local_dep_object();
    DoEach(nullptr, compute_local_dep_object->mut_mirrored_object());
  }
}

void AccessBlobArgCbPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "mut") {
    auto* compute_local_dep_object = compute_local_dep_object_->mut_local_dep_object();
    DoEach(nullptr, compute_local_dep_object->mut_mirrored_object());
  }
}

void AccessBlobArgCbPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "mut2") {
    auto* compute_local_dep_object = compute_local_dep_object_->mut_local_dep_object();
    DoEach(nullptr, compute_local_dep_object->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
