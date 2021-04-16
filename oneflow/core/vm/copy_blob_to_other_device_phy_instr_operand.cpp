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
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace vm {

namespace {

Maybe<void> MaybeForEachConstMirroredObject(
    const std::shared_ptr<one::MirroredTensor>& tensor,
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) {
  auto* infer_local_dep_object = JUST(tensor->infer_local_dep_object())->mut_local_dep_object();
  auto* compute_local_dep_object = JUST(tensor->infer_local_dep_object())->mut_local_dep_object();
  DoEach(infer_local_dep_object->mut_mirrored_object(),
         compute_local_dep_object->mut_mirrored_object());
  return Maybe<void>::Ok();
}

}  // namespace

void CopyBlobToOtherDevicePhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  CHECK_OK(MaybeForEachConstMirroredObject(tensor_, DoEach));
}

void CopyBlobToOtherDevicePhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  CHECK_OK(MaybeForEachConstMirroredObject(tensor_, DoEach));
}

void CopyBlobToOtherDevicePhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(MirroredObject* infer, MirroredObject* compute)>& DoEach) const {
  // Do nothing
}

}  // namespace vm
}  // namespace oneflow
