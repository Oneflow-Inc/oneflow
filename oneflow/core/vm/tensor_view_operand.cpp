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
#include "oneflow/core/vm/tensor_view_operand.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/intrusive/list.h"

namespace oneflow {

namespace vm {

TensorViewOperand::TensorViewOperand(
    const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
    const std::shared_ptr<vm::EagerBlobObject>& view_eager_blob_object)
    : eager_blob_object_(eager_blob_object),
      view_eager_blob_object_(view_eager_blob_object),
      input_dependences_(),
      output_dependences_() {
  ForEachConstMirroredObject(SetInserter(&input_dependences_));
  ForEachMutMirroredObject(SetInserter(&output_dependences_));
  ForEachMut2MirroredObject(SetInserter(&output_dependences_));
  if (eager_blob_object->producer_stream().has_value()) {
    stream_sequential_dependence_ =
        CHECK_JUST(eager_blob_object->producer_stream())->mut_schedule_local_dep_object();
  }
}

void TensorViewOperand::ForEachConstMirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {}

void TensorViewOperand::ForEachMutMirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {
  DoEach(CHECK_JUST(eager_blob_object_->compute_local_dep_object()));
}

void TensorViewOperand::ForEachMut2MirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {}

}  // namespace vm
}  // namespace oneflow
