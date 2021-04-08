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
#ifndef ONEFLOW_CORE_FRAMEWORK_VM_LOCAL_DEP_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_VM_LOCAL_DEP_OBJECT_H_

#include "oneflow/core/object_msg/object_msg_core.h"
#include "oneflow/core/vm/vm_object.msg.h"

namespace oneflow {

class ParallelDesc;

namespace one {

// Helps VirtualMachine building instruction edges
class VmLocalDepObject final {
 public:
  explicit VmLocalDepObject(const std::shared_ptr<const ParallelDesc>& parallel_desc);
  ~VmLocalDepObject() = default;

  const ObjectMsgPtr<vm::LogicalObject>& infer_logical_object() const {
    return infer_logical_object_;
  }
  const ObjectMsgPtr<vm::LogicalObject>& compute_logical_object() const {
    return compute_logical_object_;
  }
  const ObjectMsgPtr<vm::MirroredObject>& infer_mirrored_object() const {
    return infer_mirrored_object_;
  }
  const ObjectMsgPtr<vm::MirroredObject>& compute_mirrored_object() const {
    return compute_mirrored_object_;
  }

 private:
  ObjectMsgPtr<vm::LogicalObject> infer_logical_object_;
  ObjectMsgPtr<vm::LogicalObject> compute_logical_object_;
  ObjectMsgPtr<vm::MirroredObject> infer_mirrored_object_;
  ObjectMsgPtr<vm::MirroredObject> compute_mirrored_object_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_VM_LOCAL_DEP_OBJECT_H_
