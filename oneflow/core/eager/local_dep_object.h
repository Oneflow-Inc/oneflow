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
#ifndef ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_

#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Device;

// clang-format off

// Helps VirtualMachine building instruction edges
INTRUSIVE_BEGIN(LocalDepObject);
 public:
  // Getters
  const vm::LogicalObject& logical_object() const {
    if (logical_object_) { return logical_object_.Get(); }
    static const auto default_val = intrusive::MakeShared<vm::LogicalObject>();
    return default_val.Get();
  }
  const vm::MirroredObject& mirrored_object() const {
    if (mirrored_object_) { return mirrored_object_.Get(); }
    static const auto default_val = intrusive::MakeShared<vm::MirroredObject>();
    return default_val.Get();
  }
  bool is_pool_entry_empty() const { return pool_entry_.empty(); }
  bool is_stored_entry_empty() const { return stored_entry_.empty(); }
  bool is_lifetime_entry_empty() const { return lifetime_entry_.empty(); }

  // Setters
  vm::LogicalObject* mut_logical_object() { return mutable_logical_object(); }
  vm::LogicalObject* mutable_logical_object() {
    if (!logical_object_) { logical_object_ = intrusive::MakeShared<vm::LogicalObject>(); }
    return logical_object_.Mutable();
  }
  vm::MirroredObject* mut_mirrored_object() { return mutable_mirrored_object(); }
  vm::MirroredObject* mutable_mirrored_object() {
    if (!mirrored_object_) { mirrored_object_ = intrusive::MakeShared<vm::MirroredObject>(); }
    return mirrored_object_.Mutable();
  }


  // methods
  OF_PUBLIC static Maybe<intrusive::SharedPtr<LocalDepObject>> New(const Device& device);

  OF_PRIVATE Maybe<void> Init(const Device& device);

  // fields
  INTRUSIVE_DEFINE_FIELD(intrusive::SharedPtr<vm::LogicalObject>, logical_object_);
  INTRUSIVE_DEFINE_FIELD(intrusive::SharedPtr<vm::MirroredObject>, mirrored_object_); 

  // list entries
  INTRUSIVE_DEFINE_FIELD(intrusive::ListEntry, pool_entry_);
  INTRUSIVE_DEFINE_FIELD(intrusive::ListEntry, stored_entry_);
  INTRUSIVE_DEFINE_FIELD(intrusive::ListEntry, lifetime_entry_);
INTRUSIVE_END(LocalDepObject);
// clang-format on

Maybe<LocalDepObject*> GetLocalDepObjectFromDevicePool(Symbol<Device> device);
Maybe<void> PutLocalDepObjectToDevicePool(Symbol<Device> device, LocalDepObject* local_dep_object);
Maybe<LocalDepObject*> GetLocalDepObject4Device(const Device& device);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
