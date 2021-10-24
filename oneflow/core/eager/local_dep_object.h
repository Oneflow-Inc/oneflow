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

// Helps VirtualMachine building instruction edges
class LocalDepObject final : public intrusive::Base {
 public:
  // Getters
  const vm::LogicalObject& logical_object() const {
    if (logical_object_) { return logical_object_.Get(); }
    static const auto default_val = intrusive::make_shared<vm::LogicalObject>();
    return default_val.Get();
  }
  const vm::MirroredObject& mirrored_object() const {
    if (mirrored_object_) { return mirrored_object_.Get(); }
    static const auto default_val = intrusive::make_shared<vm::MirroredObject>();
    return default_val.Get();
  }

  const intrusive::ListHook& pool_hook() const { return pool_hook_; }
  const intrusive::ListHook& stored_hook() const { return stored_hook_; }
  const intrusive::ListHook& lifetime_hook() const { return lifetime_hook_; }

  // Setters
  vm::LogicalObject* mut_logical_object() {
    if (!logical_object_) { logical_object_ = intrusive::make_shared<vm::LogicalObject>(); }
    return logical_object_.Mutable();
  }
  vm::MirroredObject* mut_mirrored_object() {
    if (!mirrored_object_) { mirrored_object_ = intrusive::make_shared<vm::MirroredObject>(); }
    return mirrored_object_.Mutable();
  }

  // methods
  static Maybe<intrusive::shared_ptr<LocalDepObject>> New(const Device& device);

 private:
  Maybe<void> Init(const Device& device);

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  LocalDepObject()
      : intrusive_ref_(),
        logical_object_(),
        mirrored_object_(),
        pool_hook_(),
        stored_hook_(),
        lifetime_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  intrusive::shared_ptr<vm::LogicalObject> logical_object_;
  intrusive::shared_ptr<vm::MirroredObject> mirrored_object_;

 public:
  // list hooks
  intrusive::ListHook pool_hook_;
  intrusive::ListHook stored_hook_;
  intrusive::ListHook lifetime_hook_;
};

Maybe<LocalDepObject*> GetLocalDepObjectFromDevicePool(Symbol<Device> device);
Maybe<void> PutLocalDepObjectToDevicePool(Symbol<Device> device, LocalDepObject* local_dep_object);
Maybe<LocalDepObject*> GetLocalDepObject4Device(const Device& device);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
