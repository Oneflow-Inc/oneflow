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
#ifndef ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_
#define ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_

#include <memory>
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace vm {

template<typename T>
class ObjectWrapper final : public Object {
 public:
  explicit ObjectWrapper(const std::shared_ptr<T>& data) : data_(data) {}

  ~ObjectWrapper() = default;

  const T& operator*() const { return *data_; }
  T& operator*() { return *data_; }
  const T* operator->() const { return data_.get(); }
  T* operator->() { return data_.get(); }

  const std::shared_ptr<T>& GetPtr() const { return data_; }
  const T& Get() const { return *data_; }
  T* Mutable() { return data_.get(); }

 private:
  std::shared_ptr<T> data_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_OBJECT_WRAPPER_H_
