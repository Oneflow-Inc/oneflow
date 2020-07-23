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
#ifndef ONEFLOW_CORE_VM_STRING_OBJECT_H_
#define ONEFLOW_CORE_VM_STRING_OBJECT_H_

#include <string>
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace vm {

class StringObject final : public Object {
 public:
  StringObject(const StringObject&) = delete;
  StringObject(StringObject&&) = delete;

  StringObject(const std::string& str) : str_(str) {}
  ~StringObject() override = default;

  const std::string& str() const { return str_; }

 private:
  std::string str_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STRING_OBJECT_H_
