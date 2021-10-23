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
#ifndef ONEFLOW_EXTENSION_PYTHON_NUMPY_H_
#define ONEFLOW_EXTENSION_PYTHON_NUMPY_H_

#define NO_IMPORT_ARRAY
#include "oneflow/extension/python/numpy_internal.h"

namespace oneflow {

class NumPyArrayPtr final {
 public:
  NumPyArrayPtr(PyObject* obj)
      : internal_(std::make_shared<numpy::NumPyArrayInternal>(obj, []() -> void {})) {}
  NumPyArrayPtr(PyObject* obj, const std::function<void()>& deleter)
      : internal_(std::make_shared<numpy::NumPyArrayInternal>(obj, deleter)) {}

  void* data() const { return internal_->data(); }

  size_t size() const { return internal_->size(); }

 private:
  std::shared_ptr<numpy::NumPyArrayInternal> internal_;
};

}  // namespace oneflow

#endif  // ONEFLOW_EXTENSION_PYTHON_NUMPY_H_
