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

#include <pybind11/pybind11.h>
#include "oneflow/api/python/framework/throw.h"

#define NO_IMPORT_ARRAY
#include "oneflow/extension/python/numpy_internal.h"

namespace py = pybind11;

namespace oneflow {

class NumPyArrayHolder {
 public:
  NumPyArrayHolder(PyObject* obj, const std::function<void()>& deleter) {
    CHECK_OR_THROW(PyArray_Check(obj)) << "Object is not numpy array.";
    obj_ = PyArray_GETCONTIGUOUS((PyArrayObject*)obj);
    deleter_ = [deleter, this]() {
      {
        py::gil_scoped_acquire acquire;
        Py_DECREF(obj_);
      }
      if (deleter) { deleter(); }
    };
    size_ = PyArray_SIZE(obj_);
    data_ = PyArray_DATA(obj_);
  }

  ~NumPyArrayHolder() {
    if (deleter_) { deleter_(); }
  }

  void* data() const { return data_; }

  size_t size() const { return size_; }

 private:
  PyArrayObject* obj_;
  void* data_;
  size_t size_;
  std::function<void()> deleter_;
};

class NumPyArrayPtr {
 public:
  NumPyArrayPtr(PyObject* obj, const std::function<void()>& deleter)
      : holder_(std::make_shared<NumPyArrayHolder>(obj, deleter)) {}

  void* data() const { return holder_->data(); }

  size_t size() const { return holder_->size(); }

 private:
  std::shared_ptr<NumPyArrayHolder> holder_;
};

}  // namespace oneflow

#endif  // ONEFLOW_EXTENSION_PYTHON_NUMPY_H_
