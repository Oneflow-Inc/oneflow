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
#pragma once

#include <pybind11/pybind11.h>
#include "oneflow/api/python/of_api_registry.h"

// reference: pytorch/torch/csrc/utils/object_ptr.h
// https://github.com/pytorch/pytorch/blob/d69c22dd61a2f006dcfe1e3ea8468a3ecaf931aa/torch/csrc/utils/object_ptr.h
template<class T>
class OFPointer {
 public:
  OFPointer() : ptr(nullptr){};
  explicit OFPointer(T* ptr) noexcept : ptr(ptr){};
  OFPointer(OFPointer&& p) noexcept {
    free();
    ptr = p.ptr;
    p.ptr = nullptr;
  };

  ~OFPointer() { free(); };
  T* get() { return ptr; }
  const T* get() const { return ptr; }
  T* release() {
    T* tmp = ptr;
    ptr = nullptr;
    return tmp;
  }
  operator T*() { return ptr; }
  OFPointer& operator=(T* new_ptr) noexcept {
    free();
    ptr = new_ptr;
    return *this;
  }
  OFPointer& operator=(OFPointer&& p) noexcept {
    free();
    ptr = p.ptr;
    p.ptr = nullptr;
    return *this;
  }
  T* operator->() { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }

 private:
  void free();
  T* ptr = nullptr;
};

/**
 * An RAII-style, owning pointer to a PyObject.  You must protect
 * destruction of this object with the GIL.
 *
 * WARNING: Think twice before putting this as a field in a C++
 * struct.  This class does NOT take out the GIL on destruction,
 * so if you will need to ensure that the destructor of your struct
 * is either (a) always invoked when the GIL is taken or (b) takes
 * out the GIL itself.  Easiest way to avoid this problem is to
 * not use THPPointer in this situation.
 */
using OFObjectPtr = OFPointer<PyObject>;
