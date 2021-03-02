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
#ifndef ONEFLOW_API_PYTHON_OFBLOB_OFBLOB_H_
#define ONEFLOW_API_PYTHON_OFBLOB_OFBLOB_H_

#include "oneflow/core/common/type_traits.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "oneflow/core/register/ofblob.h"

namespace py = pybind11;

inline int Ofblob_GetDataType(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->data_type();
}

inline size_t OfBlob_NumAxes(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->NumAxes();
}

inline bool OfBlob_IsDynamic(uint64_t of_blob_ptr) {
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->is_dynamic();
}

inline void OfBlob_CopyShapeFromNumpy(uint64_t of_blob_ptr, py::array_t<int64_t> array) {
  py::buffer_info buf = array.request();
  int64_t* buf_ptr = (int64_t*)buf.ptr;
  size_t size = buf.size;
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeFrom(buf_ptr, size);
}

inline void OfBlob_CopyShapeToNumpy(uint64_t of_blob_ptr, py::array_t<int64_t> array) {
  py::buffer_info buf = array.request();
  int64_t* buf_ptr = (int64_t*)buf.ptr;
  size_t size = buf.size;
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyShapeTo(buf_ptr, size);
}

inline void OfBlob_CopyStaticShapeTo(uint64_t of_blob_ptr, py::array_t<int64_t> array) {
  py::buffer_info buf = array.request();
  int64_t* buf_ptr = (int64_t*)buf.ptr;
  size_t size = buf.size;
  using namespace oneflow;
  auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);
  return of_blob->CopyStaticShapeTo(buf_ptr, size);
}

#endif  // ONEFLOW_API_PYTHON_OFBLOB_OFBLOB_H_
