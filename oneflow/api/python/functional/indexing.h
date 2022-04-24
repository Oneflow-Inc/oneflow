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
#ifndef ONEFLOW_API_PYTHON_FUNCTIONAL_INDEXING_H_
#define ONEFLOW_API_PYTHON_FUNCTIONAL_INDEXING_H_

#include <Python.h>

#include "oneflow/api/python/functional/common.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/tensor_index.h"

namespace oneflow {
namespace one {
namespace functional {

namespace detail {

void PySliceUnpack(PyObject* object, Py_ssize_t* start, Py_ssize_t* stop, Py_ssize_t* step);

Maybe<Tensor> ConvertToIndexingTensor(PyObject* object);

IndexItem UnpackIndexItem(PyObject* object);

}  // namespace detail

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FUNCTIONAL_INDEXING_H_
