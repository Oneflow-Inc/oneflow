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
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/rank_group.h"

namespace py = pybind11;

namespace oneflow {

namespace {
Maybe<py::bytes> CpuBroadcast(py::bytes* in, int64_t root) {
  const auto& rank_group = JUST(RankGroup::DefaultRankGroup());
  const auto& parallel_desc = JUST(RankGroup::GetDefaultParallelDesc(DeviceType::kCPU, rank_group));
  Py_ssize_t length;
  char* buffer;
  if (GlobalProcessCtx::Rank() == root) {
    CHECK_NOTNULL_OR_RETURN(in);
    PyBytes_AsStringAndSize(in->ptr(), &buffer, &length);
  }
  JUST(ccl::Broadcast<DeviceType::kCPU>(&length, &length, sizeof(length), DataType::kChar, root,
                                        parallel_desc, nullptr));

  if (GlobalProcessCtx::Rank() == root) {
    JUST(ccl::Broadcast<DeviceType::kCPU>(buffer, buffer, length, DataType::kChar, root,
                                          parallel_desc, nullptr));
    return *in;
  } else {
    // https://github.com/pybind/pybind11/issues/1236#issuecomment-527730864
    PyBytesObject* bytesObject =
        static_cast<PyBytesObject*>(PyObject_Malloc(offsetof(PyBytesObject, ob_sval) + length + 1));

    PyObject_INIT_VAR(bytesObject, &PyBytes_Type, length);
    bytesObject->ob_shash = -1;
    bytesObject->ob_sval[length] = '\0';
    buffer = bytesObject->ob_sval;
    JUST(ccl::Broadcast<DeviceType::kCPU>(nullptr, buffer, length, DataType::kChar, root,
                                          parallel_desc, nullptr));
    return py::reinterpret_steal<py::bytes>(reinterpret_cast<PyObject*>(bytesObject));
  }
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("cpu_broadcast", [](py::bytes in, int64_t root) -> py::bytes {
    return CpuBroadcast(&in, root).GetOrThrow();
  });
  m.def("cpu_broadcast", [](py::none in, int64_t root) -> py::bytes {
    return CpuBroadcast(nullptr, root).GetOrThrow();
  });
}

}  // namespace oneflow
