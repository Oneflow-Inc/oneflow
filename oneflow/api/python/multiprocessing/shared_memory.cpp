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
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/ipc/shared_memory.h"

namespace oneflow {

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("multiprocessing", m) {
  py::class_<ipc::SharedMemory, std::shared_ptr<ipc::SharedMemory>>(m, "SharedMemory")
      .def(py::init([](const std::string& name, bool create, size_t size) {
             if (create) { return ipc::SharedMemory::Open(size, create).GetPtrOrThrow(); }
             return ipc::SharedMemory::Open(name, create).GetPtrOrThrow();
           }),
           py::arg("name") = "", py::arg("create") = false, py::arg("size") = 0)
      .def("close", &ipc::SharedMemory::Close)
      .def("unlink", &ipc::SharedMemory::Unlink)
      .def_property_readonly("buf",
                             [](ipc::SharedMemory* shm) {
                               return py::memoryview::from_memory(shm->mut_buf(), shm->size());
                             })
      .def_property_readonly("name", &ipc::SharedMemory::name)
      .def_property_readonly("size", &ipc::SharedMemory::size);
  m.def("unlink_all_shared_memory",
        []() { return ipc::SharedMemoryManager::get().UnlinkAllShms(); });
}

}  // namespace oneflow
