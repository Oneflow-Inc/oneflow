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
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/blob_register.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::map<std::string, std::shared_ptr<::oneflow::compatible_py::BlobObject>>);

namespace oneflow {

namespace compatible_py {

using BlobName2Object = std::map<std::string, std::shared_ptr<BlobObject>>;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<BlobName2Object, std::shared_ptr<BlobName2Object>>(m, "BlobName2Object")
      .def(py::init<>())
      .def("__len__", [](const std::shared_ptr<BlobName2Object>& v) { return v->size(); })
      .def(
          "items",
          [](std::shared_ptr<BlobName2Object>& v) {
            return py::make_iterator(v->begin(), v->end());
          },
          py::keep_alive<0, 1>())
      .def("__getitem__", (BlobName2Object::mapped_type
                           & (BlobName2Object::*)(const BlobName2Object::key_type& pos))
                              & BlobName2Object::operator[])
      .def(
          "__iter__",
          [](std::shared_ptr<BlobName2Object>& v) {
            return py::make_iterator(v->begin(), v->end());
          },
          py::keep_alive<0, 1>());

  py::class_<BlobRegister, std::shared_ptr<BlobRegister>>(m, "BlobRegister")
      .def(py::init<>())
      .def_property_readonly("blob_name2object", &BlobRegister::blob_name2object)
      .def("OpenRegisteredBlobAccess", &BlobRegister::OpenRegisteredBlobAccess)
      .def("CloseRegisteredBlobAccess", &BlobRegister::CloseRegisteredBlobAccess)
      .def("HasObject4BlobName", &BlobRegister::HasObject4BlobName)
      .def("GetObject4BlobName", &BlobRegister::GetObject4BlobName)
      .def("SetObject4BlobName", &BlobRegister::SetObject4BlobName)
      .def("TrySetObject4BlobName", &BlobRegister::TrySetObject4BlobName)
      .def("ClearObject4BlobName", &BlobRegister::ClearObject4BlobName)
      .def("TryClearObject4BlobName", &BlobRegister::TryClearObject4BlobName)
      .def("ForceReleaseAll", &BlobRegister::ForceReleaseAll);

  py::class_<RegisteredBlobAccess, std::shared_ptr<RegisteredBlobAccess>>(m, "RegisteredBlobAccess")
      .def(py::init([](const std::string& blob_name,
                       const std::shared_ptr<BlobRegister>& blob_register,
                       const std::shared_ptr<BlobObject>& blob_object) {
        return std::make_shared<RegisteredBlobAccess>(blob_name, blob_register, blob_object);
      }))
      .def(py::init([](const std::string& blob_name,
                       const std::shared_ptr<BlobRegister>& blob_register, py::none) {
        return std::make_shared<RegisteredBlobAccess>(blob_name, blob_register,
                                                      std::shared_ptr<BlobObject>());
      }))
      .def(py::init(
          [](const std::string& blob_name, const std::shared_ptr<BlobRegister>& blob_register) {
            return std::make_shared<RegisteredBlobAccess>(blob_name, blob_register,
                                                          std::shared_ptr<BlobObject>());
          }))
      .def_property_readonly("reference_counter", &RegisteredBlobAccess::reference_counter)
      .def_property_readonly("blob_object", &RegisteredBlobAccess::blob_object)
      .def_property_readonly("blob_register", &RegisteredBlobAccess::blob_register)
      .def("increase_reference_counter", &RegisteredBlobAccess::increase_reference_counter)
      .def("decrease_reference_counter", &RegisteredBlobAccess::decrease_reference_counter);

  m.def("GetDefaultBlobRegister", []() { return GetDefaultBlobRegister().GetPtrOrThrow(); });
}

}  // namespace compatible_py

}  // namespace oneflow
