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
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/eager_blob_util.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<EagerPhysicalBlobHeader, std::shared_ptr<EagerPhysicalBlobHeader>>(
      m, "EagerPhysicalBlobHeader")
      .def(py::init([](const py::tuple& py_static_shape, const py::list& py_shape_list, int dtype,
                       bool is_tensor_list) {
        DimVector static_shape_dims;
        CHECK(py::isinstance<py::tuple>(py_static_shape));
        for (auto dim : py_static_shape) { static_shape_dims.emplace_back(dim.cast<int64_t>()); }
        std::shared_ptr<Shape> static_shape = std::make_shared<Shape>(static_shape_dims);
        CHECK(py::isinstance<py::list>(py_shape_list));
        std::shared_ptr<std::vector<std::shared_ptr<Shape>>> shape_list =
            std::make_shared<std::vector<std::shared_ptr<Shape>>>();
        for (const auto& py_shape : py_shape_list) {
          CHECK(py::isinstance<py::tuple>(py_shape));
          DimVector sub_shape_dims;
          for (auto dim : py_shape) { sub_shape_dims.emplace_back(dim.cast<int64_t>()); }
          shape_list->emplace_back(std::make_shared<Shape>(sub_shape_dims));
        }
        return std::make_shared<EagerPhysicalBlobHeader>(
            static_shape, shape_list, static_cast<DataType>(dtype), is_tensor_list);
      }))
      .def_property_readonly("static_shape",
                             [](const std::shared_ptr<EagerPhysicalBlobHeader>& x) {
                               const auto& x_shape = x->static_shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def_property_readonly("shape",
                             [](const std::shared_ptr<EagerPhysicalBlobHeader>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def_property_readonly("shape_list",
                             [](const std::shared_ptr<EagerPhysicalBlobHeader>& x) {
                               const auto& shape_list = x->shape_list();
                               py::list ret;
                               for (const auto& sub_shape : *shape_list) {
                                 py::tuple sub_shape_tuple(sub_shape->NumAxes());
                                 for (int i = 0; i < sub_shape->NumAxes(); ++i) {
                                   sub_shape_tuple[i] = sub_shape->At(i);
                                 }
                                 ret.append(sub_shape_tuple);
                               }
                               return ret;
                             })
      .def_property_readonly("is_tensor_list", &EagerPhysicalBlobHeader::is_tensor_list)
      .def("get_dtype", [](const std::shared_ptr<EagerPhysicalBlobHeader>& x) {
        return static_cast<int>(x->dtype());
      });

  py::class_<EagerPhysicalBlob, std::shared_ptr<EagerPhysicalBlob>>(m, "EagerPhysicalBlob")
      .def(py::init([](const std::string& blob_name,
                       const std::shared_ptr<BlobRegister>& blob_register,
                       const std::function<std::shared_ptr<EagerPhysicalBlobHeader>(
                           std::shared_ptr<BlobObject>)>& get_pysical_blob_header_cache) {
        return std::make_shared<EagerPhysicalBlob>(blob_name, blob_register,
                                                   get_pysical_blob_header_cache);
      }))
      .def_property_readonly("logical_blob_name", &EagerPhysicalBlob::logical_blob_name)
      .def_property_readonly("unique_name", &EagerPhysicalBlob::logical_blob_name)
      .def_property_readonly("static_shape",
                             [](const std::shared_ptr<EagerPhysicalBlob>& x) {
                               const auto& x_shape = x->static_shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def_property_readonly("shape",
                             [](const std::shared_ptr<EagerPhysicalBlob>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def_property_readonly("is_dynamic", &EagerPhysicalBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &EagerPhysicalBlob::is_tensor_list)
      .def_property_readonly("blob_object", &EagerPhysicalBlob::blob_object)
      .def("get_dtype",
           [](const std::shared_ptr<EagerPhysicalBlob>& x) { return static_cast<int>(x->dtype()); })
      .def("__str__", &EagerPhysicalBlob::ToString)
      .def("__repr__", &EagerPhysicalBlob::ToString);
}

}  // namespace compatible_py

}  // namespace oneflow
