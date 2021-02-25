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
#include "oneflow/core/framework/py_distribute.h"
#include "oneflow/core/framework/py_remote_blob.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

class TrampLazyConsistentBlob : public LazyConsistentBlob {
 public:
  using LazyConsistentBlob::LazyConsistentBlob;
  std::string get_lazy_shape_log_warning() const override {
    PYBIND11_OVERRIDE(std::string, LazyConsistentBlob, get_lazy_shape_log_warning, );
  }
};

class TrampLazyMirroredBlob : public LazyMirroredBlob {
 public:
  using LazyMirroredBlob::LazyMirroredBlob;
  std::string get_mirror_shape_log_warning() const override {
    PYBIND11_OVERRIDE(std::string, LazyMirroredBlob, get_mirror_shape_log_warning, );
  }
};

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.attr("INVALID_SPLIT_AXIS") = INVALID_SPLIT_AXIS;

  py::module_ distribute = m.def_submodule("distribute");
  py::class_<Distribute, std::shared_ptr<Distribute>>(distribute, "Distribute");
  py::class_<AutoDistribute, Distribute, std::shared_ptr<AutoDistribute>>(distribute,
                                                                          "AutoDistribute");
  py::class_<BroadcastDistribute, Distribute, std::shared_ptr<BroadcastDistribute>>(
      distribute, "BroadcastDistribute");
  py::class_<SplitDistribute, Distribute, std::shared_ptr<SplitDistribute>>(distribute,
                                                                            "SplitDistribute")
      .def_property_readonly("axis", &SplitDistribute::axis);
  distribute.def("auto", &GlobalAutoDistribute);
  distribute.def("broadcast", &GlobalBroadcastDistribute);
  distribute.def("split", [](int axis) { return GlobalSplitDistribute(axis).GetPtrOrThrow(); });

  py::class_<BlobDesc, std::shared_ptr<BlobDesc>>(m, "BlobDesc")
      .def(py::init(
          [](std::shared_ptr<cfg::LogicalBlobId> lbi, std::shared_ptr<Distribute> distribute) {
            return std::make_shared<BlobDesc>(lbi, distribute);
          }))
      .def_property_readonly("lbi", &BlobDesc::lbi)
      .def_property_readonly("logical_blob_name", &BlobDesc::logical_blob_name)
      .def_property_readonly("op_name", &BlobDesc::op_name)
      .def_property_readonly("blob_name", &BlobDesc::blob_name)
      .def_property_readonly("shape", &BlobDesc::shape)
      .def_property_readonly("dtype", &BlobDesc::dtype)
      .def_property_readonly("is_dynamic", &BlobDesc::is_dynamic)
      .def_property_readonly("is_tensor_list", &BlobDesc::is_tensor_list)
      .def_property_readonly("parallel_conf", &BlobDesc::parallel_conf)
      .def_property_readonly("distribute", &BlobDesc::distribute)
      .def_property_readonly("unique_name", &BlobDesc::unique_name)
      .def("set_distribute", &BlobDesc::set_distribute);

  py::class_<ConsistentBlob, BlobDesc, std::shared_ptr<ConsistentBlob>>(m, "ConsistentBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<ConsistentBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("lbi", &ConsistentBlob::lbi)
      .def_property_readonly("logical_blob_name", &ConsistentBlob::logical_blob_name)
      .def_property_readonly("op_name", &ConsistentBlob::op_name)
      .def_property_readonly("blob_name", &ConsistentBlob::blob_name)
      .def_property_readonly("shape", &ConsistentBlob::shape)
      .def_property_readonly("dtype", &ConsistentBlob::dtype)
      .def_property_readonly("is_dynamic", &ConsistentBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &ConsistentBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &ConsistentBlob::parallel_conf)
      .def_property_readonly("distribute", &ConsistentBlob::distribute)
      .def_property_readonly("unique_name", &ConsistentBlob::unique_name)
      .def_property_readonly("job_name", &ConsistentBlob::job_name)
      .def_property_readonly("parallel_size", &ConsistentBlob::parallel_size)
      .def("set_job_name", &ConsistentBlob::set_job_name);

  py::class_<LazyConsistentBlob, TrampLazyConsistentBlob, ConsistentBlob,
             std::shared_ptr<LazyConsistentBlob>>(m, "LazyConsistentBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<TrampLazyConsistentBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("shape",
                             [](const std::shared_ptr<LazyConsistentBlob>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def(
          "get_dtype",
          [](const std::shared_ptr<LazyConsistentBlob>& x) { return static_cast<int>(x->dtype()); })
      .def_property_readonly("split_axis", &LazyConsistentBlob::split_axis)
      .def_property_readonly("is_dynamic", &LazyConsistentBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &LazyConsistentBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &LazyConsistentBlob::parallel_conf)
      .def("IdenticalTo", &LazyConsistentBlob::IdenticalTo)
      .def("get_lazy_shape_log_warning", &LazyConsistentBlob::get_lazy_shape_log_warning);

  py::class_<MirroredBlob, BlobDesc, std::shared_ptr<MirroredBlob>>(m, "MirroredBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<MirroredBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("lbi", &MirroredBlob::lbi)
      .def_property_readonly("logical_blob_name", &MirroredBlob::logical_blob_name)
      .def_property_readonly("op_name", &MirroredBlob::op_name)
      .def_property_readonly("blob_name", &MirroredBlob::blob_name)
      .def_property_readonly("shape", &MirroredBlob::shape)
      .def_property_readonly("dtype", &MirroredBlob::dtype)
      .def_property_readonly("is_dynamic", &MirroredBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &MirroredBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &MirroredBlob::parallel_conf)
      .def_property_readonly("distribute", &MirroredBlob::distribute)
      .def_property_readonly("unique_name", &MirroredBlob::unique_name)
      .def_property_readonly("job_name", &MirroredBlob::job_name)
      .def_property_readonly("parallel_size", &MirroredBlob::parallel_size)
      .def("set_job_name", &MirroredBlob::set_job_name);

  py::class_<LazyMirroredBlob, TrampLazyMirroredBlob, MirroredBlob,
             std::shared_ptr<LazyMirroredBlob>>(m, "LazyMirroredBlob")
      .def(py::init([](std::shared_ptr<cfg::LogicalBlobId> lbi, std::string job_name,
                       std::shared_ptr<Distribute> distribute) {
        return std::make_shared<TrampLazyMirroredBlob>(lbi, job_name, distribute);
      }))
      .def_property_readonly("shape",
                             [](const std::shared_ptr<LazyMirroredBlob>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def("get_dtype",
           [](const std::shared_ptr<LazyMirroredBlob>& x) { return static_cast<int>(x->dtype()); })
      .def_property_readonly("split_axis", &LazyMirroredBlob::split_axis)
      .def_property_readonly("is_dynamic", &LazyMirroredBlob::is_dynamic)
      .def_property_readonly("is_tensor_list", &LazyMirroredBlob::is_tensor_list)
      .def_property_readonly("parallel_conf", &LazyMirroredBlob::parallel_conf)
      // The major downside of these implicit conversions is that containers must be converted (i.e.
      // copied) on every C++->Python transition, which can have implications on the program
      // semantics and performance.
      .def_property_readonly("sub_consistent_blob_list",
                             &LazyMirroredBlob::sub_consistent_blob_list)
      .def("get_mirror_shape_log_warning", &LazyMirroredBlob::get_mirror_shape_log_warning);

  py::class_<EagerBlobTrait, std::shared_ptr<EagerBlobTrait>>(m, "EagerBlobTrait")
      .def(py::init<>())
      .def("numpy_size", &EagerBlobTrait::numpy_size)
      .def("numpy_list_size", &EagerBlobTrait::numpy_list_size)
      .def("get_dtype",
           [](const std::shared_ptr<EagerBlobTrait>& x) { return static_cast<int>(x->dtype()); })
      .def_property_readonly("shape",
                             [](const std::shared_ptr<EagerBlobTrait>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def_property_readonly("split_axis", &EagerBlobTrait::split_axis)
      .def_property_readonly("is_dynamic", &EagerBlobTrait::is_dynamic)
      .def_property_readonly("is_tensor_list", &EagerBlobTrait::is_tensor_list)
      .def_property_readonly("parallel_conf", &EagerBlobTrait::parallel_conf)
      .def_property_readonly("parallel_size", &EagerBlobTrait::parallel_size)
      .def("_Init", &EagerBlobTrait::_Init)
      .def_property_readonly("blob_object", &EagerBlobTrait::blob_object)
      .def("IdenticalTo", &EagerBlobTrait::IdenticalTo);

  py::class_<EagerConsistentBlob, EagerBlobTrait, ConsistentBlob,
             std::shared_ptr<EagerConsistentBlob>>(m, "EagerConsistentBlob")
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                       const std::shared_ptr<BlobObject>& blob_object,
                       const std::shared_ptr<BlobRegister>& blob_register,
                       const std::string& job_name, const std::shared_ptr<Distribute>& distribute) {
             return std::make_shared<EagerConsistentBlob>(lbi, blob_object, blob_register, job_name,
                                                          distribute);
           }),
           py::arg("lbi"), py::arg("blob_object"), py::arg("blob_register"),
           py::arg("job_name") = "", py::arg("distribute") = GlobalAutoDistribute());

  py::class_<EagerMirroredBlob, EagerBlobTrait, MirroredBlob, std::shared_ptr<EagerMirroredBlob>>(
      m, "EagerMirroredBlob")
      .def(py::init([](const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                       const std::shared_ptr<BlobObject>& blob_object,
                       const std::shared_ptr<BlobRegister>& blob_register,
                       const std::string& job_name, const std::shared_ptr<Distribute>& distribute) {
             return std::make_shared<EagerMirroredBlob>(lbi, blob_object, blob_register, job_name,
                                                        distribute);
           }),
           py::arg("lbi"), py::arg("blob_object"), py::arg("blob_register"),
           py::arg("job_name") = "", py::arg("distribute") = GlobalAutoDistribute());
}

}  // namespace compatible_py

}  // namespace oneflow
