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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_parallel.pb.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob_desc.cfg.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/common/protobuf.h"

namespace py = pybind11;

namespace oneflow {

namespace compatible_py {

namespace {

Maybe<cfg::SbpParallel> MakeSbpParallel(const std::string& serialized_str) {
  SbpParallel sbp_parallel;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &sbp_parallel))
      << "sbp_parallel parse failed";
  return std::make_shared<cfg::SbpParallel>(sbp_parallel);
}

Maybe<cfg::OptMirroredParallel> MakeOptMirroredParallel(const std::string& serialized_str) {
  OptMirroredParallel opt_mirrored_parallel;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &opt_mirrored_parallel))
      << "opt_mirrored_parallel parse failed";
  return std::make_shared<cfg::OptMirroredParallel>(opt_mirrored_parallel);
}

Maybe<cfg::OptInt64> MakeOptInt64(const std::string& serialized_str) {
  if (serialized_str.empty()) { return std::make_shared<cfg::OptInt64>(); }
  OptInt64 opt_int64;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &opt_int64)) << "opt_int64 parse failed";
  return std::make_shared<cfg::OptInt64>(opt_int64);
}

Maybe<cfg::BlobDescProto> MakeBlobDescProto(const std::string& serialized_str) {
  if (serialized_str.empty()) { return std::make_shared<cfg::BlobDescProto>(); }
  BlobDescProto blob_desc;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &blob_desc)) << "blob_desc parse failed";
  return std::make_shared<cfg::BlobDescProto>(blob_desc);
}

Maybe<OpArgBlobAttribute> CreatOpArgBlobAttribute(const std::string& blob_desc_str,
                                                  const std::string& logical_blob_name) {
  const std::shared_ptr<cfg::BlobDescProto>& blob_desc = JUST(MakeBlobDescProto(blob_desc_str));
  return std::make_shared<OpArgBlobAttribute>(blob_desc, logical_blob_name);
}

Maybe<OpArgParallelAttribute> CreatOpArgParallelAttribute(
    std::shared_ptr<ParallelDesc> parallel_desc, const std::string& sbp_parallel_str,
    const std::string& opt_mirrored_parallel_str) {
  std::shared_ptr<cfg::SbpParallel> sbp_parallel = JUST(MakeSbpParallel(sbp_parallel_str));
  std::shared_ptr<cfg::OptMirroredParallel> opt_mirrored_parallel =
      JUST(MakeOptMirroredParallel(opt_mirrored_parallel_str));
  return std::make_shared<OpArgParallelAttribute>(parallel_desc, sbp_parallel,
                                                  opt_mirrored_parallel);
}

Maybe<OpArgBlobAttribute> ApiGetOpArgBlobAttribute(const std::string& op_attribute_str,
                                                   const std::string& bn_in_op) {
  OpAttribute op_attribute;
  CHECK_OR_RETURN(TxtString2PbMessage(op_attribute_str, &op_attribute))
      << "op_attribute parse failed";
  return GetOpArgBlobAttribute(op_attribute, bn_in_op);
}

Maybe<OpArgParallelAttribute> ApiGetOpArgParallelAttribute(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol, const std::string& op_attribute_str,
    const std::string& bn_in_op) {
  OpAttribute op_attribute;
  CHECK_OR_RETURN(TxtString2PbMessage(op_attribute_str, &op_attribute))
      << "op_attribute parse failed";
  return GetOpArgParallelAttribute(parallel_desc_symbol, op_attribute, bn_in_op);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<OpArgBlobAttribute, std::shared_ptr<OpArgBlobAttribute>>(m, "OpArgBlobAttribute")
      .def(py::init([](const std::string& blob_desc_str, const std::string& logical_blob_name) {
        return CreatOpArgBlobAttribute(blob_desc_str, logical_blob_name).GetPtrOrThrow();
      }))
      .def_property_readonly("blob_desc", &OpArgBlobAttribute::blob_desc)
      .def_property_readonly("logical_blob_name", &OpArgBlobAttribute::logical_blob_name)
      .def_property_readonly("is_tensor_list", &OpArgBlobAttribute::is_tensor_list)
      .def_property_readonly("is_dynamic", &OpArgBlobAttribute::is_dynamic)
      .def_property_readonly("shape",
                             [](const std::shared_ptr<OpArgBlobAttribute>& x) {
                               const auto& x_shape = x->shape();
                               py::tuple ret(x_shape->NumAxes());
                               for (int i = 0; i < x_shape->NumAxes(); ++i) {
                                 ret[i] = x_shape->At(i);
                               }
                               return ret;
                             })
      .def("get_dtype",
           [](const std::shared_ptr<OpArgBlobAttribute>& x) {
             return static_cast<int>(x->get_dtype());
           })
      .def("GetPhysicalOpArgBlobAttr", &OpArgBlobAttribute::GetPhysicalOpArgBlobAttr)
      .def("DumpToInterfaceBlobConf", &OpArgBlobAttribute::DumpToInterfaceBlobConf)
      .def("DumpToOpNodeSignature", &OpArgBlobAttribute::DumpToOpNodeSignature)
      .def(py::self == py::self);

  py::class_<OpArgParallelAttribute, std::shared_ptr<OpArgParallelAttribute>>(
      m, "OpArgParallelAttribute")
      .def(py::init([](std::shared_ptr<ParallelDesc> parallel_desc,
                       const std::string& sbp_parallel_str,
                       const std::string& opt_mirrored_parallel_str) {
        return CreatOpArgParallelAttribute(parallel_desc, sbp_parallel_str,
                                           opt_mirrored_parallel_str)
            .GetPtrOrThrow();
      }))
      .def_property_readonly("parallel_desc_symbol", &OpArgParallelAttribute::parallel_desc_symbol)
      .def_property_readonly("sbp_parallel", &OpArgParallelAttribute::sbp_parallel)
      .def_property_readonly("opt_mirrored_parallel",
                             &OpArgParallelAttribute::opt_mirrored_parallel)
      .def("is_mirrored", &OpArgParallelAttribute::is_mirrored)
      .def("_Hash", &OpArgParallelAttribute::_Hash)
      .def("Assign", &OpArgParallelAttribute::Assign)
      .def("DumpToInterfaceBlobConf", &OpArgParallelAttribute::DumpToInterfaceBlobConf)
      .def("DumpToOpNodeSignature", &OpArgParallelAttribute::DumpToOpNodeSignature)
      .def("__str__", &OpArgParallelAttribute::ToString)
      .def("__repr__", &OpArgParallelAttribute::ToString)
      .def(py::self == py::self)
      .def(py::hash(py::self));
  m.def("GetOpArgBlobAttribute",
        [](const std::string& op_attribute_str, const std::string& bn_in_op) {
          return ApiGetOpArgBlobAttribute(op_attribute_str, bn_in_op).GetPtrOrThrow();
        });
  m.def("GetOpArgParallelAttribute",
        [](const std::shared_ptr<ParallelDesc>& parallel_desc_symbol,
           const std::string& op_attribute_str, const std::string& bn_in_op) {
          return ApiGetOpArgParallelAttribute(parallel_desc_symbol, op_attribute_str, bn_in_op)
              .GetPtrOrThrow();
        });
  m.def("MakeMirroredOpArgParallelAttribute",
        [](const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
          return MakeMirroredOpArgParallelAttribute(parallel_desc_symbol).GetPtrOrThrow();
        });
  m.def("MakeBroadcastOpArgParallelAttribute",
        [](const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
          return MakeBroadcastOpArgParallelAttribute(parallel_desc_symbol).GetPtrOrThrow();
        });
}

}  // namespace compatible_py

}  // namespace oneflow
