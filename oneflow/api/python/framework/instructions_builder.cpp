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
#include <functional>
#include "oneflow/api/python/framework/size.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/tensor.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(
    std::unordered_map<std::string, std::shared_ptr<::oneflow::compatible_py::BlobObject>>);

namespace oneflow {

namespace {

std::shared_ptr<compatible_py::BlobObject> PackPhysicalBlobsToLogicalBlob(
    InstructionsBuilder* x,
    std::vector<std::shared_ptr<compatible_py::BlobObject>> physical_blob_objects,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  return x
      ->PackPhysicalBlobsToLogicalBlob(physical_blob_objects, op_arg_parallel_attr,
                                       op_arg_blob_attr)
      .GetPtrOrThrow();
}

std::shared_ptr<StringSymbol> GetSymbol4String(InstructionsBuilder* x, std::string str) {
  return x->GetSymbol4String(str).GetPtrOrThrow();
}

std::shared_ptr<JobDesc> GetJobConfSymbol(InstructionsBuilder* x,
                                          const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  return x->GetJobConfSymbol(job_conf).GetPtrOrThrow();
}

std::shared_ptr<ParallelDesc> GetParallelDescSymbol(
    InstructionsBuilder* x, const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->GetParallelDescSymbol(parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<Scope> GetScopeSymbol(InstructionsBuilder* x,
                                      const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  return x->GetScopeSymbol(scope_proto).GetPtrOrThrow();
}

std::vector<std::shared_ptr<ParallelDesc>> GetPhysicalParallelDescSymbols(
    InstructionsBuilder* x, const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  return *(x->GetPhysicalParallelDescSymbols(parallel_desc_symbol).GetPtrOrThrow());
}

std::vector<std::shared_ptr<compatible_py::BlobObject>> UnpackLogicalBlobToPhysicalBlobs(
    InstructionsBuilder* x, const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return *(x->UnpackLogicalBlobToPhysicalBlobs(blob_object).GetPtrOrThrow());
}

std::shared_ptr<compatible_py::BlobObject> MakeReferenceBlobObject(
    InstructionsBuilder* x, const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr) {
  return x->MakeReferenceBlobObject(blob_object, op_arg_parallel_attr).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildInitialScope(InstructionsBuilder* x, int64_t session_id,
                                         const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                                         const std::string& device_tag,
                                         const std::vector<std::string>& machine_device_ids,
                                         const std::shared_ptr<Shape>& hierarchy,
                                         bool is_mirrored) {
  return x
      ->BuildInitialScope(session_id, job_conf, device_tag, machine_device_ids, hierarchy,
                          is_mirrored)
      .GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelDesc(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy) {
  return x->BuildScopeWithNewParallelDesc(scope, device_tag, machine_device_ids, hierarchy)
      .GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelConf(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->BuildScopeWithNewParallelConf(scope, parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewIsMirrored(InstructionsBuilder* x,
                                                   const std::shared_ptr<Scope>& scope,
                                                   bool is_mirrored) {
  return x->BuildScopeWithNewIsMirrored(scope, is_mirrored).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewScopeName(InstructionsBuilder* x,
                                                  const std::shared_ptr<Scope>& scope,
                                                  std::string scope_name) {
  return x->BuildScopeWithNewScopeName(scope, scope_name).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeByProtoSetter(
    InstructionsBuilder* x, const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& Setter) {
  return x->BuildScopeByProtoSetter(scope, Setter).GetPtrOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> BroadcastBlobReference(
    InstructionsBuilder* x,
    const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return x->BroadcastBlobReference(sole_mirrored_blob_object, parallel_desc_sym).GetPtrOrThrow();
}

void Build121AssignInstruction(
    InstructionsBuilder* x, const std::shared_ptr<compatible_py::BlobObject>& ref_blob_object,
    const std::shared_ptr<compatible_py::BlobObject>& value_blob_object) {
  return x->Build121AssignInstruction(ref_blob_object, value_blob_object).GetOrThrow();
}

void CudaHostRegisterBlob(InstructionsBuilder* x,
                          const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return x->CudaHostRegisterBlob(blob_object).GetOrThrow();
}

void CudaHostUnregisterBlob(InstructionsBuilder* x,
                            const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return x->CudaHostUnregisterBlob(blob_object).GetOrThrow();
}

std::shared_ptr<compatible_py::OpKernelObject> NewOpKernelObject(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  return x->NewOpKernelObject(op_conf).GetPtrOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> MakeLazyRefBlobObject(
    InstructionsBuilder* x, const std::string& interface_op_name,
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->MakeLazyRefBlobObject(interface_op_name, op_attribute, parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<compatible_py::Object> GetSharedOpKernelObject4ParallelConfSymbol(
    InstructionsBuilder* x, const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return x->GetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym).GetPtrOrThrow();
}

void DeleteObject(InstructionsBuilder* x, compatible_py::Object* blob_object) {
  return x->DeleteObject(blob_object).GetOrThrow();
}

void InsertRemoveForeignCallbackInstruction(InstructionsBuilder* x, int64_t object_id,
                                            int64_t callback_id) {
  return x->InsertRemoveForeignCallbackInstruction(object_id, callback_id).GetOrThrow();
}

void FetchBlobHeader(InstructionsBuilder* x,
                     const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                     int64_t callback_id) {
  return x->FetchBlobHeader(blob_object, callback_id).GetOrThrow();
}

void FetchBlobBody(InstructionsBuilder* x,
                   const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                   int64_t callback_id) {
  return x->FetchBlobBody(blob_object, callback_id).GetOrThrow();
}

void FeedBlob(InstructionsBuilder* x, const std::shared_ptr<compatible_py::BlobObject>& blob_object,
              int64_t callback_id) {
  return x->FeedBlob(blob_object, callback_id).GetOrThrow();
}

void StatefulCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<compatible_py::OpKernelObject>& opkernel_object,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<compatible_py::BlobObject>(
        InstructionsBuilder*, const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& BoxingTo) {
  return x->StatefulCall(op_attribute, opkernel_object, bn_in_op2blob_object, BoxingTo)
      .GetOrThrow();
}

void StatelessCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<compatible_py::BlobObject>(
        InstructionsBuilder*, const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& BoxingTo) {
  return x->StatelessCall(op_attribute, parallel_conf, bn_in_op2blob_object, BoxingTo).GetOrThrow();
}

void NoBoxingStatelessCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  return x->NoBoxingStatelessCall(op_attribute, parallel_conf, bn_in_op2blob_object).GetOrThrow();
}

void NoBoxingCudaD2HStatelessCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& in_parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<ParallelDesc>(InstructionsBuilder*,
                                                      const std::shared_ptr<ParallelDesc>&,
                                                      const std::string&)>& TryReplaceDeviceTag) {
  return x
      ->NoBoxingCudaD2HStatelessCall(op_attribute, in_parallel_conf, bn_in_op2blob_object,
                                     TryReplaceDeviceTag)
      .GetOrThrow();
}

void NoBoxingCudaH2DStatelessCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& out_parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  return x->NoBoxingCudaH2DStatelessCall(op_attribute, out_parallel_conf, bn_in_op2blob_object)
      .GetOrThrow();
}

void RawStatelessCall(
    InstructionsBuilder* x, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  return x->RawStatelessCall(op_attribute, parallel_conf, bn_in_op2blob_object).GetOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> Build121To(
    InstructionsBuilder* x, const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  return x->Build121To(blob_object, parallel_desc_symbol).GetPtrOrThrow();
}

Maybe<void> DeprecatedLogicalRun(const std::function<void(InstructionsBuilder*)>& Build) {
  return LogicalRun([&](InstructionsBuilder* instruction_builder) -> Maybe<void> {
    Build(instruction_builder);
    return Maybe<void>::Ok();
  });
}

Maybe<void> DeprecatedPhysicalRun(const std::function<void(InstructionsBuilder*)>& Build) {
  return PhysicalRun([&](InstructionsBuilder* instruction_builder) -> Maybe<void> {
    Build(instruction_builder);
    return Maybe<void>::Ok();
  });
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  using BnInOp2BlobObject =
      std::unordered_map<std::string, std::shared_ptr<compatible_py::BlobObject>>;

  py::class_<BnInOp2BlobObject, std::shared_ptr<BnInOp2BlobObject>>(m, "BnInOp2BlobObject")
      .def(py::init<>())
      .def("__len__", [](const std::shared_ptr<BnInOp2BlobObject>& x) { return x->size(); })
      .def(
          "items",
          [](std::shared_ptr<BnInOp2BlobObject>& x) {
            return py::make_iterator(x->begin(), x->end());
          },
          py::keep_alive<0, 1>())
      .def("__getitem__", (BnInOp2BlobObject::mapped_type
                           & (BnInOp2BlobObject::*)(const BnInOp2BlobObject::key_type& pos))
                              & BnInOp2BlobObject::operator[])
      .def("__setitem__",
           [](std::shared_ptr<BnInOp2BlobObject>& x, const BnInOp2BlobObject::key_type& k,
              BnInOp2BlobObject::mapped_type& v) {
             auto it = x->find(k);
             if (it != x->end()) {
               it->second = v;
             } else {
               x->emplace(k, v);
             }
           })
      .def(
          "__iter__",
          [](std::shared_ptr<BnInOp2BlobObject>& x) {
            return py::make_iterator(x->begin(), x->end());
          },
          py::keep_alive<0, 1>());

  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def("id_generator", &InstructionsBuilder::id_generator)
      .def("eager_symbol_list", &InstructionsBuilder::eager_symbol_list)
      .def("object_releaser", &InstructionsBuilder::object_releaser)
      .def("PackPhysicalBlobsToLogicalBlob", &PackPhysicalBlobsToLogicalBlob)
      .def("GetSymbol4String", &GetSymbol4String)
      .def("GetJobConfSymbol", &GetJobConfSymbol)
      .def("GetParallelDescSymbol", &GetParallelDescSymbol)
      .def("GetScopeSymbol", &GetScopeSymbol)
      .def("GetPhysicalParallelDescSymbols", &GetPhysicalParallelDescSymbols)
      .def("UnpackLogicalBlobToPhysicalBlobs", &UnpackLogicalBlobToPhysicalBlobs)
      .def("MakeReferenceBlobObject", &MakeReferenceBlobObject)
      .def("BuildInitialScope", &BuildInitialScope, py::arg("session_id").none(false),
           py::arg("job_conf").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true),
           py::arg("is_mirrored").none(false))
      .def("BuildScopeWithNewParallelDesc", &BuildScopeWithNewParallelDesc,
           py::arg("scope").none(false), py::arg("device_tag").none(false),
           py::arg("machine_device_ids").none(false), py::arg("hierarchy").none(true))
      .def("BuildScopeWithNewParallelConf", &BuildScopeWithNewParallelConf)
      .def("BuildScopeWithNewIsMirrored", &BuildScopeWithNewIsMirrored)
      .def("BuildScopeWithNewScopeName", &BuildScopeWithNewScopeName)
      .def("BuildScopeByProtoSetter", &BuildScopeByProtoSetter)
      .def("BroadcastBlobReference", &BroadcastBlobReference)
      .def("Build121AssignInstruction", &Build121AssignInstruction)
      .def("CudaHostRegisterBlob", &CudaHostRegisterBlob)
      .def("CudaHostUnregisterBlob", &CudaHostUnregisterBlob)
      .def("NewOpKernelObject", &NewOpKernelObject)
      .def("MakeLazyRefBlobObject", &MakeLazyRefBlobObject)
      .def("GetSharedOpKernelObject4ParallelConfSymbol",
           &GetSharedOpKernelObject4ParallelConfSymbol)
      .def("DeleteObject", &DeleteObject)
      .def("StatefulCall", &StatefulCall)
      .def("InsertRemoveForeignCallbackInstruction", &InsertRemoveForeignCallbackInstruction)
      .def("FetchBlobHeader", &FetchBlobHeader)
      .def("FetchBlobBody", &FetchBlobBody)
      .def("FeedBlob", &FeedBlob)
      .def("StatelessCall", &StatelessCall)
      .def("NoBoxingStatelessCall", &NoBoxingStatelessCall)
      .def("NoBoxingCudaD2HStatelessCall", &NoBoxingCudaD2HStatelessCall)
      .def("NoBoxingCudaH2DStatelessCall", &NoBoxingCudaH2DStatelessCall)
      .def("RawStatelessCall", &RawStatelessCall)
      .def("Build121To", &Build121To);

  m.def(
      "LogicalRun",
      [](const std::function<void(InstructionsBuilder*)>& Build) {
        return DeprecatedLogicalRun(Build).GetOrThrow();
      },
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "PhysicalRun",
      [](const std::function<void(InstructionsBuilder*)>& Build) {
        return DeprecatedPhysicalRun(Build).GetOrThrow();
      },
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace oneflow
