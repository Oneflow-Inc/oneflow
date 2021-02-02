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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace py = pybind11;

namespace oneflow {

namespace {

int64_t NewObjectId(const std::shared_ptr<InstructionsBuilder>& x,
                    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return x->NewObjectId(parallel_desc_sym).GetOrThrow();
}

int64_t NewSymbolId(const std::shared_ptr<InstructionsBuilder>& x) {
  return x->NewSymbolId().GetOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> PackPhysicalBlobsToLogicalBlob(
    const std::shared_ptr<InstructionsBuilder>& x,
    std::vector<std::shared_ptr<compatible_py::BlobObject>> physical_blob_objects,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  return x
      ->PackPhysicalBlobsToLogicalBlob(physical_blob_objects, op_arg_parallel_attr,
                                       op_arg_blob_attr)
      .GetPtrOrThrow();
}

std::shared_ptr<StringSymbol> GetSymbol4String(const std::shared_ptr<InstructionsBuilder>& x,
                                               std::string str) {
  return x->GetSymbol4String(str).GetPtrOrThrow();
}

std::shared_ptr<JobDesc> GetJobConfSymbol(const std::shared_ptr<InstructionsBuilder>& x,
                                          const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  return x->GetJobConfSymbol(job_conf).GetPtrOrThrow();
}

std::shared_ptr<ParallelDesc> GetParallelDescSymbol(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->GetParallelDescSymbol(parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<Scope> GetScopeSymbol(const std::shared_ptr<InstructionsBuilder>& x,
                                      const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  return x->GetScopeSymbol(scope_proto).GetPtrOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> NewBlobObject(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  return x->NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr).GetPtrOrThrow();
}

int64_t NewSymbolId4OpNodeSignature(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym) {
  return x->NewSymbolId4OpNodeSignature(op_node_signature_sym).GetOrThrow();
}

int64_t NewSharedOpKernelObjectId4ParallelConfSymbolId(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return x->NewSharedOpKernelObjectId4ParallelConfSymbolId(parallel_desc_sym).GetOrThrow();
}

std::vector<std::shared_ptr<ParallelDesc>> GetPhysicalParallelDescSymbols(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  return *(x->GetPhysicalParallelDescSymbols(parallel_desc_symbol).GetPtrOrThrow());
}

std::vector<std::shared_ptr<compatible_py::BlobObject>> UnpackLogicalBlobToPhysicalBlobs(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return *(x->UnpackLogicalBlobToPhysicalBlobs(blob_object).GetPtrOrThrow());
}

std::shared_ptr<compatible_py::BlobObject> MakeReferenceBlobObject(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr) {
  return x->MakeReferenceBlobObject(blob_object, op_arg_parallel_attr).GetPtrOrThrow();
}

void ReplaceMirrored(const std::shared_ptr<InstructionsBuilder>& x,
                     const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
                     std::vector<std::shared_ptr<compatible_py::BlobObject>> lhs_objects,
                     std::vector<std::shared_ptr<compatible_py::BlobObject>> rhs_objects) {
  return x->ReplaceMirrored(parallel_desc_sym, lhs_objects, rhs_objects).GetOrThrow();
}

std::shared_ptr<Scope> BuildInitialScope(const std::shared_ptr<InstructionsBuilder>& x,
                                         int64_t session_id,
                                         const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                                         const std::string& device_tag,
                                         const std::vector<std::string>& machine_device_ids,
                                         bool is_mirrored) {
  return x->BuildInitialScope(session_id, job_conf, device_tag, machine_device_ids, is_mirrored)
      .GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelDesc(
    const std::shared_ptr<InstructionsBuilder>& x, const std::shared_ptr<Scope>& scope,
    const std::string& device_tag, const std::vector<std::string>& machine_device_ids) {
  return x->BuildScopeWithNewParallelDesc(scope, device_tag, machine_device_ids).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewParallelConf(
    const std::shared_ptr<InstructionsBuilder>& x, const std::shared_ptr<Scope>& scope,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  return x->BuildScopeWithNewParallelConf(scope, parallel_conf).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewIsMirrored(const std::shared_ptr<InstructionsBuilder>& x,
                                                   const std::shared_ptr<Scope>& scope,
                                                   bool is_mirrored) {
  return x->BuildScopeWithNewIsMirrored(scope, is_mirrored).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeWithNewScopeName(const std::shared_ptr<InstructionsBuilder>& x,
                                                  const std::shared_ptr<Scope>& scope,
                                                  std::string scope_name) {
  return x->BuildScopeWithNewScopeName(scope, scope_name).GetPtrOrThrow();
}

std::shared_ptr<Scope> BuildScopeByProtoSetter(
    const std::shared_ptr<InstructionsBuilder>& x, const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& setter) {
  return x->BuildScopeByProtoSetter(scope, setter).GetPtrOrThrow();
}

std::shared_ptr<compatible_py::BlobObject> BroadcastBlobReference(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return x->BroadcastBlobReference(sole_mirrored_blob_object, parallel_desc_sym).GetPtrOrThrow();
}

void Build121AssignInstruction(
    const std::shared_ptr<InstructionsBuilder>& x,
    const std::shared_ptr<compatible_py::BlobObject>& ref_blob_object,
    const std::shared_ptr<compatible_py::BlobObject>& value_blob_object) {
  return x->Build121AssignInstruction(ref_blob_object, value_blob_object).GetOrThrow();
}

void CudaHostRegisterBlob(const std::shared_ptr<InstructionsBuilder>& x,
                          const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return x->CudaHostRegisterBlob(blob_object).GetOrThrow();
}

void CudaHostUnregisterBlob(const std::shared_ptr<InstructionsBuilder>& x,
                            const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  return x->CudaHostUnregisterBlob(blob_object).GetOrThrow();
}

void LazyReference(const std::shared_ptr<InstructionsBuilder>& x,
                   const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                   std::string interface_op_name) {
  return x->LazyReference(blob_object, interface_op_name).GetOrThrow();
}

void DeleteObject(const std::shared_ptr<InstructionsBuilder>& x,
                  compatible_py::BlobObject* blob_object) {
  return x->DeleteObject(blob_object).GetOrThrow();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  py::class_<InstructionsBuilder, std::shared_ptr<InstructionsBuilder>>(m, "InstructionsBuilder")
      .def(py::init([](const std::shared_ptr<vm::IdGenerator>& id_generator,
                       const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                       const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list,
                       const std::function<void(compatible_py::BlobObject*)>& release_object) {
        return std::make_shared<InstructionsBuilder>(id_generator, instruction_list, symbol_list,
                                                     release_object);
      }))
      .def("id_generator", &InstructionsBuilder::id_generator)
      .def("instruction_list", &InstructionsBuilder::instruction_list)
      .def("eager_symbol_list", &InstructionsBuilder::eager_symbol_list)
      .def("object_releaser", &InstructionsBuilder::object_releaser)
      .def("NewObjectId", &NewObjectId)
      .def("NewSymbolId", &NewSymbolId)
      .def("PackPhysicalBlobsToLogicalBlob", &PackPhysicalBlobsToLogicalBlob)
      .def("GetSymbol4String", &GetSymbol4String)
      .def("GetJobConfSymbol", &GetJobConfSymbol)
      .def("GetParallelDescSymbol", &GetParallelDescSymbol)
      .def("GetScopeSymbol", &GetScopeSymbol)
      .def("NewBlobObject", &NewBlobObject)
      .def("NewSymbolId4OpNodeSignature", &NewSymbolId4OpNodeSignature)
      .def("NewSharedOpKernelObjectId4ParallelConfSymbolId",
           &NewSharedOpKernelObjectId4ParallelConfSymbolId)
      .def("GetPhysicalParallelDescSymbols", &GetPhysicalParallelDescSymbols)
      .def("UnpackLogicalBlobToPhysicalBlobs", &UnpackLogicalBlobToPhysicalBlobs)
      .def("MakeReferenceBlobObject", &MakeReferenceBlobObject)
      .def("ReplaceMirrored", &ReplaceMirrored)
      .def("BuildInitialScope", &BuildInitialScope)
      .def("BuildScopeWithNewParallelDesc", &BuildScopeWithNewParallelDesc)
      .def("BuildScopeWithNewParallelConf", &BuildScopeWithNewParallelConf)
      .def("BuildScopeWithNewIsMirrored", &BuildScopeWithNewIsMirrored)
      .def("BuildScopeWithNewScopeName", &BuildScopeWithNewScopeName)
      .def("BuildScopeByProtoSetter", &BuildScopeByProtoSetter)
      .def("BroadcastBlobReference", &BroadcastBlobReference)
      .def("Build121AssignInstruction", &Build121AssignInstruction)
      .def("CudaHostRegisterBlob", &CudaHostRegisterBlob)
      .def("CudaHostUnregisterBlob", &CudaHostUnregisterBlob)
      .def("LazyReference", &LazyReference)
      .def("DeleteObject", &DeleteObject);

  // these API will be removed when InstructionsBuilder is refactor competely
  py::module_ vm_sub_module = m.def_submodule("vm");

  vm_sub_module.def("DelObjectOperand", &DelObjectOperand);
  vm_sub_module.def("MutOperand", &MutOperand);
  vm_sub_module.def("Int64Operand", &Int64Operand);

  vm_sub_module.def("InitSymbolOperand", &InitSymbolOperand);
  vm_sub_module.def("SymbolOperand", &SymbolOperand);
  vm_sub_module.def("ConstOperand", &ConstOperand);

  vm_sub_module.def("OperandSeparator", &OperandSeparator);
  vm_sub_module.def("Uint64Operand", &Uint64Operand);
  vm_sub_module.def("Mut2Operand", &Mut2Operand);
}

}  // namespace oneflow
