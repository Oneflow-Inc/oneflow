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
#ifndef ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_

#include "oneflow/core/vm/instruction.cfg.h"
#include "oneflow/core/vm/id_generator.h"
#include "oneflow/core/vm/string_symbol.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/job/scope.pb.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/object.h"

namespace oneflow {

namespace detail {

template<typename T>
struct CreateSymbolIdHelper {
  static Maybe<int64_t> Call(vm::IdGenerator* id_generator,
                             vm::cfg::InstructionListProto* instruction_list,
                             eager::cfg::EagerSymbolList* eager_symbol_list, const T& conf);
};

}  // namespace detail

class InstructionsBuilder {
 public:
  InstructionsBuilder(const InstructionsBuilder&) = delete;
  InstructionsBuilder(InstructionsBuilder&&) = delete;
  explicit InstructionsBuilder(const std::shared_ptr<vm::IdGenerator>& id_generator)
      : id_generator_(id_generator),
        instruction_list_(std::make_shared<vm::cfg::InstructionListProto>()),
        eager_symbol_list_(std::make_shared<eager::cfg::EagerSymbolList>()),
        release_object_([](compatible_py::BlobObject*) {}) {}
  InstructionsBuilder(const std::shared_ptr<vm::IdGenerator>& id_generator,
                      const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
                      const std::shared_ptr<eager::cfg::EagerSymbolList>& symbol_list,
                      const std::function<void(compatible_py::BlobObject*)>& release_object)
      : id_generator_(id_generator),
        instruction_list_(instruction_list),
        eager_symbol_list_(symbol_list),
        release_object_(release_object) {}
  ~InstructionsBuilder() = default;

  std::shared_ptr<vm::IdGenerator> id_generator() const { return id_generator_; }
  std::shared_ptr<vm::cfg::InstructionListProto> instruction_list() const {
    return instruction_list_;
  }
  std::shared_ptr<eager::cfg::EagerSymbolList> eager_symbol_list() const {
    return eager_symbol_list_;
  }

  const std::function<void(compatible_py::BlobObject*)>& object_releaser() const {
    return release_object_;
  }

  Maybe<int64_t> NewSymbolId();

  Maybe<compatible_py::BlobObject> PackPhysicalBlobsToLogicalBlob(
      std::vector<std::shared_ptr<compatible_py::BlobObject>> physical_blob_objects,
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
      const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr);

  Maybe<int64_t> NewObjectId(const std::shared_ptr<ParallelDesc>& parallel_desc_sym);

  Maybe<StringSymbol> GetSymbol4String(std::string str);

  Maybe<JobDesc> GetJobConfSymbol(const std::shared_ptr<cfg::JobConfigProto>& job_conf);

  Maybe<ParallelDesc> GetParallelDescSymbol(
      const std::shared_ptr<cfg::ParallelConf>& parallel_conf);

  Maybe<Scope> GetScopeSymbol(const std::shared_ptr<cfg::ScopeProto>& scope_proto);

  Maybe<compatible_py::BlobObject> NewBlobObject(
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
      const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr);

  Maybe<int64_t> NewSymbolId4OpNodeSignature(
      const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym);

  Maybe<int64_t> NewSharedOpKernelObjectId4ParallelConfSymbolId(
      const std::shared_ptr<ParallelDesc>& parallel_desc_sym);

  Maybe<void> DeleteObject(compatible_py::BlobObject* blob_object);

  Maybe<std::vector<std::shared_ptr<ParallelDesc>>> GetPhysicalParallelDescSymbols(
      const std::shared_ptr<ParallelDesc>& parallel_desc_symbol);

  Maybe<std::vector<std::shared_ptr<compatible_py::BlobObject>>> UnpackLogicalBlobToPhysicalBlobs(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object);

  Maybe<compatible_py::BlobObject> MakeReferenceBlobObject(
      const std::shared_ptr<compatible_py::BlobObject>& blob_object,
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr);

  Maybe<void> ReplaceMirrored(const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
                              std::vector<std::shared_ptr<compatible_py::BlobObject>> lhs_objects,
                              std::vector<std::shared_ptr<compatible_py::BlobObject>> rhs_objects);

  Maybe<Scope> BuildInitialScope(int64_t session_id,
                                 const std::shared_ptr<cfg::JobConfigProto>& job_conf,
                                 const std::string& device_tag,
                                 const std::vector<std::string>& machine_device_ids,
                                 bool is_mirrored);

  Maybe<Scope> BuildScopeWithNewParallelDesc(const std::shared_ptr<Scope>& scope,
                                             const std::string& device_tag,
                                             const std::vector<std::string>& machine_device_ids);

  Maybe<Scope> BuildScopeWithNewParallelConf(
      const std::shared_ptr<Scope>& scope, const std::shared_ptr<cfg::ParallelConf>& parallel_conf);

  Maybe<Scope> BuildScopeWithNewIsMirrored(const std::shared_ptr<Scope>& scope, bool is_mirrored);

  Maybe<Scope> BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                          std::string scope_name);

  Maybe<Scope> BuildScopeByProtoSetter(
      const std::shared_ptr<Scope>& scope,
      const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& setter);

  Maybe<compatible_py::BlobObject> BroadcastBlobReference(
      const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_blob_object,
      const std::shared_ptr<ParallelDesc>& parallel_desc_sym);

  Maybe<void> Build121AssignInstruction(
      const std::shared_ptr<compatible_py::BlobObject>& ref_blob_object,
      const std::shared_ptr<compatible_py::BlobObject>& value_blob_object);

  Maybe<void> CudaHostRegisterBlob(const std::shared_ptr<compatible_py::BlobObject>& blob_object);

  Maybe<void> CudaHostUnregisterBlob(const std::shared_ptr<compatible_py::BlobObject>& blob_object);

  Maybe<void> LazyReference(const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                            std::string interface_op_name);

  template<typename T>
  Maybe<int64_t> FindOrCreateSymbolId(const T& conf) {
    auto* id_cache = Global<symbol::IdCache<T>>::Get();
    return id_cache->FindOrCreate(conf, [&] { return CreateSymbolId<T>(conf); });
  }

 private:
  Maybe<std::vector<std::shared_ptr<compatible_py::OpArgBlobAttribute>>> GetPhysicalOpArgBlobAttrs(
      const std::shared_ptr<compatible_py::BlobObject>& logical_blob_object) const;

  Maybe<int64_t> NewSymbolId4String(std::string str);

  Maybe<int64_t> NewSymbolId4JobConf(const std::shared_ptr<cfg::JobConfigProto>& job_conf);

  Maybe<int64_t> NewSymbolId4ParallelConf(const std::shared_ptr<cfg::ParallelConf>& parallel_conf);

  Maybe<int64_t> NewSymbolId4Scope(const std::shared_ptr<cfg::ScopeProto>& scope_proto);

  Maybe<void> InitStringSymbol(int64_t symbol_id, std::string str);

  Maybe<void> InitJobConfSymbol(int64_t symbol_id,
                                const std::shared_ptr<cfg::JobConfigProto>& job_conf);

  Maybe<void> NewParallelConfSymbol(int64_t symbol_id,
                                    const std::shared_ptr<cfg::ParallelConf>& parallel_conf);

  Maybe<void> NewScopeSymbol(int64_t symbol_id,
                             const std::shared_ptr<cfg::ScopeProto>& scope_proto);

  Maybe<void> InitOpNodeSignatureDescSymbol(
      int64_t symbol_id, const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym);

  Maybe<int64_t> BroadcastObjectReference(
      const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_object,
      const std::shared_ptr<ParallelDesc>& parallel_desc_sym);

  Maybe<void> BuildSendInstruction(
      const std::shared_ptr<ParallelDesc>& dst_parallel_desc_symbol,
      const std::shared_ptr<compatible_py::BlobObject>& src_blob_object,
      const std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>& token_ids);

  Maybe<void> BuildRecvInstruction(
      const std::shared_ptr<ParallelDesc>& src_parallel_desc_symbol,
      const std::shared_ptr<compatible_py::BlobObject>& dst_blob_object,
      const std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>& token_ids);

  Maybe<void> _TryClearObject(compatible_py::BlobObject* blob_object);

  Maybe<void> _DeleteObject(compatible_py::BlobObject* blob_object);

  template<typename T>
  Maybe<int64_t> CreateSymbolId(const T& conf) {
    return detail::CreateSymbolIdHelper<T>::Call(mut_id_generator(), mut_instruction_list(),
                                                 mut_eager_symbol_list(), conf);
  }

  vm::cfg::InstructionListProto* mut_instruction_list() { return instruction_list_.get(); }
  eager::cfg::EagerSymbolList* mut_eager_symbol_list() { return eager_symbol_list_.get(); }

  vm::IdGenerator* mut_id_generator() { return id_generator_.get(); }

  std::shared_ptr<vm::IdGenerator> id_generator_;
  std::shared_ptr<vm::cfg::InstructionListProto> instruction_list_;
  std::shared_ptr<eager::cfg::EagerSymbolList> eager_symbol_list_;
  std::function<void(compatible_py::BlobObject*)> release_object_;
};

std::shared_ptr<vm::cfg::InstructionOperandProto> DelObjectOperand(int64_t object_id);

std::shared_ptr<vm::cfg::InstructionOperandProto> MutOperand(int64_t object_id);

std::shared_ptr<vm::cfg::InstructionOperandProto> Int64Operand(int64_t val);

std::shared_ptr<vm::cfg::InstructionOperandProto> InitSymbolOperand(int64_t val);

std::shared_ptr<vm::cfg::InstructionOperandProto> SymbolOperand(int64_t val);

std::shared_ptr<vm::cfg::InstructionOperandProto> ConstOperand(int64_t val);

std::shared_ptr<vm::cfg::InstructionOperandProto> OperandSeparator();

std::shared_ptr<vm::cfg::InstructionOperandProto> Uint64Operand(int64_t val);

std::shared_ptr<vm::cfg::InstructionOperandProto> Mut2Operand(int64_t val);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_INSTRUCTIONS_BUILDER_H_
