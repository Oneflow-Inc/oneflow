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
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"

namespace oneflow {

namespace {

void SetSoleMirroredOperand(vm::cfg::OperandProto* operand, int64_t symbol_id) {
  operand->set_logical_object_id(symbol_id);
  operand->mutable_sole_mirrored_object();
}

void SetInitSymbolOperand(vm::cfg::InstructionOperandProto* instr_operand, int64_t symbol_id) {
  SetSoleMirroredOperand(instr_operand->mutable_init_symbol_operand(), symbol_id);
}

void SetInt64Operand(vm::cfg::InstructionOperandProto* instr_operand, int64_t symbol_id) {
  instr_operand->set_int64_operand(symbol_id);
}

void SetAllMirroredOperand(vm::cfg::OperandProto* operand, int64_t object_id) {
  operand->set_logical_object_id(object_id);
  operand->mutable_all_mirrored_object();
}

void SetMirroredOperand(vm::cfg::OperandProto* operand, int64_t object_id) {
  operand->set_logical_object_id(object_id);
  operand->mutable_current_global_device_id();
}

Maybe<int64_t> NewSymbolId(vm::IdGenerator* id_generator,
                           vm::cfg::InstructionListProto* instruction_list) {
  int64_t symbol_id = JUST(id_generator->NewSymbolId());
  auto* instruction = instruction_list->mutable_instruction()->Add();
  instruction->set_instr_type_name("NewSymbol");
  SetInt64Operand(instruction->mutable_operand()->Add(), symbol_id);
  return symbol_id;
}

template<typename T>
const char* GetInstrTypeName();

template<>
const char* GetInstrTypeName<cfg::JobConfigProto>() {
  return "InitJobDescSymbol";
}
template<>
const char* GetInstrTypeName<cfg::ParallelConf>() {
  return "NewParallelDescSymbol";
}
template<>
const char* GetInstrTypeName<cfg::ScopeProto>() {
  return "InitScopeSymbol";
}

template<typename T>
T* MutEagerSymbolConf(eager::cfg::EagerSymbol*);

template<>
cfg::JobConfigProto* MutEagerSymbolConf<cfg::JobConfigProto>(
    eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_job_conf_symbol();
}

template<>
cfg::ParallelConf* MutEagerSymbolConf<cfg::ParallelConf>(eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_parallel_conf_symbol();
}

template<>
cfg::ScopeProto* MutEagerSymbolConf<cfg::ScopeProto>(eager::cfg::EagerSymbol* eager_symbol) {
  return eager_symbol->mutable_scope_symbol();
}

Maybe<void> AddStringSymbol(int64_t symbol_id, const std::string& data) {
  JUST(Global<symbol::Storage<StringSymbol>>::Get()->Add(symbol_id, data));
  auto* id_cache = JUST(GlobalMaybe<symbol::IdCache<std::string>>());
  CHECK_OR_RETURN(!id_cache->Has(data));
  JUST(id_cache->FindOrCreate(data, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
  return Maybe<void>::Ok();
}

uint64_t NewTokenId() {
  static std::atomic<uint64_t> token_id(0);
  token_id++;
  return token_id;
}

}  // namespace

namespace detail {

template<typename T>
Maybe<int64_t> CreateSymbolIdHelper<T>::Call(vm::IdGenerator* id_generator,
                                             vm::cfg::InstructionListProto* instruction_list,
                                             eager::cfg::EagerSymbolList* eager_symbol_list,
                                             const T& conf) {
  int64_t symbol_id = JUST(NewSymbolId(id_generator, instruction_list));
  {
    auto* instruction = instruction_list->mutable_instruction()->Add();
    instruction->set_instr_type_name(GetInstrTypeName<T>());
    SetInitSymbolOperand(instruction->mutable_operand()->Add(), symbol_id);
  }
  {
    auto* eager_symbol = eager_symbol_list->mutable_eager_symbol()->Add();
    eager_symbol->set_symbol_id(symbol_id);
    MutEagerSymbolConf<T>(eager_symbol)->CopyFrom(conf);
  }
  return symbol_id;
}

template struct CreateSymbolIdHelper<cfg::JobConfigProto>;
template struct CreateSymbolIdHelper<cfg::ScopeProto>;

template<>
Maybe<int64_t> CreateSymbolIdHelper<cfg::ParallelConf>::Call(
    vm::IdGenerator* id_generator, vm::cfg::InstructionListProto* instruction_list,
    eager::cfg::EagerSymbolList* eager_symbol_list, const cfg::ParallelConf& conf) {
  int64_t symbol_id = JUST(id_generator->NewSymbolId());
  {
    auto* instruction = instruction_list->mutable_instruction()->Add();
    instruction->set_instr_type_name(GetInstrTypeName<cfg::ParallelConf>());
    SetInt64Operand(instruction->mutable_operand()->Add(), symbol_id);
  }
  {
    auto* eager_symbol = eager_symbol_list->mutable_eager_symbol()->Add();
    eager_symbol->set_symbol_id(symbol_id);
    MutEagerSymbolConf<cfg::ParallelConf>(eager_symbol)->CopyFrom(conf);
  }
  return symbol_id;
}

}  // namespace detail

Maybe<int64_t> InstructionsBuilder::NewSymbolId() {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewObjectId(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  int64_t object_id = JUST(id_generator_->NewObjectId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewObject");
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(object_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return object_id;
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::PackPhysicalBlobsToLogicalBlob(
    std::vector<std::shared_ptr<compatible_py::BlobObject>> physical_blob_objects,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  std::shared_ptr<ParallelDesc> parallel_desc_symbol = op_arg_parallel_attr->parallel_desc_symbol();
  std::shared_ptr<HashMap<int64_t, std::shared_ptr<std::vector<int64_t>>>> machine_id2device_ids =
      parallel_desc_symbol->machine_id2sorted_dev_phy_ids();
  std::string device_tag = parallel_desc_symbol->parallel_conf().device_tag();
  HashSet<std::pair<int64_t, int64_t>> machine_device_ids;
  for (const auto& physical_blob_object : physical_blob_objects) {
    std::shared_ptr<ParallelDesc> phy_paralle_desc_sym =
        physical_blob_object->parallel_desc_symbol();
    CHECK_EQ(phy_paralle_desc_sym->parallel_num(), 1);
    CHECK_EQ(phy_paralle_desc_sym->device_tag(), device_tag);
    std::shared_ptr<HashMap<int64_t, std::shared_ptr<std::vector<int64_t>>>>
        phy_machine_id2device_ids = phy_paralle_desc_sym->machine_id2sorted_dev_phy_ids();
    int64_t machine_id = phy_machine_id2device_ids->begin()->first;
    machine_device_ids.insert(
        std::make_pair(machine_id, phy_machine_id2device_ids->at(machine_id)->at(0)));
  }
  for (const auto& pair : *machine_id2device_ids) {
    int64_t machine_id = pair.first;
    for (const auto& device_id : *(pair.second)) {
      CHECK(machine_device_ids.find(std::make_pair(machine_id, device_id))
            != machine_device_ids.end());
    }
  }
  std::shared_ptr<compatible_py::BlobObject> logical_blob_object =
      JUST(NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr));
  JUST(ReplaceMirrored(op_arg_parallel_attr->parallel_desc_symbol(), {logical_blob_object},
                       physical_blob_objects));
  return logical_blob_object;
}

Maybe<StringSymbol> InstructionsBuilder::GetSymbol4String(std::string str) {
  if (JUST(HasSymbol<std::string>(str))) { return GetSymbol<std::string, StringSymbol>(str); }
  int64_t symbol_id = JUST(NewSymbolId4String(str));
  JUST(AddStringSymbol(symbol_id, str));
  return GetSymbol<std::string, StringSymbol>(str);
}

Maybe<JobDesc> InstructionsBuilder::GetJobConfSymbol(
    const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  if (JUST(HasSymbol<cfg::JobConfigProto>(*job_conf))) {
    return GetSymbol<cfg::JobConfigProto, JobDesc>(*job_conf);
  }
  int64_t symbol_id = JUST(NewSymbolId4JobConf(job_conf));
  JUST(AddSymbol<cfg::JobConfigProto, JobConfigProto, JobDesc>(symbol_id, *job_conf));
  return GetSymbol<cfg::JobConfigProto, JobDesc>(*job_conf);
}

Maybe<ParallelDesc> InstructionsBuilder::GetParallelDescSymbol(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  if (JUST(HasSymbol<cfg::ParallelConf>(*parallel_conf))) {
    return GetSymbol<cfg::ParallelConf, ParallelDesc>(*parallel_conf);
  }
  int64_t symbol_id = JUST(NewSymbolId4ParallelConf(parallel_conf));
  JUST(AddSymbol<cfg::ParallelConf, ParallelConf, ParallelDesc>(symbol_id, *parallel_conf));
  return GetSymbol<cfg::ParallelConf, ParallelDesc>(*parallel_conf);
}

Maybe<Scope> InstructionsBuilder::GetScopeSymbol(
    const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  if (JUST(HasSymbol<cfg::ScopeProto>(*scope_proto))) {
    return GetSymbol<cfg::ScopeProto, Scope>(*scope_proto);
  }
  int64_t symbol_id = JUST(NewSymbolId4Scope(scope_proto));
  JUST(AddSymbol<cfg::ScopeProto, ScopeProto, Scope>(symbol_id, *scope_proto));
  return GetSymbol<cfg::ScopeProto, Scope>(*scope_proto);
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4String(std::string str) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitStringSymbol(symbol_id, str));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4JobConf(
    const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitJobConfSymbol(symbol_id, job_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4ParallelConf(
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  int64_t symbol_id = JUST(id_generator_->NewSymbolId());
  JUST(NewParallelConfSymbol(symbol_id, parallel_conf));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4Scope(
    const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(NewScopeSymbol(symbol_id, scope_proto));
  return symbol_id;
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::NewBlobObject(
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr,
    const std::shared_ptr<compatible_py::OpArgBlobAttribute>& op_arg_blob_attr) {
  int64_t object_id = JUST(NewObjectId(op_arg_parallel_attr->parallel_desc_symbol()));
  std::shared_ptr<compatible_py::BlobObject> obj = std::make_shared<compatible_py::BlobObject>(
      object_id, op_arg_parallel_attr, op_arg_blob_attr);
  obj->add_releaser(release_object_);
  return obj;
}

Maybe<int64_t> InstructionsBuilder::NewSymbolId4OpNodeSignature(
    const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitOpNodeSignatureDescSymbol(symbol_id, op_node_signature_sym));
  return symbol_id;
}

Maybe<int64_t> InstructionsBuilder::NewSharedOpKernelObjectId4ParallelConfSymbolId(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  return NewObjectId(parallel_desc_sym);
}

Maybe<void> InstructionsBuilder::DeleteObject(compatible_py::BlobObject* blob_object) {
  JUST(_TryClearObject(blob_object));
  JUST(_DeleteObject(blob_object));
  return Maybe<void>::Ok();
}

std::vector<std::shared_ptr<ParallelDesc>> InstructionsBuilder::GetPhysicalParallelDescSymbols(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  const auto& machine_id2device_ids = parallel_desc_symbol->machine_id2sorted_dev_phy_ids();
  std::string device_tag = parallel_desc_symbol->parallel_conf().device_tag();
  std::vector<std::shared_ptr<ParallelDesc>> phy_parallel_desc_symbols;
  const auto AppendPhyParallelDescSymbol = [this, &phy_parallel_desc_symbols, &device_tag](
                                               int64_t machine_id, int64_t device_id) {
    std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
    parallel_conf->set_device_tag(device_tag);
    parallel_conf->add_device_name(std::to_string(machine_id) + ":" + std::to_string(device_id));
    phy_parallel_desc_symbols.emplace_back(CHECK_JUST(GetParallelDescSymbol(parallel_conf)));
  };

  for (const auto& pair : *machine_id2device_ids) {
    for (int64_t device_id : *pair.second) { AppendPhyParallelDescSymbol(pair.first, device_id); }
  }
  return phy_parallel_desc_symbols;
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::MakeReferenceBlobObject(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr) {
  std::shared_ptr<ParallelDesc> parallel_desc_symbol = blob_object->parallel_desc_symbol();
  CHECK((*parallel_desc_symbol) == (*op_arg_parallel_attr->parallel_desc_symbol()));
  std::shared_ptr<compatible_py::BlobObject> ref_blob_object =
      JUST(NewBlobObject(op_arg_parallel_attr, blob_object->op_arg_blob_attr()));
  ReplaceMirrored(parallel_desc_symbol, {ref_blob_object}, {blob_object});
  return ref_blob_object;
}

Maybe<void> InstructionsBuilder::ReplaceMirrored(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
    std::vector<std::shared_ptr<compatible_py::BlobObject>> lhs_objects,
    std::vector<std::shared_ptr<compatible_py::BlobObject>> rhs_objects) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("ReplaceMirrored");
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_sym->symbol_id()));
  for (const auto& lhs_object : lhs_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(lhs_object->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());
  for (const auto& rhs_object : rhs_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(rhs_object->object_id()));
  }
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewIsMirrored(const std::shared_ptr<Scope>& scope,
                                                              bool is_mirrored) {
  const auto SetScopeProto = [is_mirrored](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
    if (is_mirrored) {
      scope_proto->mutable_opt_mirrored_parallel_conf()->mutable_mirrored_parallel();
    } else {
      scope_proto->mutable_opt_mirrored_parallel_conf()->clear_mirrored_parallel();
    }
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewScopeName(const std::shared_ptr<Scope>& scope,
                                                             std::string scope_name) {
  const auto SetScopeProto = [&scope_name](const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
    scope_proto->add_scope_op_name_prefixes(scope_name);
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeByProtoSetter(
    const std::shared_ptr<Scope>& scope,
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& setter) {
  std::shared_ptr<cfg::ScopeProto> scope_proto = JUST(scope->MakeChildScopeProto());
  setter(scope_proto);
  return GetScopeSymbol(scope_proto);
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::BroadcastBlobReference(
    const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  std::shared_ptr<HashMap<int64_t, std::shared_ptr<std::vector<int64_t>>>> device_ids =
      sole_mirrored_blob_object->parallel_desc_symbol()->machine_id2sorted_dev_phy_ids();
  for (const auto& pair : *device_ids) { CHECK_EQ(pair.second->size(), 1); }
  int64_t object_id = JUST(BroadcastObjectReference(sole_mirrored_blob_object, parallel_desc_sym));
  std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
      JUST(compatible_py::MakeBroadcastOpArgParallelAttribute(parallel_desc_sym));
  std::shared_ptr<compatible_py::BlobObject> obj = std::make_shared<compatible_py::BlobObject>(
      object_id, op_arg_parallel_attr, sole_mirrored_blob_object->op_arg_blob_attr());
  obj->add_releaser(release_object_);
  return obj;
}

Maybe<int64_t> InstructionsBuilder::BroadcastObjectReference(
    const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  int64_t object_id = JUST(id_generator_->NewObjectId());
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("BroadcastObjectReference");
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(object_id));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(sole_mirrored_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return object_id;
}

Maybe<void> InstructionsBuilder::Build121AssignInstruction(
    const std::shared_ptr<compatible_py::BlobObject>& ref_blob_object,
    const std::shared_ptr<compatible_py::BlobObject>& value_blob_object) {
  int64_t parallel_num = ref_blob_object->parallel_desc_symbol()->parallel_num();
  CHECK_EQ(parallel_num, value_blob_object->parallel_desc_symbol()->parallel_num());
  std::vector<uint64_t> token_id_0;
  std::vector<uint64_t> token_id_1;
  for (int64_t i = 0; i < parallel_num; ++i) { token_id_0.emplace_back(NewTokenId()); }
  for (int64_t i = 0; i < parallel_num; ++i) { token_id_1.emplace_back(NewTokenId()); }
  std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> token_ids =
      std::make_tuple(token_id_0, token_id_1);
  BuildSendInstruction(ref_blob_object->parallel_desc_symbol(), value_blob_object, token_ids);
  BuildRecvInstruction(value_blob_object->parallel_desc_symbol(), ref_blob_object, token_ids);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::BuildSendInstruction(
    const std::shared_ptr<ParallelDesc>& dst_parallel_desc_symbol,
    const std::shared_ptr<compatible_py::BlobObject>& src_blob_object,
    const std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>& token_ids) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("SendBlob");
  instruction.set_parallel_desc_symbol_id(
      JUST(src_blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(
      *SymbolOperand(JUST(dst_parallel_desc_symbol->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*ConstOperand(src_blob_object->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());
  for (uint64_t token_id : std::get<0>(token_ids)) {
    instruction.mutable_operand()->Add()->CopyFrom(*Uint64Operand(token_id));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());
  for (uint64_t token_id : std::get<1>(token_ids)) {
    instruction.mutable_operand()->Add()->CopyFrom(*Uint64Operand(token_id));
  }
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::BuildRecvInstruction(
    const std::shared_ptr<ParallelDesc>& src_parallel_desc_symbol,
    const std::shared_ptr<compatible_py::BlobObject>& dst_blob_object,
    const std::tuple<std::vector<uint64_t>, std::vector<uint64_t>>& token_ids) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("ReceiveBlob");
  instruction.set_parallel_desc_symbol_id(
      JUST(dst_blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(
      *SymbolOperand(JUST(src_parallel_desc_symbol->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*Mut2Operand(dst_blob_object->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());
  for (uint64_t token_id : std::get<0>(token_ids)) {
    instruction.mutable_operand()->Add()->CopyFrom(*Uint64Operand(token_id));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());
  for (uint64_t token_id : std::get<1>(token_ids)) {
    instruction.mutable_operand()->Add()->CopyFrom(*Uint64Operand(token_id));
  }
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::CudaHostRegisterBlob(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("CudaHostRegisterBlob");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::CudaHostUnregisterBlob(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("CudaHostUnregisterBlob");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::LazyReference(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object, std::string interface_op_name) {
  vm::cfg::InstructionProto instruction;
  std::string device_tag = blob_object->parallel_desc_symbol()->device_tag();
  instruction.set_instr_type_name(device_tag + ".LazyReference");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(blob_object->object_id()));
  std::shared_ptr<StringSymbol> interface_op_name_sym =
      JUST(GetSymbol4String(blob_object->op_arg_blob_attr()->logical_blob_name()));
  instruction.mutable_operand()->Add()->CopyFrom(
      *SymbolOperand(JUST(interface_op_name_sym->symbol_id())));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InitStringSymbol(int64_t symbol_id, std::string str) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitStringSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.set_string_symbol(str);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InitJobConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::JobConfigProto>& job_conf) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitJobDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_job_conf_symbol()->CopyFrom(*job_conf);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NewParallelConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("NewParallelDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_parallel_conf_symbol()->CopyFrom(*parallel_conf);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NewScopeSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::ScopeProto>& scope_proto) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitScopeSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_scope_symbol()->CopyFrom(*scope_proto);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InitOpNodeSignatureDescSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::OpNodeSignature>& op_node_signature_sym) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitOpNodeSignatureDescSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_op_node_signature_symbol()->CopyFrom(*op_node_signature_sym);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_TryClearObject(compatible_py::BlobObject* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("TryClearObject");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_DeleteObject(compatible_py::BlobObject* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("DeleteObject");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*DelObjectOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

std::shared_ptr<vm::cfg::InstructionOperandProto> DelObjectOperand(int64_t object_id) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetAllMirroredOperand(operand->mutable_mut_operand(), object_id);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> MutOperand(int64_t object_id) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_mut_operand(), object_id);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Int64Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->set_int64_operand(val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> InitSymbolOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetSoleMirroredOperand(operand->mutable_init_symbol_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> SymbolOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetSoleMirroredOperand(operand->mutable_symbol_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> ConstOperand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_const_operand(), val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> OperandSeparator() {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->mutable_separator();
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Uint64Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  operand->set_uint64_operand(val);
  return operand;
}

std::shared_ptr<vm::cfg::InstructionOperandProto> Mut2Operand(int64_t val) {
  std::shared_ptr<vm::cfg::InstructionOperandProto> operand =
      std::make_shared<vm::cfg::InstructionOperandProto>();
  SetMirroredOperand(operand->mutable_mut2_operand(), val);
  return operand;
}

}  // namespace oneflow
