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
#include <atomic>
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/placement.cfg.h"
#include "oneflow/core/job/scope.cfg.h"
#include "oneflow/core/framework/parallel_conf_util.h"
#include "oneflow/core/framework/object_storage.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/operator/interface_blob_conf.cfg.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/framework/blob_cache.h"
#include "oneflow/core/common/container_util.h"

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

using IntList = std::vector<int64_t>;
using Int2IntListMap = HashMap<int64_t, std::shared_ptr<IntList>>;
// This function is used to determine whether the machine_id2sorted_dev_phy_ids of ParallelDesc are
// equal
bool Int2IntListMapContaining(const Int2IntListMap& bigger, const Int2IntListMap& smaller) {
  for (const auto& pair : smaller) {
    if (bigger.find(pair.first) == bigger.end()) { return false; }
    const auto& bigger_device_ids = bigger.find(pair.first)->second;
    std::vector<int64_t>::iterator ret;
    for (int64_t device_id : *pair.second) {
      ret = std::find(bigger_device_ids->begin(), bigger_device_ids->end(), device_id);
      if (ret == bigger_device_ids->end()) { return false; }
    }
  }
  return true;
}

Maybe<compatible_py::BlobObject> MakeNewBlobObjectLike(
    const std::shared_ptr<InstructionsBuilder>& builder,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<ParallelDesc>& new_parallel_desc_symbol) {
  OperatorConf op_conf;
  op_conf.set_name(*JUST(UniqueStr("Input")));
  op_conf.set_device_tag(new_parallel_desc_symbol->device_tag());
  op_conf.mutable_input_conf()->set_out("out");
  std::shared_ptr<cfg::InterfaceBlobConf> cfg_interface_blob_conf =
      std::make_shared<cfg::InterfaceBlobConf>();
  blob_object->op_arg_parallel_attr()->DumpToInterfaceBlobConf(cfg_interface_blob_conf);
  blob_object->op_arg_blob_attr()->DumpToInterfaceBlobConf(cfg_interface_blob_conf);
  cfg_interface_blob_conf->ToProto(op_conf.mutable_input_conf()->mutable_blob_conf());
  std::shared_ptr<Scope> cur_scope = JUST(GetCurrentScope());
  op_conf.set_scope_symbol_id(JUST(cur_scope->symbol_id()));
  OpNodeSignature upstream_signature;
  const auto& op = JUST(ConstructAndInferOp(op_conf, upstream_signature, *cur_scope));
  const auto& op_attribute = op->GetOpAttributeWithoutOpNameAndLbn();
  std::shared_ptr<cfg::ParallelConf> parallel_conf = new_parallel_desc_symbol->cfg_parallel_conf();
  std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>
      bn_in_op2blob_object =
          std::make_shared<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>();
  builder->RawStatelessCall(std::make_shared<cfg::OpAttribute>(*op_attribute), parallel_conf,
                            bn_in_op2blob_object);
  return JUST(MapAt(*bn_in_op2blob_object, "out"));
}

Maybe<void> _Run(
    const std::function<void(const std::shared_ptr<InstructionsBuilder>&)>& Build,
    const std::shared_ptr<vm::IdGenerator>& id_generator,
    const std::function<Maybe<void>(const std::shared_ptr<vm::cfg::InstructionListProto>&,
                                    const std::shared_ptr<eager::cfg::EagerSymbolList>&)>&
        RunInstruction,
    const std::function<Maybe<void>(compatible_py::Object*)>& ReleaseObject) {
  std::shared_ptr<Session> sess = JUST(GetDefaultSession());
  std::shared_ptr<vm::cfg::InstructionListProto> instruction_list = sess->instruction_list();
  std::shared_ptr<eager::cfg::EagerSymbolList> eager_symbol_list = sess->eager_symbol_list();
  Build(std::make_shared<InstructionsBuilder>(id_generator, instruction_list, eager_symbol_list,
                                              ReleaseObject));
  JUST(RunInstruction(instruction_list, eager_symbol_list));
  instruction_list->clear_instruction();
  eager_symbol_list->clear_eager_symbol();
  return Maybe<void>::Ok();
}

Maybe<void> _ReleaseLogicalObject(compatible_py::Object* obj) {
  JUST(LogicalRun(
      [&obj](const std::shared_ptr<InstructionsBuilder>& build) { build->DeleteObject(obj); }));
  return Maybe<void>::Ok();
}

Maybe<void> _ReleasePhysicalObject(compatible_py::Object* obj) {
  JUST(PhysicalRun(
      [&obj](const std::shared_ptr<InstructionsBuilder>& build) { build->DeleteObject(obj); }));
  return Maybe<void>::Ok();
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
    const std::vector<std::shared_ptr<compatible_py::BlobObject>>& physical_blob_objects,
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
    CHECK_EQ_OR_RETURN(phy_paralle_desc_sym->parallel_num(), 1);
    CHECK_EQ_OR_RETURN(phy_paralle_desc_sym->device_tag(), device_tag);
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

Maybe<OperatorConfSymbol> InstructionsBuilder::GetOpConfSymbol(
    const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  if (JUST(HasSymbol<cfg::OperatorConf>(*op_conf))) {
    return GetSymbol<cfg::OperatorConf, OperatorConfSymbol>(*op_conf);
  }
  int64_t symbol_id = JUST(NewSymbolId4OpConf(op_conf));
  JUST(AddSymbol<cfg::OperatorConf, OperatorConf, OperatorConfSymbol>(symbol_id, *op_conf));
  return GetSymbol<cfg::OperatorConf, OperatorConfSymbol>(*op_conf);
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

Maybe<int64_t> InstructionsBuilder::NewSymbolId4OpConf(
    const std::shared_ptr<cfg::OperatorConf> op_conf) {
  int64_t symbol_id = JUST(NewSymbolId());
  JUST(InitOpConfSymbol(symbol_id, op_conf));
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

Maybe<void> InstructionsBuilder::DeleteObject(compatible_py::Object* blob_object) {
  JUST(_TryClearObject(blob_object));
  JUST(_DeleteObject(blob_object));
  return Maybe<void>::Ok();
}

Maybe<std::vector<std::shared_ptr<ParallelDesc>>>
InstructionsBuilder::GetPhysicalParallelDescSymbols(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::string device_tag = parallel_desc_symbol->parallel_conf().device_tag();
  std::vector<std::shared_ptr<ParallelDesc>> phy_parallel_desc_symbols;
  const auto AppendPhyParallelDescSymbol = [this, &phy_parallel_desc_symbols, &device_tag](
                                               int64_t machine_id,
                                               int64_t device_id) -> Maybe<void> {
    std::shared_ptr<cfg::ParallelConf> parallel_conf = std::make_shared<cfg::ParallelConf>();
    parallel_conf->set_device_tag(device_tag);
    parallel_conf->add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                   + std::to_string(device_id));
    phy_parallel_desc_symbols.emplace_back(JUST(GetParallelDescSymbol(parallel_conf)));
    return Maybe<void>::Ok();
  };

  for (const int64_t machine_id : parallel_desc_symbol->sorted_machine_ids()) {
    for (const int64_t device_id : parallel_desc_symbol->sorted_dev_phy_ids(machine_id)) {
      JUST(AppendPhyParallelDescSymbol(machine_id, device_id));
    }
  }

  return phy_parallel_desc_symbols;
}

Maybe<std::vector<std::shared_ptr<compatible_py::OpArgBlobAttribute>>>
InstructionsBuilder::GetPhysicalOpArgBlobAttrs(
    const std::shared_ptr<compatible_py::BlobObject>& logical_blob_object) const {
  int64_t parallel_num = logical_blob_object->parallel_desc_symbol()->parallel_num();
  std::shared_ptr<compatible_py::OpArgBlobAttribute> logical_blob_attr =
      logical_blob_object->op_arg_blob_attr();
  std::shared_ptr<cfg::SbpParallel> sbp_parallel =
      logical_blob_object->op_arg_parallel_attr()->sbp_parallel();
  std::vector<std::shared_ptr<compatible_py::OpArgBlobAttribute>> pyh_op_arg_blob_attrs;
  if (sbp_parallel->has_split_parallel()) {
    int64_t split_axis = sbp_parallel->split_parallel().axis();
    for (int64_t i = 0; i < parallel_num; ++i) {
      pyh_op_arg_blob_attrs.emplace_back(
          logical_blob_attr->GetPhysicalOpArgBlobAttr(split_axis, parallel_num, i));
    }
  } else {
    for (int64_t i = 0; i < parallel_num; ++i) {
      pyh_op_arg_blob_attrs.emplace_back(logical_blob_attr);
    }
  }
  return pyh_op_arg_blob_attrs;
}

Maybe<std::vector<std::shared_ptr<compatible_py::BlobObject>>>
InstructionsBuilder::UnpackLogicalBlobToPhysicalBlobs(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  std::vector<std::shared_ptr<ParallelDesc>> phy_parallel_desc_symbols =
      *JUST(GetPhysicalParallelDescSymbols(blob_object->parallel_desc_symbol()));
  auto phy_op_arg_blob_attrs = JUST(GetPhysicalOpArgBlobAttrs(blob_object));
  const auto GetPhysicalBlob =
      [this](const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
             const std::shared_ptr<compatible_py::OpArgBlobAttribute>& blob_attr)
      -> Maybe<compatible_py::BlobObject> {
    std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
        JUST(compatible_py::MakeMirroredOpArgParallelAttribute(parallel_desc_sym));
    std::shared_ptr<compatible_py::BlobObject> pyhsical_blob_object =
        JUST(NewBlobObject(op_arg_parallel_attr, blob_attr));
    return pyhsical_blob_object;
  };
  std::vector<std::shared_ptr<compatible_py::BlobObject>> physical_blob_objects;
  for (int64_t i = 0; i < phy_parallel_desc_symbols.size(); ++i) {
    physical_blob_objects.emplace_back(JUST(GetPhysicalBlob(
        JUST(VectorAt(phy_parallel_desc_symbols, i)), JUST(VectorAt(*phy_op_arg_blob_attrs, i)))));
  }
  JUST(ReplaceMirrored(blob_object->parallel_desc_symbol(), physical_blob_objects, {blob_object}));
  return physical_blob_objects;
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
    const std::vector<std::shared_ptr<compatible_py::BlobObject>>& lhs_objects,
    const std::vector<std::shared_ptr<compatible_py::BlobObject>>& rhs_objects) {
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

Maybe<Scope> InstructionsBuilder::BuildInitialScope(
    int64_t session_id, const std::shared_ptr<cfg::JobConfigProto>& job_conf,
    const std::string& device_tag, const std::vector<std::string>& machine_device_ids,
    const std::shared_ptr<Shape>& hierarchy, bool is_mirrored) {
  std::shared_ptr<cfg::ScopeProto> scope_proto = std::make_shared<cfg::ScopeProto>();
  scope_proto->set_session_id(session_id);
  std::shared_ptr<JobDesc> job_conf_sym = JUST(GetJobConfSymbol(job_conf));
  scope_proto->set_job_desc_symbol_id(JUST(job_conf_sym->symbol_id()));
  std::shared_ptr<cfg::ParallelConf> parallel_conf =
      JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
      JUST(GetParallelDescSymbol(parallel_conf));
  scope_proto->set_device_parallel_desc_symbol_id(JUST(device_parallel_desc_sym->symbol_id()));
  parallel_conf = JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
  std::shared_ptr<ParallelDesc> host_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  scope_proto->set_host_parallel_desc_symbol_id(JUST(host_parallel_desc_sym->symbol_id()));
  if (is_mirrored) {
    scope_proto->mutable_opt_mirrored_parallel_conf()->mutable_mirrored_parallel();
  } else {
    scope_proto->mutable_opt_mirrored_parallel_conf()->clear_mirrored_parallel();
  }
  return GetScopeSymbol(scope_proto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelDesc(
    const std::shared_ptr<Scope>& scope, const std::string& device_tag,
    const std::vector<std::string>& machine_device_ids, const std::shared_ptr<Shape>& hierarchy) {
  const auto SetScopeProto =
      [this, &device_tag, &machine_device_ids,
       &hierarchy](const std::shared_ptr<cfg::ScopeProto>& scope_proto) -> Maybe<void> {
    std::shared_ptr<cfg::ParallelConf> parallel_conf =
        JUST(MakeParallelConf(device_tag, machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> device_parallel_desc_sym =
        JUST(GetParallelDescSymbol(parallel_conf));
    parallel_conf = JUST(MakeParallelConf("cpu", machine_device_ids, hierarchy));
    std::shared_ptr<ParallelDesc> host_parallel_desc_sym =
        JUST(GetParallelDescSymbol(parallel_conf));
    scope_proto->set_device_parallel_desc_symbol_id(JUST(device_parallel_desc_sym->symbol_id()));
    scope_proto->set_host_parallel_desc_symbol_id(JUST(host_parallel_desc_sym->symbol_id()));
    return Maybe<void>::Ok();
  };

  return BuildScopeByProtoSetter(scope, SetScopeProto);
}

Maybe<Scope> InstructionsBuilder::BuildScopeWithNewParallelConf(
    const std::shared_ptr<Scope>& scope, const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  const std::shared_ptr<
      std::tuple<std::string, std::vector<std::string>, std::shared_ptr<cfg::ShapeProto>>>&
      tag_and_dev_ids_and_hierarchy =
          JUST(GetDeviceTagAndMachineDeviceIdsAndHierarchy(parallel_conf));
  std::shared_ptr<Shape> hierarchy;
  if (std::get<2>(*tag_and_dev_ids_and_hierarchy)) {
    ShapeProto hierarchy_proto;
    parallel_conf->hierarchy().ToProto(&hierarchy_proto);
    hierarchy.reset(new Shape(hierarchy_proto));
  }
  return BuildScopeWithNewParallelDesc(scope, std::get<0>(*tag_and_dev_ids_and_hierarchy),
                                       std::get<1>(*tag_and_dev_ids_and_hierarchy), hierarchy);
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
    const std::function<void(const std::shared_ptr<cfg::ScopeProto>&)>& Setter) {
  std::shared_ptr<cfg::ScopeProto> scope_proto = JUST(scope->MakeChildScopeProto());
  Setter(scope_proto);
  return GetScopeSymbol(scope_proto);
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::BroadcastBlobReference(
    const std::shared_ptr<compatible_py::BlobObject>& sole_mirrored_blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  std::shared_ptr<HashMap<int64_t, std::shared_ptr<std::vector<int64_t>>>> device_ids =
      sole_mirrored_blob_object->parallel_desc_symbol()->machine_id2sorted_dev_phy_ids();
  for (const auto& pair : *device_ids) { CHECK_EQ_OR_RETURN(pair.second->size(), 1); }
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
  CHECK_EQ_OR_RETURN(parallel_num, value_blob_object->parallel_desc_symbol()->parallel_num());
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

Maybe<compatible_py::OpKernelObject> InstructionsBuilder::NewOpKernelObject(
    const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  CHECK_OR_RETURN(op_conf->has_scope_symbol_id());
  std::shared_ptr<Scope> scope_symbol =
      JUST(GetSymbol<cfg::ScopeProto, Scope>(op_conf->scope_symbol_id()));
  std::shared_ptr<OperatorConfSymbol> op_conf_sym = JUST(GetOpConfSymbol(op_conf));
  const auto& scope = Global<symbol::Storage<Scope>>::Get()->Get(op_conf->scope_symbol_id());
  OperatorConf pb_op_conf;
  op_conf->ToProto(&pb_op_conf);
  int64_t parallel_desc_sym_id = JUST(scope.GetParallelDescSymbolId(pb_op_conf));
  std::shared_ptr<ParallelDesc> parallel_desc_symbol =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));
  int64_t object_id =
      JUST(_NewOpKernelObject(parallel_desc_symbol, scope_symbol->job_desc_symbol(), op_conf_sym));
  return std::make_shared<compatible_py::OpKernelObject>(object_id, op_conf, release_object_);
}

Maybe<void> InstructionsBuilder::LazyReference(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::string& interface_op_name) {
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

Maybe<compatible_py::BlobObject> InstructionsBuilder::MakeLazyRefBlobObject(
    const std::string& interface_op_name, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf) {
  CHECK_EQ_OR_RETURN(op_attribute->output_bns().size(), 1);
  const std::string& obn = op_attribute->output_bns().at(0);
  std::shared_ptr<ParallelDesc> blob_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  OpAttribute pb_op_attribute;
  op_attribute->ToProto(&pb_op_attribute);
  std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, pb_op_attribute, obn));
  std::shared_ptr<compatible_py::OpArgBlobAttribute> op_arg_blob_attr =
      JUST(compatible_py::GetOpArgBlobAttribute(pb_op_attribute, obn));
  std::shared_ptr<compatible_py::BlobObject> blob_object =
      JUST(NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr));
  JUST(LazyReference(blob_object, interface_op_name));
  return blob_object;
}

Maybe<compatible_py::Object> InstructionsBuilder::GetSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym) {
  if (JUST(HasSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym))) {
    return GetOpKernelObject4ParallelConfSymbol(parallel_desc_sym);
  }
  int64_t object_id = JUST(NewSharedOpKernelObjectId4ParallelConfSymbolId(parallel_desc_sym));
  std::shared_ptr<compatible_py::Object> obj =
      std::make_shared<compatible_py::Object>(object_id, parallel_desc_sym);
  JUST(SetSharedOpKernelObject4ParallelConfSymbol(parallel_desc_sym, obj));
  return obj;
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

Maybe<int64_t> InstructionsBuilder::_NewOpKernelObject(
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol,
    const std::shared_ptr<JobDesc>& job_desc_sym,
    const std::shared_ptr<OperatorConfSymbol>& op_conf_sym) {
  int64_t object_id = JUST(NewObjectId(parallel_desc_symbol));
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitOpKernelObject");
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_symbol->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(job_desc_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(op_conf_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(object_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return object_id;
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

Maybe<void> InstructionsBuilder::InitOpConfSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::OperatorConf>& op_conf) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("InitOperatorConfSymbol");
  instruction.mutable_operand()->Add()->CopyFrom(*InitSymbolOperand(symbol_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  eager::cfg::EagerSymbol eager_symbol;
  eager_symbol.set_symbol_id(symbol_id);
  eager_symbol.mutable_op_conf_symbol()->CopyFrom(*op_conf);
  eager_symbol_list_->mutable_eager_symbol()->Add()->CopyFrom(eager_symbol);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::InsertRemoveForeignCallbackInstruction(int64_t object_id,
                                                                        int64_t callback_id) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("RemoveForeignCallback");
  instruction.mutable_operand()->Add()->CopyFrom(*DelObjectOperand(object_id));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(callback_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_FetchBlob(
    const std::string& instruction_name,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object, int64_t callback_id) {
  vm::cfg::InstructionProto instruction;
  const std::string& device_tag = blob_object->parallel_desc_symbol()->device_tag();
  instruction.set_instr_type_name(device_tag + "." + instruction_name);
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*ConstOperand(blob_object->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(callback_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::FeedBlob(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object, int64_t callback_id) {
  vm::cfg::InstructionProto instruction;
  const std::string& device_tag = blob_object->parallel_desc_symbol()->device_tag();
  instruction.set_instr_type_name(device_tag + "." + "FeedBlob");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Mut2Operand(blob_object->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*Int64Operand(callback_id));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::FetchBlobHeader(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object, int64_t callback_id) {
  JUST(_FetchBlob("FetchBlobHeader", blob_object, callback_id));
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::FetchBlobBody(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object, int64_t callback_id) {
  JUST(_FetchBlob("FetchBlobBody", blob_object, callback_id));
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_TryClearObject(compatible_py::Object* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("TryClearObject");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_DeleteObject(compatible_py::Object* blob_object) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name("DeleteObject");
  instruction.set_parallel_desc_symbol_id(JUST(blob_object->parallel_desc_symbol()->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*DelObjectOperand(blob_object->object_id()));
  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_StatefulCallOpKernel(
    const std::string& instr_name, const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
    const std::shared_ptr<compatible_py::OpKernelObject> opkernel_object,
    const std::shared_ptr<OpNodeSignatureDesc> op_node_signature_sym,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        const_input_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mutable_input_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mut1_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mut2_operand_blob_objects) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name(parallel_desc_sym->device_tag() + "." + instr_name);
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(opkernel_object->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(
      *SymbolOperand(JUST(op_node_signature_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : const_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : const_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*ConstOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mutable_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mutable_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mut1_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mut1_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mut2_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mut2_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*Mut2Operand(pair.second->object_id()));
  }

  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_StatelessCallOpKernel(
    const std::string& instr_name, const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
    const std::shared_ptr<JobDesc>& job_desc_sym,
    const std::shared_ptr<OperatorConfSymbol>& op_conf_sym,
    const std::shared_ptr<OpNodeSignatureDesc>& op_node_signature_sym,
    const std::shared_ptr<compatible_py::Object>& shared_opkernel_obj,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        const_input_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mutable_input_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mut1_operand_blob_objects,
    const std::vector<
        std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>&
        mut2_operand_blob_objects) {
  vm::cfg::InstructionProto instruction;
  instruction.set_instr_type_name(parallel_desc_sym->device_tag() + "." + instr_name);
  instruction.set_parallel_desc_symbol_id(JUST(parallel_desc_sym->symbol_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(job_desc_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(op_conf_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(
      *SymbolOperand(JUST(op_node_signature_sym->symbol_id())));
  instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(shared_opkernel_obj->object_id()));
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : const_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : const_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*ConstOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mutable_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mutable_input_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mut1_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mut1_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*MutOperand(pair.second->object_id()));
  }
  instruction.mutable_operand()->Add()->CopyFrom(*OperandSeparator());

  for (const auto& pair : mut2_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*SymbolOperand(JUST(pair.first->symbol_id())));
  }
  for (const auto& pair : mut2_operand_blob_objects) {
    instruction.mutable_operand()->Add()->CopyFrom(*Mut2Operand(pair.second->object_id()));
  }

  instruction_list_->mutable_instruction()->Add()->CopyFrom(instruction);
  return Maybe<void>::Ok();
}

Maybe<OpNodeSignatureDesc> InstructionsBuilder::GetOpNodeSignatureSymbol(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute) {
  std::shared_ptr<cfg::OpNodeSignature> op_node_signature =
      std::make_shared<cfg::OpNodeSignature>();
  {
    op_node_signature->mutable_sbp_signature()->CopyFrom(op_attribute->sbp_signature());
    op_node_signature->mutable_mirrored_signature()->CopyFrom(op_attribute->mirrored_signature());
    op_node_signature->mutable_logical_blob_desc_signature()->CopyFrom(
        op_attribute->logical_blob_desc_signature());
    op_node_signature->mutable_parallel_signature()->CopyFrom(op_attribute->parallel_signature());
  }
  if (JUST(HasSymbol<cfg::OpNodeSignature>(*op_node_signature))) {
    return GetSymbol<cfg::OpNodeSignature, OpNodeSignatureDesc>(*op_node_signature);
  }
  int64_t symbol_id = JUST(NewSymbolId4OpNodeSignature(op_node_signature));
  JUST(AddSymbol<cfg::OpNodeSignature, OpNodeSignature, OpNodeSignatureDesc>(symbol_id,
                                                                             *op_node_signature));
  return GetSymbol<cfg::OpNodeSignature, OpNodeSignatureDesc>(*op_node_signature);
}

Maybe<void> InstructionsBuilder::StatefulCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<compatible_py::OpKernelObject>& opkernel_object,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<compatible_py::BlobObject>(
        const std::shared_ptr<InstructionsBuilder>&,
        const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& BoxingTo) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym = opkernel_object->parallel_desc_symbol();
  const auto& parallel_sig = op_attribute->parallel_signature();
  CHECK_OR_RETURN(parallel_sig.has_op_parallel_desc_symbol_id());
  CHECK_OR_RETURN(JUST(op_parallel_desc_sym->symbol_id())
                  == parallel_sig.op_parallel_desc_symbol_id());
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));
  const auto FetchDelegateBlobObject =
      [this, &BoxingTo](
          const std::shared_ptr<compatible_py::BlobObject>& x_blob_object,
          const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> std::shared_ptr<compatible_py::BlobObject> {
    return BoxingTo(shared_from_this(), x_blob_object, op_arg_parallel_attr);
  };

  const auto GetDelegateBlobObject =
      [this, &FetchDelegateBlobObject](
          const std::shared_ptr<compatible_py::BlobObject>& blob_object,
          const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> {
    return FindOrCreateDelegateBlobObject(FetchDelegateBlobObject, blob_object,
                                          op_arg_parallel_attr);
  };

  JUST(_StatefulCall(op_attribute, opkernel_object, bn_in_op2blob_object, GetDelegateBlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::StatelessCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<compatible_py::BlobObject>(
        const std::shared_ptr<InstructionsBuilder>&,
        const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& BoxingTo) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));

  const auto FetchDelegateBlobObject =
      [this, &BoxingTo](
          const std::shared_ptr<compatible_py::BlobObject>& x_blob_object,
          const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> std::shared_ptr<compatible_py::BlobObject> {
    // TODO(hanbinbin): use Maybe as return after blobcache is migrated
    return BoxingTo(shared_from_this(), x_blob_object, op_arg_parallel_attr);
  };

  const auto GetDelegateBlobObject =
      [this, &FetchDelegateBlobObject](
          const std::shared_ptr<compatible_py::BlobObject>& blob_object,
          const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> {
    return FindOrCreateDelegateBlobObject(FetchDelegateBlobObject, blob_object,
                                          op_arg_parallel_attr);
  };

  JUST(_StatelessCall("compute", op_attribute, op_parallel_desc_sym, op_parallel_desc_sym,
                      bn_in_op2blob_object, GetDelegateBlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NoBoxingStatelessCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));

  const auto FetchDelegateBlobObject =
      [this](const std::shared_ptr<compatible_py::BlobObject>& blob_object,
             const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> std::shared_ptr<compatible_py::BlobObject> {
    std::shared_ptr<ParallelDesc> from_pd = blob_object->parallel_desc_symbol();
    std::shared_ptr<ParallelDesc> to_pd = op_arg_parallel_attr->parallel_desc_symbol();
    if (*from_pd == *to_pd) { return blob_object; }
    CHECK(from_pd->device_tag() == "cpu");
    CHECK(to_pd->device_tag() == "cpu");
    CHECK(from_pd->parallel_num() == to_pd->parallel_num());

    auto from_machine_ids = from_pd->machine_id2sorted_dev_phy_ids();
    auto to_machine_ids = to_pd->machine_id2sorted_dev_phy_ids();
    if ((from_pd->machine_id2sorted_dev_phy_ids()->size() == from_pd->parallel_num())
        && (Int2IntListMapContaining(*from_machine_ids, *to_machine_ids))
        && (Int2IntListMapContaining(*to_machine_ids, *from_machine_ids))) {
      return CHECK_JUST(BroadcastBlobReference(blob_object, to_pd));
    }
    return CHECK_JUST(Build121To(blob_object, to_pd));
  };

  const auto GetDirectOr121BlobObject =
      [this, &FetchDelegateBlobObject](
          const std::shared_ptr<compatible_py::BlobObject>& blob_object,
          const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> {
    return FindOrCreateDelegateBlobObject(FetchDelegateBlobObject, blob_object,
                                          op_arg_parallel_attr);
  };
  JUST(_StatelessCall("compute", op_attribute, op_parallel_desc_sym, op_parallel_desc_sym,
                      bn_in_op2blob_object, GetDirectOr121BlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NoBoxingCudaD2HStatelessCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& in_parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<std::shared_ptr<ParallelDesc>(const std::shared_ptr<InstructionsBuilder>&,
                                                      const std::shared_ptr<ParallelDesc>&,
                                                      const std::string&)>& TryReplaceDeviceTag) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym =
      JUST(GetParallelDescSymbol(in_parallel_conf));
  std::shared_ptr<ParallelDesc> blob_parallel_desc_sym =
      TryReplaceDeviceTag(shared_from_this(), op_parallel_desc_sym, "cpu");
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));
  const auto GetDirectBlobObject =
      [](const std::shared_ptr<compatible_py::BlobObject>& blob_object,
         const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> { return blob_object; };

  JUST(_StatelessCall("copy_d2h", op_attribute, op_parallel_desc_sym, blob_parallel_desc_sym,
                      bn_in_op2blob_object, GetDirectBlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::NoBoxingCudaH2DStatelessCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& out_parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym =
      JUST(GetParallelDescSymbol(out_parallel_conf));
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));

  const auto GetDirectBlobObject =
      [](const std::shared_ptr<compatible_py::BlobObject>& blob_object,
         const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> { return blob_object; };

  JUST(_StatelessCall("copy_h2d", op_attribute, op_parallel_desc_sym, op_parallel_desc_sym,
                      bn_in_op2blob_object, GetDirectBlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::RawStatelessCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<cfg::ParallelConf>& parallel_conf,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym = JUST(GetParallelDescSymbol(parallel_conf));
  JUST(CheckRefInBlobObjectParallelDesc(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));

  const auto GetDirectBlobObject =
      [](const std::shared_ptr<compatible_py::BlobObject>& blob_object,
         const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      -> Maybe<compatible_py::BlobObject> { return blob_object; };

  JUST(_StatelessCall("compute", op_attribute, op_parallel_desc_sym, op_parallel_desc_sym,
                      bn_in_op2blob_object, GetDirectBlobObject));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_StatefulCall(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<compatible_py::OpKernelObject>& opkernel_object,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<Maybe<compatible_py::BlobObject>(
        const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& GetDelegateBlobObject) {
  std::shared_ptr<ParallelDesc> op_parallel_desc_sym = opkernel_object->parallel_desc_symbol();

  const auto DelegateBlobObject4Ibn =
      [&op_attribute, &bn_in_op2blob_object, &op_parallel_desc_sym,
       &GetDelegateBlobObject](const std::string& ibn) -> Maybe<compatible_py::BlobObject> {
    OpAttribute pb_op_attribute;
    op_attribute->ToProto(&pb_op_attribute);
    std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(op_parallel_desc_sym, pb_op_attribute, ibn));
    return GetDelegateBlobObject(JUST(MapAt(*bn_in_op2blob_object, ibn)), op_arg_parallel_attr);
  };

  std::shared_ptr<OpNodeSignatureDesc> op_node_signature_sym =
      JUST(GetOpNodeSignatureSymbol(op_attribute));

  const auto& const_input_operand_blob_objects =
      JUST(GetConstInputOperandBlobObjects(op_attribute, DelegateBlobObject4Ibn));
  const auto& mutable_input_operand_blob_objects =
      JUST(GetMutableInputOperandBlobObjects(op_attribute, DelegateBlobObject4Ibn));
  const auto& mut1_operand_blob_objects =
      JUST(GetMut1OperandBlobObjects(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));
  const auto& mut2_operand_blob_objects =
      JUST(GetMut2OperandBlobObjects(op_attribute, op_parallel_desc_sym, bn_in_op2blob_object));

  std::string instruction_prefix;
  {
    bool is_user_op = op_attribute->op_conf().has_user_conf();
    CHECK_OR_RETURN(is_user_op);
    if (is_user_op) {
      instruction_prefix = "";
    } else {
      instruction_prefix = "System";
    }
  }

  JUST(_StatefulCallOpKernel(instruction_prefix + "CallOpKernel", op_parallel_desc_sym,
                             opkernel_object, op_node_signature_sym,
                             *const_input_operand_blob_objects, *mutable_input_operand_blob_objects,
                             *mut1_operand_blob_objects, *mut2_operand_blob_objects));

  return Maybe<void>::Ok();
}

Maybe<void> InstructionsBuilder::_StatelessCall(
    const std::string& stream_tag, const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    std::shared_ptr<ParallelDesc> op_parallel_desc_sym,
    const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object,
    const std::function<Maybe<compatible_py::BlobObject>(
        const std::shared_ptr<compatible_py::BlobObject>&,
        const std::shared_ptr<compatible_py::OpArgParallelAttribute>&)>& GetDelegateBlobObject) {
  if (op_attribute->parallel_signature().has_op_parallel_desc_symbol_id()) {
    int64_t symbol_id = op_attribute->parallel_signature().op_parallel_desc_symbol_id();
    op_parallel_desc_sym = JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(symbol_id));
  }
  CHECK_OR_RETURN(op_parallel_desc_sym);
  const auto DelegateBlobObject4Ibn =
      [&op_attribute, &bn_in_op2blob_object, &GetDelegateBlobObject,
       op_parallel_desc_sym](const std::string& ibn) -> Maybe<compatible_py::BlobObject> {
    OpAttribute pb_op_attribute;
    op_attribute->ToProto(&pb_op_attribute);
    std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(op_parallel_desc_sym, pb_op_attribute, ibn));
    return GetDelegateBlobObject(JUST(MapAt(*bn_in_op2blob_object, ibn)), op_arg_parallel_attr);
  };

  const auto& op_conf = op_attribute->op_conf();
  CHECK_OR_RETURN(op_conf.has_scope_symbol_id());
  std::shared_ptr<Scope> scope_symbol =
      JUST(GetSymbol<cfg::ScopeProto, Scope>(op_conf.scope_symbol_id()));
  std::shared_ptr<JobDesc> job_desc_sym = scope_symbol->job_desc_symbol();
  std::shared_ptr<OperatorConfSymbol> op_conf_sym =
      JUST(GetOpConfSymbol(std::make_shared<cfg::OperatorConf>(op_conf)));
  std::shared_ptr<OpNodeSignatureDesc> op_node_signature_sym =
      JUST(GetOpNodeSignatureSymbol(op_attribute));
  std::shared_ptr<compatible_py::Object> opkernel_obj =
      JUST(GetSharedOpKernelObject4ParallelConfSymbol(op_parallel_desc_sym));
  CHECK_OR_RETURN((*opkernel_obj->parallel_desc_symbol()) == *op_parallel_desc_sym);
  const auto& const_input_operand_blob_objects =
      JUST(GetConstInputOperandBlobObjects(op_attribute, DelegateBlobObject4Ibn));
  const auto& mutable_input_operand_blob_objects =
      JUST(GetMutableInputOperandBlobObjects(op_attribute, DelegateBlobObject4Ibn));
  const auto& mut1_operand_blob_objects =
      JUST(GetMut1OperandBlobObjects(op_attribute, blob_parallel_desc_sym, bn_in_op2blob_object));
  const auto& mut2_operand_blob_objects =
      JUST(GetMut2OperandBlobObjects(op_attribute, blob_parallel_desc_sym, bn_in_op2blob_object));
  std::string instruction_prefix;
  {
    bool is_user_op = op_attribute->op_conf().has_user_conf();
    if (is_user_op) {
      instruction_prefix = "User";
    } else {
      instruction_prefix = "System";
    }
  }
  JUST(_StatelessCallOpKernel(
      stream_tag + "." + instruction_prefix + "StatelessCallOpKernel", op_parallel_desc_sym,
      job_desc_sym, op_conf_sym, op_node_signature_sym, opkernel_obj,
      *const_input_operand_blob_objects, *mutable_input_operand_blob_objects,
      *mut1_operand_blob_objects, *mut2_operand_blob_objects));
  return Maybe<void>::Ok();
}

Maybe<compatible_py::BlobObject> InstructionsBuilder::Build121To(
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<ParallelDesc>& parallel_desc_symbol) {
  std::shared_ptr<compatible_py::BlobObject> ref_blob_object =
      JUST(MakeNewBlobObjectLike(shared_from_this(), blob_object, parallel_desc_symbol));
  JUST(Build121AssignInstruction(ref_blob_object, blob_object));
  return ref_blob_object;
}

Maybe<std::vector<
    std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
InstructionsBuilder::GetConstInputOperandBlobObjects(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::function<Maybe<compatible_py::BlobObject>(const std::string&)>& BlobObject4Ibn) {
  std::shared_ptr<std::vector<
      std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
      const_input_operand_blob_objects = std::make_shared<std::vector<
          std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>();
  for (const auto& ibn : op_attribute->input_bns()) {
    const auto& ibn2modifier = op_attribute->arg_modifier_signature().ibn2input_blob_modifier();
    if (JUST(MapAt(ibn2modifier, ibn)).is_mutable()) { continue; }
    std::shared_ptr<StringSymbol> ibn_sym = JUST(GetSymbol4String(ibn));
    std::shared_ptr<compatible_py::BlobObject> in_object = JUST(BlobObject4Ibn(ibn));
    const_input_operand_blob_objects->emplace_back(std::make_pair(ibn_sym, in_object));
  }
  return const_input_operand_blob_objects;
}

Maybe<std::vector<
    std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
InstructionsBuilder::GetMutableInputOperandBlobObjects(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::function<Maybe<compatible_py::BlobObject>(const std::string&)>& BlobObject4Ibn) {
  std::shared_ptr<std::vector<
      std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
      mutable_input_operand_blob_objects = std::make_shared<std::vector<
          std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>();
  for (const auto& ibn : op_attribute->input_bns()) {
    const auto& ibn2modifier = op_attribute->arg_modifier_signature().ibn2input_blob_modifier();
    if (!(JUST(MapAt(ibn2modifier, ibn)).is_mutable())) { continue; }
    std::shared_ptr<StringSymbol> ibn_sym = JUST(GetSymbol4String(ibn));
    std::shared_ptr<compatible_py::BlobObject> in_object = JUST(BlobObject4Ibn(ibn));
    mutable_input_operand_blob_objects->emplace_back(std::make_pair(ibn_sym, in_object));
  }
  return mutable_input_operand_blob_objects;
}

Maybe<std::vector<
    std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
InstructionsBuilder::GetMut1OperandBlobObjects(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  std::shared_ptr<std::vector<
      std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
      mut1_operand_blob_objects = std::make_shared<std::vector<
          std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>();
  const auto GetOutBlobParallelDescSymbol =
      [&op_attribute, &parallel_desc_sym](const std::string& obn) -> Maybe<ParallelDesc> {
    const auto& parallel_signature = op_attribute->parallel_signature();
    const auto& bn2symbol_id = parallel_signature.bn_in_op2parallel_desc_symbol_id();
    if (bn2symbol_id.find(obn) != bn2symbol_id.end()) {
      return GetSymbol<cfg::ParallelConf, ParallelDesc>(bn2symbol_id.at(obn));
    } else {
      return parallel_desc_sym;
    }
  };
  const auto OutputBns = [&op_attribute]() -> std::vector<std::string> {
    const auto& obn2modifier = op_attribute->arg_modifier_signature().obn2output_blob_modifier();
    std::vector<std::string> output_bns;
    for (const auto& obn : op_attribute->output_bns()) {
      if (obn2modifier.at(obn).header_infered_before_compute()) { output_bns.emplace_back(obn); }
    }
    for (const auto& tmp_bn : op_attribute->tmp_bns()) { output_bns.emplace_back(tmp_bn); }
    return output_bns;
  };
  OpAttribute pb_op_attribute;
  op_attribute->ToProto(&pb_op_attribute);
  for (const auto& obn : OutputBns()) {
    std::shared_ptr<StringSymbol> obn_sym = JUST(GetSymbol4String(obn));
    std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(JUST(GetOutBlobParallelDescSymbol(obn)),
                                                      pb_op_attribute, obn));
    std::shared_ptr<compatible_py::OpArgBlobAttribute> op_arg_blob_attr =
        JUST(compatible_py::GetOpArgBlobAttribute(pb_op_attribute, obn));
    std::shared_ptr<compatible_py::BlobObject> out_blob_object =
        JUST(NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr));
    (*bn_in_op2blob_object)[obn] = out_blob_object;
    mut1_operand_blob_objects->emplace_back(std::make_pair(obn_sym, out_blob_object));
  }
  return mut1_operand_blob_objects;
}

Maybe<void> InstructionsBuilder::CheckRefInBlobObjectParallelDesc(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<ParallelDesc>& op_parallel_desc_sym,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  for (const std::string& ibn : op_attribute->input_bns()) {
    const auto& ibn2modifier = op_attribute->arg_modifier_signature().ibn2input_blob_modifier();
    if (!(JUST(MapAt(ibn2modifier, ibn)).is_mutable())) { continue; }
    std::shared_ptr<compatible_py::BlobObject> ref_blob_object = bn_in_op2blob_object->at(ibn);
    CHECK_OR_RETURN(*op_parallel_desc_sym == *ref_blob_object->parallel_desc_symbol());
  }
  return Maybe<void>::Ok();
}

Maybe<std::vector<
    std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
InstructionsBuilder::GetMut2OperandBlobObjects(
    const std::shared_ptr<cfg::OpAttribute>& op_attribute,
    const std::shared_ptr<ParallelDesc>& parallel_desc_sym,
    const std::shared_ptr<HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>>>&
        bn_in_op2blob_object) {
  std::shared_ptr<std::vector<
      std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>
      mut2_operand_blob_objects = std::make_shared<std::vector<
          std::pair<std::shared_ptr<StringSymbol>, std::shared_ptr<compatible_py::BlobObject>>>>();
  const auto GetOutBlobParallelDescSymbol =
      [&op_attribute, &parallel_desc_sym](const std::string& obn) -> Maybe<ParallelDesc> {
    const auto& parallel_signature = op_attribute->parallel_signature();
    const auto& bn2symbol_id = parallel_signature.bn_in_op2parallel_desc_symbol_id();
    if (bn2symbol_id.find(obn) != bn2symbol_id.end()) {
      return GetSymbol<cfg::ParallelConf, ParallelDesc>(JUST(MapAt(bn2symbol_id, obn)));
    } else {
      return parallel_desc_sym;
    }
  };
  OpAttribute pb_op_attribute;
  op_attribute->ToProto(&pb_op_attribute);
  for (const auto& obn : op_attribute->output_bns()) {
    const auto& obn2modifier = op_attribute->arg_modifier_signature().obn2output_blob_modifier();
    if (JUST(MapAt(obn2modifier, obn)).header_infered_before_compute()) { continue; }
    std::shared_ptr<StringSymbol> obn_sym = JUST(GetSymbol4String(obn));

    std::shared_ptr<compatible_py::OpArgParallelAttribute> op_arg_parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(JUST(GetOutBlobParallelDescSymbol(obn)),
                                                      pb_op_attribute, obn));
    std::shared_ptr<compatible_py::OpArgBlobAttribute> op_arg_blob_attr =
        JUST(compatible_py::GetOpArgBlobAttribute(pb_op_attribute, obn));
    std::shared_ptr<compatible_py::BlobObject> out_blob_object =
        JUST(NewBlobObject(op_arg_parallel_attr, op_arg_blob_attr));
    (*bn_in_op2blob_object)[obn] = out_blob_object;
    mut2_operand_blob_objects->emplace_back(std::make_pair(obn_sym, out_blob_object));
  }
  return mut2_operand_blob_objects;
}

Maybe<void> LogicalRun(
    const std::function<void(const std::shared_ptr<InstructionsBuilder>&)>& Build) {
  const auto& RunInstruction =
      [](const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
         const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list) -> Maybe<void> {
    JUST(Global<eager::EagerOneflow>::Get()->RunLogicalInstruction(instruction_list,
                                                                   eager_symbol_list));
    return Maybe<void>::Ok();
  };
  JUST(_Run(Build, std::make_shared<vm::LogicalIdGenerator>(), RunInstruction,
            _ReleaseLogicalObject));
  return Maybe<void>::Ok();
}

Maybe<void> PhysicalRun(
    const std::function<void(const std::shared_ptr<InstructionsBuilder>&)>& Build) {
  const auto& RunInstruction =
      [](const std::shared_ptr<vm::cfg::InstructionListProto>& instruction_list,
         const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list) -> Maybe<void> {
    JUST(Global<eager::EagerOneflow>::Get()->RunPhysicalInstruction(instruction_list,
                                                                    eager_symbol_list));
    return Maybe<void>::Ok();
  };
  JUST(_Run(Build, std::make_shared<vm::PhysicalIdGenerator>(), RunInstruction,
            _ReleasePhysicalObject));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
