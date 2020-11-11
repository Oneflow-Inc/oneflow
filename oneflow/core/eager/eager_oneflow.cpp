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
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/eager/eager_symbol.pb.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/eager/eager_symbol_storage.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace eager {

namespace {

void StorageAdd(const EagerSymbol& symbol) {
  int64_t symbol_id = symbol.symbol_id();
  if (symbol.has_string_symbol()) {
    Global<vm::SymbolStorage<std::string>>::Get()->Add(symbol_id, symbol.string_symbol());
  } else if (symbol.has_scope_symbol()) {
    Global<vm::SymbolStorage<Scope>>::Get()->Add(symbol_id, symbol.scope_symbol());
  } else if (symbol.has_job_conf_symbol()) {
    Global<vm::SymbolStorage<JobDesc>>::Get()->Add(symbol_id, symbol.job_conf_symbol());
  } else if (symbol.has_parallel_conf_symbol()) {
    Global<vm::SymbolStorage<ParallelDesc>>::Get()->Add(symbol_id, symbol.parallel_conf_symbol());
  } else if (symbol.has_op_conf_symbol()) {
    Global<vm::SymbolStorage<OperatorConf>>::Get()->Add(symbol_id, symbol.op_conf_symbol());
  } else if (symbol.has_op_node_signature_symbol()) {
    Global<vm::SymbolStorage<OpNodeSignatureDesc>>::Get()->Add(symbol_id,
                                                               symbol.op_node_signature_symbol());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

Maybe<void> EagerOneflow::RunPhysicalInstruction(
    const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction) {
  const vm::InstructionListProto& instruction_list_proto =
      cluster_instruction->eager_instruction().instruction_list();
  const EagerSymbolList& eager_symbol_list =
      cluster_instruction->eager_instruction().eager_symbol_list();
  for (const auto& eager_symbol : eager_symbol_list.eager_symbol()) { StorageAdd(eager_symbol); }
  return vm::Run(instruction_list_proto);
}

Maybe<void> EagerOneflow::RunPhysicalInstruction(const std::string& instruction_list_proto_str,
                                                 const std::string& eager_symbol_list_str) {
  auto cluster_instruction = std::make_shared<ClusterInstructionProto>();
  vm::InstructionListProto* instruction_list_proto =
      cluster_instruction->mutable_eager_instruction()->mutable_instruction_list();
  CHECK_OR_RETURN(TxtString2PbMessage(instruction_list_proto_str, instruction_list_proto))
      << "InstructionListProto parse failed";
  EagerSymbolList* eager_symbol_list =
      cluster_instruction->mutable_eager_instruction()->mutable_eager_symbol_list();
  CHECK_OR_RETURN(TxtString2PbMessage(eager_symbol_list_str, eager_symbol_list))
      << "EagerSymbolList parse failed";
  return RunPhysicalInstruction(
      std::const_pointer_cast<const ClusterInstructionProto>(cluster_instruction));
}

Maybe<void> EagerOneflow::RunLogicalInstruction(
    const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction) {
  CHECK(cluster_instruction->has_eager_instruction());
  CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
  ClusterInstruction::MasterSendEagerInstruction(*cluster_instruction);
  return RunPhysicalInstruction(cluster_instruction);
}

Maybe<void> EagerOneflow::RunLogicalInstruction(const std::string& instruction_list_proto_str,
                                                const std::string& eager_symbol_list_str) {
  auto cluster_instruction = std::make_shared<ClusterInstructionProto>();
  vm::InstructionListProto* instruction_list_proto =
      cluster_instruction->mutable_eager_instruction()->mutable_instruction_list();
  CHECK_OR_RETURN(TxtString2PbMessage(instruction_list_proto_str, instruction_list_proto))
      << "InstructionListProto parse failed";
  EagerSymbolList* eager_symbol_list =
      cluster_instruction->mutable_eager_instruction()->mutable_eager_symbol_list();
  CHECK_OR_RETURN(TxtString2PbMessage(eager_symbol_list_str, eager_symbol_list))
      << "EagerSymbolList parse failed";
  return RunLogicalInstruction(
      std::const_pointer_cast<const ClusterInstructionProto>(cluster_instruction));
}

COMMAND(Global<EagerOneflow>::SetAllocated(new EagerOneflow()));

}  // namespace eager
}  // namespace oneflow
