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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/eager/eager_symbol.pb.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/instruction.cfg.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/vm/string_symbol.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/cluster_instruction.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_conf_symbol.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace eager {

namespace {

Maybe<void> StorageAdd(const EagerSymbol& symbol) {
  int64_t symbol_id = symbol.symbol_id();
  if (symbol.has_string_symbol()) {
    JUST(Global<symbol::Storage<StringSymbol>>::Get()->TryAdd(symbol_id, symbol.string_symbol()));
  } else if (symbol.has_scope_symbol()) {
    JUST(Global<symbol::Storage<Scope>>::Get()->TryAdd(symbol_id, symbol.scope_symbol()));
  } else if (symbol.has_job_conf_symbol()) {
    JUST(Global<symbol::Storage<JobDesc>>::Get()->TryAdd(symbol_id, symbol.job_conf_symbol()));
  } else if (symbol.has_parallel_conf_symbol()) {
    JUST(Global<symbol::Storage<ParallelDesc>>::Get()->TryAdd(symbol_id,
                                                              symbol.parallel_conf_symbol()));
  } else if (symbol.has_op_conf_symbol()) {
    JUST(Global<symbol::Storage<OperatorConfSymbol>>::Get()->TryAdd(symbol_id,
                                                                    symbol.op_conf_symbol()));
  } else if (symbol.has_op_node_signature_symbol()) {
    JUST(Global<symbol::Storage<OpNodeSignatureDesc>>::Get()->TryAdd(
        symbol_id, symbol.op_node_signature_symbol()));
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> EagerOneflow::RunPhysicalInstruction(
    const std::shared_ptr<const ClusterInstructionProto>& cluster_instruction) {
  const auto& instruction_list = std::make_shared<vm::InstructionMsgList>();
  const auto& eage_instructions = cluster_instruction->eager_instruction();
  for (const auto& instr_proto : eage_instructions.instruction_list().instruction()) {
    instruction_list->EmplaceBack(ObjectMsgPtr<vm::InstructionMsg>::New(instr_proto));
  }
  for (const auto& eager_symbol : eage_instructions.eager_symbol_list().eager_symbol()) {
    JUST(StorageAdd(eager_symbol));
  }
  return vm::Run(instruction_list);
}

Maybe<void> EagerOneflow::RunPhysicalInstruction(
    const std::shared_ptr<vm::InstructionMsgList>& instruction_list,
    const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list) {
  for (const auto& cfg_symbol : eager_symbol_list->eager_symbol()) {
    EagerSymbol eager_symbol;
    cfg_symbol.ToProto(&eager_symbol);
    JUST(StorageAdd(eager_symbol));
  }
  return vm::Run(instruction_list);
}

Maybe<void> EagerOneflow::RunLogicalInstruction(
    const std::shared_ptr<vm::InstructionMsgList>& instruction_list,
    const std::shared_ptr<eager::cfg::EagerSymbolList>& eager_symbol_list) {
  auto cluster_instruction = std::make_shared<ClusterInstructionProto>();
  auto* repeated_instruction_proto = cluster_instruction->mutable_eager_instruction()
                                         ->mutable_instruction_list()
                                         ->mutable_instruction();
  OBJECT_MSG_LIST_FOR_EACH_PTR(instruction_list.get(), instruction_msg) {
    instruction_msg->ToProto(repeated_instruction_proto->Add());
  }
  eager_symbol_list->ToProto(
      cluster_instruction->mutable_eager_instruction()->mutable_eager_symbol_list());
  CHECK(GlobalProcessCtx::IsThisProcessMaster());
  ClusterInstruction::MasterSendEagerInstruction(*cluster_instruction);
  return RunPhysicalInstruction(instruction_list, eager_symbol_list);
}

COMMAND(Global<EagerOneflow>::SetAllocated(new EagerOneflow()));

}  // namespace eager
}  // namespace oneflow
