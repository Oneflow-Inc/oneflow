#include "oneflow/core/eager/eager_util.h"
#include "oneflow/core/eager/eager_symbol.pb.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/storage.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace eager {

namespace {

void StorageAdd(const EagerSymbol& symbol) {
  int64_t symbol_id = symbol.symbol_id();
  if (symbol.has_string_symbol()) {
    const auto& str = std::make_shared<std::string>(symbol.string_symbol());
    Global<vm::Storage<std::string>>::Get()->Add(symbol_id, str);
  } else if (symbol.has_job_conf_symbol()) {
    const auto& job_conf = std::make_shared<JobConfigProto>(symbol.job_conf_symbol());
    Global<vm::Storage<JobConfigProto>>::Get()->Add(symbol_id, job_conf);
  } else if (symbol.has_parallel_conf_symbol()) {
    const auto& parallel_conf = std::make_shared<ParallelConf>(symbol.parallel_conf_symbol());
    Global<vm::Storage<ParallelConf>>::Get()->Add(symbol_id, parallel_conf);
  } else if (symbol.has_op_conf_symbol()) {
    const auto& op_conf = std::make_shared<OperatorConf>(symbol.op_conf_symbol());
    Global<vm::Storage<OperatorConf>>::Get()->Add(symbol_id, op_conf);
  } else {
    UNIMPLEMENTED();
  }
}

Maybe<void> RunPhysicalInstruction(const vm::InstructionListProto& instruction_list_proto,
                                   const EagerSymbolList& eager_symbol_list) {
  for (const auto& eager_symbol : eager_symbol_list.eager_symbol()) { StorageAdd(eager_symbol); }
  return vm::Run(instruction_list_proto);
}

}  // namespace

Maybe<void> RunPhysicalInstruction(const std::string& instruction_list_proto_str,
                                   const std::string& eager_symbol_list_str) {
  vm::InstructionListProto instruction_list_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(instruction_list_proto_str, &instruction_list_proto))
      << "InstructionListProto parse failed";
  EagerSymbolList eager_symbol_list;
  CHECK_OR_RETURN(TxtString2PbMessage(eager_symbol_list_str, &eager_symbol_list))
      << "EagerSymbolList parse failed";
  return RunPhysicalInstruction(instruction_list_proto, eager_symbol_list);
}

}  // namespace eager
}  // namespace oneflow
