#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<StreamTypeId, StreamTypeId>* InferStreamTypeId4ComputeStreamTypeId() {
  static HashMap<StreamTypeId, StreamTypeId> map;
  return &map;
}

}  // namespace

HashMap<std::type_index, const StreamType*>* StreamType4TypeIndex() {
  static HashMap<std::type_index, const StreamType*> map;
  return &map;
}

const StreamTypeId& LookupInferStreamTypeId(const StreamTypeId& compute_stream_type_id) {
  return InferStreamTypeId4ComputeStreamTypeId()->at(compute_stream_type_id);
}

void StreamType::Run(InstrChain* instr_chain) const {
  const auto& stream_type_id = instr_chain->stream().stream_id().stream_type_id();
  auto interpret_type = stream_type_id.interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(instr_chain);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(instr_chain);
  } else {
    UNIMPLEMENTED();
  }
}

void StreamType::Run(Scheduler* scheduler, InstructionMsg* instr_msg) const {
  InterpretType interpret_type = instr_msg->instr_type_id().stream_type_id().interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(scheduler, instr_msg);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(scheduler, instr_msg);
  } else {
    UNIMPLEMENTED();
  }
}

void StreamType::Run(Scheduler* scheduler, InstrChain* instr_chain) const {
  auto interpret_type = instr_chain->stream().stream_id().stream_type_id().interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(scheduler, instr_chain);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(scheduler, instr_chain);
  } else {
    UNIMPLEMENTED();
  }
}

void TryRegisterInferStreamTypeId(const StreamType* infer_stream_type,
                                  const StreamType* compute_stream_type) {
  StreamTypeId compute_stream_type_id;
  compute_stream_type_id.__Init__(compute_stream_type, InterpretType::kCompute);
  StreamTypeId infer_stream_type_id;
  infer_stream_type_id.__Init__(infer_stream_type, InterpretType::kInfer);
  InferStreamTypeId4ComputeStreamTypeId()->emplace(compute_stream_type_id, infer_stream_type_id);
}

}  // namespace vm
}  // namespace oneflow
