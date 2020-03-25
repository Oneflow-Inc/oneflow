#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"

namespace oneflow {
namespace vm {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(StreamTypeRegistry);
  // methods
  PUBLIC void __Init__(const StreamType* stream_type, const StreamTypeId& stream_type_id) {
    set_stream_type(stream_type);
    mutable_stream_type_id()->CopyFrom(stream_type_id);
  }
  // fields
  OBJECT_MSG_DEFINE_PTR(const StreamType, stream_type);
  // links
  OBJECT_MSG_DEFINE_MAP_KEY(StreamTypeId, stream_type_id);
OBJECT_MSG_END(StreamTypeRegistry);
// clang-format on

using StreamTypeRegistryMap = OBJECT_MSG_MAP(StreamTypeRegistry, stream_type_id);

StreamTypeRegistryMap* StreamType4StreamTypeId() {
  static StreamTypeRegistryMap map;
  return &map;
}

}  // namespace

void StreamType::Run(InstrChain* instr_chain) const {
  auto interpret_type = instr_chain->stream().stream_id().stream_type_id().interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(instr_chain);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(instr_chain);
  } else {
    UNIMPLEMENTED();
  }
}

const StreamType* LookupStreamType(const StreamTypeId& stream_type_id) {
  auto* map = StreamType4StreamTypeId();
  auto* registry = map->FindPtr(stream_type_id);
  CHECK_NOTNULL(registry);
  return &registry->stream_type();
}

void RegisterStreamType(const std::type_index& stream_type_index, const StreamType* stream_type) {
  auto Register = [&](InterpretType interpret_type) {
    FlatMsg<StreamTypeId> stream_type_id;
    stream_type_id->__Init__(stream_type_index, interpret_type);
    auto registry = ObjectMsgPtr<StreamTypeRegistry>::New(stream_type, stream_type_id.Get());
    CHECK(StreamType4StreamTypeId()->Insert(registry.Mutable()).second);
  };
  Register(InterpretType::kCompute);
  Register(InterpretType::kInfer);
}

}  // namespace vm
}  // namespace oneflow
