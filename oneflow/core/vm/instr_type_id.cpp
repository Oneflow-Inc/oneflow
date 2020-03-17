#include "oneflow/core/vm/instr_type_id.msg.h"
#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

void InstrTypeId::__Init__(const std::string& instr_type_name) {
  CopyFrom(LookupInstrTypeId(instr_type_name));
}

}  // namespace vm
}  // namespace oneflow
