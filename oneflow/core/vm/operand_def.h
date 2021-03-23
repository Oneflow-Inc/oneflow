#ifndef ONEFLOW_CORE_VM_OPERAND_DEF_H_
#define ONEFLOW_CORE_VM_OPERAND_DEF_H_

#include "oneflow/core/object_msg/dss.h"
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/common/fixed_vector.h"

namespace oneflow {

namespace vm {

template<typename T>
using OperandVec = std::vector<T>;

using OperandListT = OperandVec<FlatMsg<InstructionOperand>>;

}

#define FLAT_MSG_VIEW_DEFINE_OPERAND(type, field_name) \
  FLAT_MSG_VIEW_DEFINE_PATTERN(type, field_name)

#define FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(type, field_name) \
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(type, field_name)

}

#endif  // ONEFLOW_CORE_VM_OPERAND_DEF_H_
