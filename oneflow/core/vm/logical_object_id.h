#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_

#include <cstdint>
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {
namespace vm {

using LogicalObjectId = int64_t;

int64_t NewNaiveLogicalObjectId();
int64_t NewConstHostLogicalObjectId();
bool IsNaiveLogicalObjectId(int64_t logical_object_id);
bool IsConstHostLogicalObjectId(int64_t logical_object_id);

int64_t GetTypeLogicalObjectId(int64_t value_logical_object_id);
bool IsTypeLogicalObjectId(int64_t logical_object_id);
bool IsValueLogicalObjectId(int64_t logical_object_id);
int64_t GetSelfLogicalObjectId(int64_t logical_object_id);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
