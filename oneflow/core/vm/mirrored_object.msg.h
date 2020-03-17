#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/mirrored_object_id.msg.h"
#include "oneflow/core/vm/mem_zone_desc.msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"

namespace oneflow {

namespace vm {

class Instruction;
class MirroredObject;

// clang-format off
OBJECT_MSG_BEGIN(MirroredObjectAccess);
  // methods
  PUBLIC void __Init__(Instruction* instruction, MirroredObject* mirrored_object,
                       bool is_const_operand);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(bool, is_const_operand);
  OBJECT_MSG_DEFINE_PTR(Instruction, instruction);
  OBJECT_MSG_DEFINE_PTR(MirroredObject, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(mirrored_object_access_link);
  OBJECT_MSG_DEFINE_SKIPLIST_FLAT_MSG_KEY(10, MirroredObjectId, mirrored_object_id);
  
OBJECT_MSG_END(MirroredObjectAccess);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(HostMemBuffer);
  // methods
  PUBLIC void __Init__(size_t size, char* data) {
    set_size(size);
    set_data(data);
  }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(size_t, size);
  OBJECT_MSG_DEFINE_PTR(char, data);
OBJECT_MSG_END(HostMemBuffer);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(CudaMemBuffer);
  // methods
  PUBLIC void __Init__(size_t size, char* data) {
    set_size(size);
    set_data(data);
  }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(size_t, size);
  OBJECT_MSG_DEFINE_PTR(char, data);
OBJECT_MSG_END(CudaMemBuffer);
// clang-format on

class LogicalObject;
// clang-format off
OBJECT_MSG_BEGIN(MirroredObject);
  // methods
  PUBLIC void __Init__(LogicalObject* logical_object, int64_t parallel_id);
  //fields
  OBJECT_MSG_DEFINE_FLAT_MSG(MirroredObjectId, mirrored_object_id);
  OBJECT_MSG_DEFINE_PTR(LogicalObject, logical_object);
  OBJECT_MSG_DEFINE_ONEOF(object_type,
      OBJECT_MSG_ONEOF_FIELD(HostMemBuffer, host_mem_buffer)
      OBJECT_MSG_ONEOF_FIELD(CudaMemBuffer, cuda_mem_buffer));

  // links
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, parallel_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, mirrored_object_access_link, access_list);
OBJECT_MSG_END(MirroredObject);
// clang-format on

class Scheduler;
// clang-format off
OBJECT_MSG_BEGIN(LogicalObject);
  // methods
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id,
                       Scheduler* scheduler) {
    set_logical_object_id(logical_object_id);
    set_scheduler(scheduler);
  }
  // fields
  OBJECT_MSG_DEFINE_PTR(Scheduler, scheduler);
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, parallel_id, parallel_id2mirrored_object);
  OBJECT_MSG_DEFINE_MAP_KEY(LogicalObjectId, logical_object_id);
OBJECT_MSG_END(LogicalObject);
// clang-format on

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
