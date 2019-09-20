#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/runtime_buffer_managers_scope.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

RuntimeBufferManagersScope::RuntimeBufferManagersScope() {
  Global<BufferMgr<int64_t>>::New();
  Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::New();
}

RuntimeBufferManagersScope::~RuntimeBufferManagersScope() {
  Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Delete();
  Global<BufferMgr<int64_t>>::Delete();
}

}  // namespace oneflow
