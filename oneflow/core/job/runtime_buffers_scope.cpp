#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/runtime_buffers_scope.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

RuntimeBuffersScope::RuntimeBuffersScope() {
  Global<BufferMgr<int64_t>>::New();
  Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::New();
  const auto& job_descs = *Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  Global<BufferMgr<int64_t>>::Get()->NewBuffer(kBufferNameGlobalWaitJobId, job_descs.size());
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  FOR_RANGE(int64_t, job_id, 0, job_descs.size()) {
    const auto& job_name = GlobalJobDesc(job_id).job_name();
    buffer_mgr->NewBuffer(GetForeignInputBufferName(job_name), 2);
    buffer_mgr->NewBuffer(GetForeignOutputBufferName(job_name), 2);
    buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name),
                          job_descs.at(0)->concurrency_width());
  }
}

RuntimeBuffersScope::~RuntimeBuffersScope() {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  const auto& job_descs = *Global<std::vector<std::unique_ptr<JobDesc>>>::Get();
  FOR_RANGE(int64_t, job_id, 0, job_descs.size()) {
    const auto& job_name = GlobalJobDesc(job_id).job_name();
    buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Close();
    buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Close();
    buffer_mgr->Get(GetForeignInputBufferName(job_name))->Close();
  }
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Close();
  Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Delete();
  Global<BufferMgr<int64_t>>::Delete();
}

}  // namespace oneflow
