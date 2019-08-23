#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/job/runtime_buffers_scope.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

RuntimeBuffersScope::RuntimeBuffersScope(const Plan& plan) {
  size_t job_size = Global<JobName2JobId>::Get()->size();
  Global<BufferMgr<int64_t>>::Get()->NewBuffer(kBufferNameGlobalWaitJobId, job_size);
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  for (const auto& pair : plan.job_confs().job_id2job_conf()) {
    const auto& job_name = pair.second.job_name();
    CHECK_EQ(pair.first, Global<JobName2JobId>::Get()->at(job_name));
    buffer_mgr->NewBuffer(GetForeignInputBufferName(job_name), 2);
    buffer_mgr->NewBuffer(GetForeignOutputBufferName(job_name), 2);
    size_t concurrency_width = pair.second.concurrency_width();
    buffer_mgr->NewBuffer(GetCallbackNotifierBufferName(job_name), concurrency_width);
  }
}

RuntimeBuffersScope::~RuntimeBuffersScope() {
  auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<ForeignJobInstance>>>::Get();
  for (const auto& pair : *Global<JobName2JobId>::Get()) {
    const auto& job_name = pair.first;
    buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Close();
    buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Close();
    buffer_mgr->Get(GetForeignInputBufferName(job_name))->Close();
  }
  Global<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Close();
}

}  // namespace oneflow
