#include "oneflow/core/job/nvtx_ctx.h"

namespace oneflow {

void NvtxCtx::PutRangeId(const std::string& msg, const nvtxRangeId_t id) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  auto iter = this->msg2range_id_.find(msg);
  if (iter != this->msg2range_id_.end()) {
    LOG(FATAL) << "NVTX range id entry not empty when putting new id, message: " << msg
               << ", range id: " << iter->second;
  } else {
    this->msg2range_id_[msg] = id;
  }
}

nvtxRangeId_t NvtxCtx::PopRangeId(const std::string& msg) {
  nvtxRangeId_t ret = 0;
  std::lock_guard<std::mutex> lock(this->mutex_);
  auto iter = this->msg2range_id_.find(msg);
  if (iter != this->msg2range_id_.end()) {
    ret = iter->second;
    this->msg2range_id_.erase(iter);
  } else {
    LOG(FATAL) << "NVTX range id not found when poping message: " << msg;
  }
  return ret;
}

}  // namespace oneflow
