#ifndef ONEFLOW_CORE_JOB_NVTX_CTX_H_
#define ONEFLOW_CORE_JOB_NVTX_CTX_H_
#include <map>
#include "oneflow/core/common/util.h"
#include "oneflow/core/nvtx3/nvToolsExt.h"

namespace oneflow {

class NvtxCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NvtxCtx);
  NvtxCtx() = default;
  ~NvtxCtx() = default;
  void PutRangeId(const std::string& msg, const nvtxRangeId_t id);
  nvtxRangeId_t PopRangeId(const std::string& msg);

 private:
  friend class Global<NvtxCtx>;
  std::map<std::string, nvtxRangeId_t> msg2range_id_;
  mutable std::mutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_NVTX_CTX_H_
