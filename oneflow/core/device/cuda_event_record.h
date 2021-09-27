#ifndef ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_
#define ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_

#include "oneflow/core/device/event_record.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA
class DeviceCtx;
class CudaEventRecord final : public EventRecord {
 public:
  CudaEventRecord(const CudaEventRecord&) = delete;
  CudaEventRecord(CudaEventRecord&&) = delete;
  CudaEventRecord& operator=(const CudaEventRecord&) = delete;
  CudaEventRecord& operator=(CudaEventRecord&&) = delete;

  explicit CudaEventRecord(DeviceCtx* device_ctx);
  CudaEventRecord(int64_t device_id, DeviceCtx* device_ctx);
  ~CudaEventRecord() = default;

  bool QueryDone() const override;

 private:
  int64_t device_id_;
  cudaEvent_t event_;
};
#endif

}

#endif  // ONEFLOW_CORE_DEVICE_CUDA_EVENT_RECORD_H_
