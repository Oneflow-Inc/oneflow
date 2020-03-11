#ifndef ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx;

class CudaVmInstrStatusQuerier {
 public:
  ~CudaVmInstrStatusQuerier() = default;

  bool done() const { return launched_ && event_completed(); }
  void SetLaunched(DeviceCtx* device_ctx);

  static const CudaVmInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const CudaVmInstrStatusQuerier*>(mem_ptr);
  }
  static CudaVmInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<CudaVmInstrStatusQuerier*>(mem_ptr);
  }
  static CudaVmInstrStatusQuerier* PlacementNew(char* mem_ptr, int64_t device_id) {
    return new (mem_ptr) CudaVmInstrStatusQuerier(device_id);
  }

 private:
  explicit CudaVmInstrStatusQuerier(int64_t device_id) : launched_(false), device_id_(device_id) {}
  bool event_completed() const;

  volatile bool launched_;
  int64_t device_id_;
  cudaEvent_t event_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
