#ifndef ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx;

namespace vm {

class CudaInstrStatusQuerier {
 public:
  ~CudaInstrStatusQuerier() = default;

  bool done() const { return launched_ && event_completed(); }
  void SetLaunched(DeviceCtx* device_ctx);

  static const CudaInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const CudaInstrStatusQuerier*>(mem_ptr);
  }
  static CudaInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<CudaInstrStatusQuerier*>(mem_ptr);
  }
  static CudaInstrStatusQuerier* PlacementNew(char* mem_ptr, int64_t device_id) {
    return new (mem_ptr) CudaInstrStatusQuerier(device_id);
  }

 private:
  explicit CudaInstrStatusQuerier(int64_t device_id) : launched_(false), device_id_(device_id) {}
  bool event_completed() const;

  volatile bool launched_;
  int64_t device_id_;
  cudaEvent_t event_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CUDA_VM_INSTRUCTION_STATUS_QUERIER_H_
