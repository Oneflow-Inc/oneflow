#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_

#include "oneflow/core/kernel/util/interface_bridge.h"

namespace oneflow {

template<DeviceType deivce_type>
struct NewKernelUtil : public DnnIf<deivce_type>,
                       public BlasIf<deivce_type>,
                       public ArithemeticIf<deivce_type> {};

template<DeviceType device_type>
struct GetCudaMemcpyKind;

template<>
struct GetCudaMemcpyKind<DeviceType::kCPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToHost;
};

template<>
struct GetCudaMemcpyKind<DeviceType::kGPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

template<DeviceType device_type>
void Memcpy(DeviceCtx*, void* dst, const void* src, size_t sz,
            cudaMemcpyKind kind = GetCudaMemcpyKind<device_type>::val);

template<DeviceType device_type>
void Memset(DeviceCtx*, void* dst, const char value, size_t sz);

void WithHostBlobAndStreamSynchronizeEnv(DeviceCtx* ctx, Blob* blob,
                                         std::function<void(Blob*)> Callback);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
