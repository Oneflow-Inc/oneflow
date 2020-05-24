#ifndef ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/device_context.h"
#include <curand.h>
#include <curand_kernel.h>

namespace oneflow {

template<DeviceType device_type>
class FusedDropout;

template<>
class FusedDropout<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FusedDropout);
  FusedDropout(int64_t seed, DeviceCtx* device_ctx) : mt19937_generator_(seed) {}
  ~FusedDropout() {}

  template<typename T>
  void Dropout(const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y,
               int8_t* mask);

 private:
  std::mt19937 mt19937_generator_;
};

template<>
class FusedDropout<DeviceType::kGPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FusedDropout);
  FusedDropout(int64_t seed, DeviceCtx* device_ctx);
  ~FusedDropout();

  template<typename T>
  void Dropout(const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y,
               int8_t* mask);

 private:
  curandState* curand_states_;
  int32_t block_num_;
  int32_t thread_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_
