#ifndef ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/device/device_context.h"
#include <curand.h>

#include <curand_kernel.h>

namespace oneflow {

template<DeviceType device_type>
class DropoutUtil;

template<>
class DropoutUtil<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutUtil);
  DropoutUtil(int64_t seed, DeviceCtx* device_ctx) : mt19937_generator_(seed) {}
  ~DropoutUtil() {}

  template<typename T>
  void Dropout(const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y,
               int8_t* mask);

 private:
  std::mt19937 mt19937_generator_;
};

template<>
class DropoutUtil<DeviceType::kGPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DropoutUtil);
  DropoutUtil(int64_t seed, DeviceCtx* device_ctx);
  ~DropoutUtil();

  template<typename T>
  void Dropout(const int64_t elem_cnt, const float threshold, const float scale, const T* x, T* y,
               int8_t* mask);

 private:
  curandState* curand_states_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_DORPOUT_UTIL_H_
