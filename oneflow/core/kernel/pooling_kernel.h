#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class PoolingKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernel);
  PoolingKernel() = default;
  ~PoolingKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename FloatingPointType>
class PoolingKernelUtil;

template<typename FloatingPointType>
class PoolingKernelUtil<DeviceType::kCPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void RangeMaxQuery(
      const FloatingPointType* in_dptr, int64_t in_width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
  static void RangeAveQuery(
      const FloatingPointType* in_dptr, int64_t in_width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
  static void RangeStoQuery(
      const FloatingPointType* in_dptr, int64_t width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
}

template<typename FloatingPointType>
class PoolingKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void RangeMaxQuery(
      const FloatingPointType* in_dptr, int64_t in_width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
  static void RangeAveQuery(
      const FloatingPointType* in_dptr, int64_t in_width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
  static void RangeStoQuery(
      const FloatingPointType* in_dptr, int64_t width,
      int64_t hstart, int64_t wstart, int64_t hend, int64_t wend,
      FloatingPointType* out_dptr, int64_t* mask_dptr, int64_t out_index);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
