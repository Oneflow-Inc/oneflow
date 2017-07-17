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

  void InitFromOpProto(const OperatorProto& op_proto) override;
  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  void (*PoolingMethodForwardFunc_)(const FloatingPointType*, const int64_t,
                                    const int64_t, const int64_t, const int64_t,
                                    const int64_t, const int64_t, const int64_t,
                                    const int64_t, FloatingPointType*,
                                    int64_t*);
  void (*PoolingMethodBackwardFunc_)(const FloatingPointType*, const int64_t*,
                                     const int64_t, const int64_t,
                                     const int64_t, const int64_t,
                                     const int64_t, const int64_t,
                                     const int64_t, const int64_t,
                                     FloatingPointType*);
};

template<DeviceType device_type, typename FloatingPointType>
class PoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void RangeMaxQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr);

  static void RangeAveQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr);

  static void RangeStoQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr);

  static void PoolingMaxBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr);

  static void PoolingAveBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr);

  static void PoolingStoBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_H_
