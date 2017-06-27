#ifndef ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#define ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#include <string>
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type> class KernelUtil;

template<typename floating_point_type>
class KernelUtil<DeviceType::kCPU, floating_point_type> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = default;
  ~KernelUtil() = default;

  OF_SINGLETON(KernelUtil);

  void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz);

  void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz);

  void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY);

  void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx);
};

template<typename floating_point_type>
class KernelUtil<DeviceType::kGPU, floating_point_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = default;
  ~KernelUtil() = default;

  OF_SINGLETON(KernelUtil);

  void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz);

  void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz);

  void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY);

  void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx);
};

}  // namespace oneflow
#endif // ONEFLOW_CORE_BLAS_KERNELUTIL_H__
