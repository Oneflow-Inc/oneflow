#ifndef ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#define ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#include <string>
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type> 
class KernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = default;
  ~KernelUtil() = default;

  static void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz);

  static void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz);

  static void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY);

  static void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx);

};

}  // namespace oneflow
#endif // ONEFLOW_CORE_BLAS_KERNELUTIL_H__
