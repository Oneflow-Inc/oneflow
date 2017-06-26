#ifndef ONEFLOW_CORE_BLAS_MATH_H_
#define ONEFLOW_CORE_BLAS_MATH_H_
#include <string>
#include "oneflow/core/kernel/kernel_context.h"
//#include "oneflow/core/actor/device_context.h"
#include "oneflow/core/job/resource.pb.h"
//#include "oneflow/core/actor/cuda_device_context.h"
//#include "oneflow/core/blas/cblas.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type> class Math;

template<typename floating_point_type>
class Math<DeviceType::kCPU, floating_point_type> {
 public:
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
class Math<DeviceType::kGPU, floating_point_type> {
 public:
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
#endif // ONEFLOW_CORE_BLAS_MATH_H__
