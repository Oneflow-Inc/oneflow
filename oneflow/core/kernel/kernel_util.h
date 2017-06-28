#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/blas/cblas_template.h"
#include "oneflow/core/blas/cublas_template.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type> 
class KernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  // memory copy
  static void Memcpy(const KernelCtx& ctx, 
     void* dst, const void* src, size_t sz);

  // memory set
  static void Memset(const KernelCtx& ctx, void* dst, const char value,
      size_t sz);

  // level 1 vector and vector
  // dot product
  static void  BlasDot(const KernelCtx& ctx,
      const int n, const floating_point_type* x, const int incx,
      const floating_point_type* y, const int incy, floating_point_type* result);

  // swap x and y
  static void BlasSwap(const KernelCtx& ctx,
      const int n,
      floating_point_type* x, const int incx,
      floating_point_type* y, const int incy);

  // copy x into y
  static void BlasCopy(const KernelCtx& ctx,
      const int n,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy);

  // y = a*x + y
  static void BlasAxpy(const KernelCtx& ctx, const int n,
      const floating_point_type alpha,
      const floating_point_type* x, const int incx,
      floating_point_type* y, const int incy);

  // x = a*x
  static void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, const int incx);

  // level 2 matrix and vector
  // matrix vector multiply
  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* a, int lda, const floating_point_type* x, 
      const int incx, const floating_point_type beta, 
      floating_point_type* y, const int incy);

  // level 3 matrix and matrix
  // matrix matrix multiply
  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
      const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
      const floating_point_type alpha, const floating_point_type* a,
      const int lda, const floating_point_type* b, const int ldb,
      const floating_point_type beta, floating_point_type* c, const int ldc);
};

}  // namespace oneflow
#endif // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
