#ifndef ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#define ONEFLOW_CORE_BLAS_KERNELUTIL_H_
#include <string>
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
      const int N, const floating_point_type* X, const int incX,
      const floating_point_type* Y, const int incY, floating_point_type* result);

  // swap x and y
  static void BlasSwap(const KernelCtx& ctx,
      const int N,
      floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY);

  // copy x into y
  static void BlasCopy(const KernelCtx& ctx,
      const int N,
      const floating_point_type* X, const int incX,
      floating_point_type* Y, const int incY);

  // y = a*x + y
  static void BlasAxpy(const KernelCtx& ctx, const int N,
      const floating_point_type alpha,
      const floating_point_type* X, const int incX,
      floating_point_type *Y, const int incY);

  // x = a*x
  static void BlasScal(const KernelCtx& ctx, const int n,
      const floating_point_type alpha, floating_point_type* x, int incx);

  // level 2 matrix and vector
  // matrix vector multiply
  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans, 
      int m, int n, const floating_point_type alpha, 
      const floating_point_type* A, int lda, const floating_point_type* x, 
      int incx, const floating_point_type beta, 
      floating_point_type* y, int incy);

  // level 3 matrix and matrix
  // matrix matrix multiply
  static void BlasGemm(const KernelCtx& ctx,
      const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
      const floating_point_type alpha, const floating_point_type* A,
      const int lda, const floating_point_type* B, const int ldb,
      const floating_point_type beta, floating_point_type* C, const int ldc);

};

}  // namespace oneflow
#endif // ONEFLOW_CORE_BLAS_KERNELUTIL_H__
