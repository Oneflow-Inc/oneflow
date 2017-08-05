#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/blas/cblas_template.h"
#include "oneflow/core/blas/cublas_template.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class KernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelUtil);
  KernelUtil() = delete;

  // memory copy
  static void Memcpy(
      const KernelCtx& ctx, void* dst, const void* src, size_t sz,
      cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToHost);

  // memory set
  static void Memset(const KernelCtx& ctx, void* dst, const char value,
                     size_t sz);

  // level 1 vector and vector
  // dot product
  static void BlasDot(const KernelCtx& ctx, const int n,
                      const FloatingPointType* x, const int incx,
                      const FloatingPointType* y, const int incy,
                      FloatingPointType* result);

  // swap x and y
  static void BlasSwap(const KernelCtx& ctx, const int n, FloatingPointType* x,
                       const int incx, FloatingPointType* y, const int incy);

  // copy x into y
  static void BlasCopy(const KernelCtx& ctx, const int n,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy);

  // y = a*x + y
  static void BlasAxpy(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha,
                       const FloatingPointType* x, const int incx,
                       FloatingPointType* y, const int incy);

  // x = a*x
  static void BlasScal(const KernelCtx& ctx, const int n,
                       const FloatingPointType alpha, FloatingPointType* x,
                       const int incx);
  // max(x)
  // no Template specialization for gpu
  static void Max(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* max_ptr);

  // max(x)
  // temp_storage is for gpu parallel
  static void Max(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* max_ptr,
                  FloatingPointType* temp_storage, size_t temp_storage_bytes);

  // y = exp(x)
  static void Exp(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* y);

  // sum(x)
  // no Template specialization for gpu
  static void Sum(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* sum_ptr);

  // sum(x)
  // temp_storage is for gpu parallel
  static void Sum(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, FloatingPointType* sum_ptr,
                  FloatingPointType* temp_storage, size_t temp_storage_bytes);

  // x = x / a
  static void Div(const KernelCtx& ctx, const int64_t n, FloatingPointType* x,
                  const FloatingPointType* alpha_ptr);

  // element-wise multiplication
  // z[i] = x[i] * y[i]
  static void Mul(const KernelCtx& ctx, const int64_t n,
                  const FloatingPointType* x, const FloatingPointType* y,
                  FloatingPointType* z);

  // level 2 matrix and vector
  // matrix vector multiply
  static void BlasGemv(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans,
                       int m, int n, const FloatingPointType alpha,
                       const FloatingPointType* a, int lda,
                       const FloatingPointType* x, const int incx,
                       const FloatingPointType beta, FloatingPointType* y,
                       const int incy);

  // level 3 matrix and matrix
  // matrix matrix multiply
  static void BlasGemm(const KernelCtx& ctx, const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a,
                       const enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const FloatingPointType alpha,
                       const FloatingPointType* a, const int lda,
                       const FloatingPointType* b, const int ldb,
                       const FloatingPointType beta, FloatingPointType* c,
                       const int ldc);

  // Generate random number of specific distribution
  static void Fill(const FillConf& fill_conf, uint32_t random_seed, Blob* blob);
  static void Fill(const KernelCtx& ctx, const FillConf& fill_conf,
                   uint32_t random_seed, Blob* blob);

  // detect fill conf
  static void FillWithProperConf(const KernelCtx& ctx,
                                 const FillConf* fill_conf,
                                 uint32_t random_seed, Blob* blob) {
    if (fill_conf == nullptr) {
      fill_conf = JobDesc::Singleton()->default_fill_conf();
    }
    Fill(ctx, *fill_conf, random_seed, blob);
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
