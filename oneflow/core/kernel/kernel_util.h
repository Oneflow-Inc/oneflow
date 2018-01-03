#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/blas/cblas_template.h"
#include "oneflow/core/blas/cublas_template.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<DeviceType device_type>
struct GetCudaMemcpyKind;

template<>
struct GetCudaMemcpyKind<DeviceType::kCPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyHostToHost;
};

template<>
struct GetCudaMemcpyKind<DeviceType::kGPU> {
  static const cudaMemcpyKind val = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

template<DeviceType device_type>
void Memcpy(DeviceCtx*, void* dst, const void* src, size_t sz,
            cudaMemcpyKind kind = GetCudaMemcpyKind<device_type>::val);

template<DeviceType device_type>
void Memset(DeviceCtx*, void* dst, const char value, size_t sz);

template<DeviceType device_type, typename T>
struct KernelUtil final {
  // dot product
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx,
                  const T* y, const int incy, T* result);

  // copy x into y
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx,
                   T* y, const int incy);

  // y = a*x + y
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x,
                   const int incx, T* y, const int incy);

  // x = a*x
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x,
                   const int incx);
  // max(x) only cpu
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr);

  // max(x) temp_storage is for gpu
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr,
                  T* temp_storage, size_t temp_storage_bytes);

  // y = exp(x)
  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y);

  // sum(x), only cpu
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr);

  // sum(x) temp_storage is for gpu parallel
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr,
                  T* temp_storage, size_t temp_storage_bytes);

  // x = x / a
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha);

  // z[i] = x[i] * y[i]
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                  T* z);

  // matrix vector multiply
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m,
                   int n, const T alpha, const T* a, int lda, const T* x,
                   const int incx, const T beta, T* y, const int incy);

  // matrix matrix multiply
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                   const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                   const int k, const T alpha, const T* a, const int lda,
                   const T* b, const int ldb, const T beta, T* c,
                   const int ldc);

  // Generate random number of specific distribution
  static void Fill(DeviceCtx* ctx, const FillConf& fill_conf,
                   uint32_t random_seed, Blob* blob);

  // detect fill conf
  static void FillWithProperConf(DeviceCtx* ctx, const FillConf* fill_conf,
                                 uint32_t random_seed, Blob* blob) {
    if (fill_conf == nullptr) {
      fill_conf = JobDesc::Singleton()->DefaultFillConf();
    }
    Fill(ctx, *fill_conf, random_seed, blob);
  }

  // fill blob with model dir
  static void FillWithModelDir(DeviceCtx* ctx, int32_t part_id,
                               int32_t part_num, const std::string& model_dir,
                               Blob* blob, const std::string& bn_in_op,
                               int32_t dim_num, int64_t num_in_each_dim);
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
