/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

class Blob;
class InitializerConf;
class MemCase;

size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num);

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemCase& dst_mem_case, const MemCase& src_mem_case);
void SyncAutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                    const MemCase& dst_mem_case, const MemCase& src_mem_case);
void AutoMemset(DeviceCtx* ctx, void* dst, const char value, size_t sz,
                const MemCase& dst_mem_case);

template<typename T>
OF_DEVICE_FUNC T ReduceCoreAdd(const T x, const T y) {
  return x + y;
}

template<typename T>
OF_DEVICE_FUNC T ReduceCoreMax(const T x, const T y) {
  return x > y ? x : y;
}

// CPU, GPU, Integral, Floating
template<DeviceType device_type, typename T, typename Derived>
struct KernelUtilIf {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    Derived::Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c,
                  ldc);
  }

  static void OFGemmTrans(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                          const T alpha, const T* a, const T* b, const T beta, T* c) {
    trans_a = (trans_a == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    trans_b = (trans_b == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    OFGemm(ctx, trans_b, trans_a, n, m, k, alpha, b, a, beta, c);
  }

  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
    const int m = c->shape().At(0);
    const int n = c->shape().Count(1);
    const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

    OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
  }

  static void OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const T alpha, const T* a, const T* b,
                            const T beta, T* c, T** buf) {
    Derived::BatchedGemm(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c, buf);
  }

  static void InitializeWithProperConf(DeviceCtx* ctx, const InitializerConf* initializer_conf,
                                       uint32_t random_seed, Blob* blob) {
    CHECK_NOTNULL(initializer_conf);
    Derived::InitializeWithConf(ctx, *initializer_conf, random_seed, blob);
  }
};

template<DeviceType device_type, typename T, typename U = void>
struct KernelUtil;

// CPU, Integral, Floating
template<typename T, typename Derived>
struct CpuKernelUtilIf {
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);
};

// CPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);

  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7, const T* in_8);

  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
};

// CPU, Integral
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
};

// GPU, Integral, Floating
template<typename T, typename Derived>
struct GpuKernelUtilIf {
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
};

// GPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);

  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7);
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7, const T* in_8);
};

// GPU, Integral
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
};

template<typename T, typename U>
typename std::enable_if<std::is_same<T, U>::value>::type CopyElem(const T* in_dptr, U* out_dptr,
                                                                  int64_t elem_num) {
  Memcpy<DeviceType::kCPU>(nullptr, out_dptr, in_dptr, elem_num * sizeof(T));
}

template<typename T, typename U>
typename std::enable_if<!std::is_same<T, U>::value>::type CopyElem(const T* in_dptr, U* out_dptr,
                                                                   int64_t elem_num) {
  FOR_RANGE(int64_t, i, 0, elem_num) { *(out_dptr++) = static_cast<U>(*(in_dptr++)); }
}

#ifdef WITH_CUDA
template<typename T, typename U>
void CopyElemOnGpu(DeviceCtx* ctx, const T* in_dptr, U* out_dptr, int64_t elem_num);
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
