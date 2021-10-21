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
#include "oneflow/core/kernel/util/host_blas_interface.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace {

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                 enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                 const double alpha, const T* a, const T* b, const double beta, T* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cblas_gemm<T>(order, trans_a, trans_b, m, n, k, static_cast<T>(alpha), a, lda, b, ldb,
                static_cast<T>(beta), c, ldc);
}

template<typename T>
static void AxpyImpl(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                     const int incy) {
  FOR_RANGE(int, i, 0, n) {
    *y += alpha * *x;
    x += incx;
    y += incy;
  }
}

template<typename T>
void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                     int batch_size, int m, int n, int k, const double alpha, const T* a,
                     const T* b, const double beta, T* c) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  FOR_RANGE(int32_t, i, 0, batch_size) {
    BlasIf<DeviceType::kCPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a + i * a_stride,
                                     b + i * b_stride, beta, c + i * c_stride);
  }
}

}  // namespace

void BlasIf<DeviceType::kCPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const float* a,
                                      const float* b, const double beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void BlasIf<DeviceType::kCPU>::OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                      enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                                      const int k, const double alpha, const double* a,
                                      const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

void BlasIf<DeviceType::kCPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const float* a, const float* b,
                                             const double beta, float* c) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c);
}

void BlasIf<DeviceType::kCPU>::OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                             enum CBLAS_TRANSPOSE trans_b, const int batch_size,
                                             const int m, const int n, const int k,
                                             const double alpha, const double* a, const double* b,
                                             const double beta, double* c) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                          beta, c);
}

void BlasIf<DeviceType::kCPU>::Axpy(DeviceCtx* ctx, const int n, const float alpha, const float* x,
                                    const int incx, float* y, const int incy) {
  AxpyImpl<float>(ctx, n, alpha, x, incx, y, incy);
}

void BlasIf<DeviceType::kCPU>::Axpy(DeviceCtx* ctx, const int n, const double alpha,
                                    const double* x, const int incx, double* y, const int incy) {
  AxpyImpl<double>(ctx, n, alpha, x, incx, y, incy);
}

}  // namespace oneflow
