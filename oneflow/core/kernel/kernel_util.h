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
class MemoryCase;

size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num);

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void SyncAutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);

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
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr);
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                     T* y);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
    RowMax(ctx, row_num, col_num, x, y);
  }
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,
                     T* y);
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
    RowSum(ctx, row_num, col_num, x, y);
  }
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void Set(DeviceCtx* ctx, const T value, T* addr);
  static void Replicate(DeviceCtx* ctx, const int64_t n, T* y, const T* x);
};

// CPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public KernelUtilIf<DeviceType::kCPU, T, KernelUtil<DeviceType::kCPU, T>>,
      public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y, const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx);
  static void Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx);
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n, const T alpha,
                   const T* a, int lda, const T* x, const int incx, const T beta, T* y,
                   const int incy);
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc);
  static void BatchedGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                          const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                          int batch_size, int m, int n, int k, const T alpha, const T* a,
                          const T* b, const T beta, T* c, T** buf);

  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T alpha);
  static void Div(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Reciprocal(DeviceCtx* ctx, const int n, const T* x, T* y);
  static void Square(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y, const float epsilon);
  static void Powx(DeviceCtx* ctx, const int64_t n, const T* x, const float power, T* y);

  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx);
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);

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
  static void Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                       const T* in_2, const T* in_3, const T* in_4, const T* in_5, const T* in_6,
                       const T* in_7, const T* in_8, const T* in_9);

  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
};

// CPU, Integral
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public KernelUtilIf<DeviceType::kCPU, T, KernelUtil<DeviceType::kCPU, T>>,
      public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
};

// GPU, Integral, Floating
template<typename T, typename Derived>
struct GpuKernelUtilIf {
  static void Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                  size_t temp_storage_bytes);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes);
  static void RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void Set(DeviceCtx* ctx, const T value, T* addr);
  static void Replicate(DeviceCtx* ctx, const int64_t n, T* y, const T* x);
};

// GPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public KernelUtilIf<DeviceType::kGPU, T, KernelUtil<DeviceType::kGPU, T>>,
      public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y, const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx);
  static void Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx);
  static void Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n, const T alpha,
                   const T* a, int lda, const T* x, const int incx, const T beta, T* y,
                   const int incy);
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc);
  static void BatchedGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                          const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                          int batch_size, int m, int n, int k, const T alpha, const T* a,
                          const T* b, const T beta, T* c, T** buf);

  static void Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha);
  static void Div(DeviceCtx* ctx, const int64_t n, T* x, const T alpha);
  static void Div(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Reciprocal(DeviceCtx* ctx, const int n, const T* x, T* y);
  static void Square(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon);
  static void Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y, const float epsilon);

  static void Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx);
  static void Relu(DeviceCtx* ctx, int64_t n, const T* x, T* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);

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
    : public KernelUtilIf<DeviceType::kGPU, T, KernelUtil<DeviceType::kGPU, T>>,
      public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
};

using CopyBlobFieldMthd = void (Blob::*)(DeviceCtx*, const Blob*);

template<DeviceType device_type, typename Iter>
void CopyFromIterToIter(DeviceCtx* ctx, Iter& src_it, Iter& dst_it) {
  const char* src_ptr = nullptr;
  size_t src_size = 0;
  char* dst_ptr = nullptr;
  size_t dst_size = 0;
  while (true) {
    if (src_size == 0) { std::tie(src_ptr, src_size) = src_it.GetNext(); }
    if (dst_size == 0) { std::tie(dst_ptr, dst_size) = dst_it.GetNext(); }
    if (src_size == 0) {
      CHECK_EQ(src_size, dst_size);
      break;
    }
    size_t cp_size = std::min(src_size, dst_size);
    if (dst_ptr != nullptr) {
      if (src_ptr != nullptr) {
        Memcpy<device_type>(ctx, dst_ptr, src_ptr, cp_size);
      } else {
        Memset<device_type>(ctx, dst_ptr, 0, cp_size);
      }
      dst_ptr += cp_size;
    }
    if (src_ptr != nullptr) { src_ptr += cp_size; }
    src_size -= cp_size;
    dst_size -= cp_size;
  }
}

class DataContentIterator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataContentIterator);
  DataContentIterator() = delete;
  ~DataContentIterator() = default;

  DataContentIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                      const PbRpf<std::string>* bns, int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    seg_num_ = BnInOp2Blob(bns->Get(0))->static_shape().Count(0, axis);
    seg_idx_ = 0;
    bns_ = bns;
    bn_idx_ = 0;
    axis_ = axis;
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (seg_idx_ == seg_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_));
    int64_t elem_num = blob->static_shape().Count(axis_);
    std::get<1>(ret) = elem_num * GetSizeOfDataType(blob->data_type());
    std::get<0>(ret) = blob->mut_dptr<char>() + seg_idx_ * std::get<1>(ret);
    bn_idx_ += 1;
    if (bn_idx_ == bns_->size()) {
      bn_idx_ = 0;
      seg_idx_ += 1;
    }
    return ret;
  }

  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDataContentFrom; }

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  int64_t seg_num_;
  int64_t seg_idx_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t axis_;
};

/*
class FieldIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FieldIterator);
  FieldIterator() = delete;
  virtual ~FieldIterator() = default;

  FieldIterator(std::function<Blob*(const std::string&)> BnInOp2Blob, const PbRpf<std::string>* bns,
                int32_t axis) {
    BnInOp2Blob_ = BnInOp2Blob;
    bns_ = bns;
    bn_idx_ = 0;
    if (axis == 0) {
      bn_num_ = bns_->size();
    } else {
      bn_num_ = 1;
    }
  }

  std::tuple<char*, size_t> GetNext() {
    std::tuple<char*, size_t> ret(nullptr, 0);
    if (bn_idx_ == bn_num_) { return ret; }
    Blob* blob = BnInOp2Blob_(bns_->Get(bn_idx_++));
    std::get<0>(ret) = GetMutPtr(blob);
    std::get<1>(ret) = GetSizeOfField(blob);
    return ret;
  }

 protected:
  virtual char* GetMutPtr(Blob* blob) = 0;
  virtual size_t GetSizeOfField(Blob* blob) const = 0;

 private:
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  const PbRpf<std::string>* bns_;
  int32_t bn_idx_;
  int32_t bn_num_;
};

class DataIdIterator final : public FieldIterator {
 public:
  DataIdIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDataIdFrom; }

 private:
  char* GetMutPtr(Blob* blob) override { return blob->mut_data_id(); }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfDataIdField(); }
};

class ColNumIterator final : public FieldIterator {
 public:
  ColNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                 const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyColNumFrom; }

 private:
  char* GetMutPtr(Blob* blob) override { return reinterpret_cast<char*>(blob->mut_col_num()); }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfColNumField(); }
};

class Dim1ValidNumIterator final : public FieldIterator {
 public:
  Dim1ValidNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                       const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDim1ValidNumFrom; }

 private:
  char* GetMutPtr(Blob* blob) override {
    return reinterpret_cast<char*>(blob->mut_dim1_valid_num_ptr());
  }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfDim1ValidNumField(); }
};

class Dim2ValidNumIterator final : public FieldIterator {
 public:
  Dim2ValidNumIterator(std::function<Blob*(const std::string&)> BnInOp2Blob,
                       const PbRpf<std::string>* bns, int32_t axis)
      : FieldIterator(BnInOp2Blob, bns, axis) {}
  static CopyBlobFieldMthd GetCopyBlobFieldMthd() { return &Blob::CopyDim2ValidNumFrom; }

 private:
  char* GetMutPtr(Blob* blob) override {
    return reinterpret_cast<char*>(blob->mut_dim2_valid_num_ptr());
  }

  size_t GetSizeOfField(Blob* blob) const override { return blob->ByteSizeOfDim2ValidNumField(); }
};
*/

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
