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

void AutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void SyncAutoMemcpy(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void AutoMemset(DeviceCtx* ctx, void* dst, const char value, size_t sz,
                const MemoryCase& dst_mem_case);

template<DeviceType device_type, typename T, typename U = void>
struct KernelUtil;

// CPU, Integral, Floating
template<typename T, typename Derived>
struct CpuKernelUtilIf {
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void Set(DeviceCtx* ctx, const T value, T* addr);
};

// CPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public CpuKernelUtilIf<T, KernelUtil<DeviceType::kCPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);

  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y);

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
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
};

// GPU, Integral, Floating
template<typename T, typename Derived>
struct GpuKernelUtilIf {
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                             const int64_t y_col_offset, const int64_t y_lda);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void Set(DeviceCtx* ctx, const T value, T* addr);
};

// GPU, Floating
template<typename T>
struct KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public GpuKernelUtilIf<T, KernelUtil<DeviceType::kGPU, T>> {
  static void Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                  const int incy, T* result);
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy);
  static void Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx, T* y,
                   const int incy);

  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
  static void Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y);

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
  static void Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z);
};

using CopyBlobFieldMthd = void (Blob::*)(DeviceCtx*, const Blob*);

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
