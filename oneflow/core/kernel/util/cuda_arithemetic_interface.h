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
#ifdef WITH_CUDA

#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_

#include "oneflow/core/kernel/util/arithemetic_interface.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class Blob;
class ConstantInitializerConf;

template<>
struct ArithemeticIf<DeviceType::kGPU> {
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const float* x, float* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const double* x, double* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const float16* x, float16* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const int8_t* x, int8_t* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const int32_t* x, int32_t* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                        int64_t elem_cnt, const int64_t* x, int64_t* y);

  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const float* x, float* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const double* x, double* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const float16* x, float16* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const int8_t* x, int8_t* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const int32_t* x, int32_t* y);
  static void Transpose(DeviceCtx* ctx, int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        int64_t elem_cnt, const int64_t* x, int64_t* y);

  static void InitializeWithConstConf(DeviceCtx* ctx,
                                      const ConstantInitializerConf& initializer_conf, Blob* blob);

  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const float* x, const float y, float* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const double* x, const double y,
                          double* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const float16* x, const float16 y,
                          float16* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t y,
                          int8_t* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t y,
                          int32_t* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t y,
                          int64_t* z);

  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const float* x, const float y, float* z);
  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const double* x, const double y,
                          double* z);
  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const float16* x, const float16 y,
                          float16* z);
  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t y,
                          int8_t* z);
  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t y,
                          int32_t* z);
  static void AddByScalar(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t y,
                          int64_t* z);

  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                             float* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                             double* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                             float16* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t* y,
                             int8_t* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t* y,
                             int32_t* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t* y,
                             int64_t* z);

  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                             float* z);
  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                             double* z);
  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                             float16* z);
  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t* y,
                             int8_t* z);
  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t* y,
                             int32_t* z);
  static void AddByScalarPtr(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t* y,
                             int64_t* z);

  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                             float* z);
  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                             double* z);
  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                             float16* z);
  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t* y,
                             int8_t* z);
  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t* y,
                             int32_t* z);
  static void SubByScalarPtr(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t* y,
                             int64_t* z);

  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                             float* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                             double* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                             float16* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t* y,
                             int8_t* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t* y,
                             int32_t* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t* y,
                             int64_t* z);

  static void Fill(DeviceCtx* ctx, const int64_t n, const float16 value, float16* y);
  static void Fill(DeviceCtx* ctx, const int64_t n, const float value, float* y);
  static void Fill(DeviceCtx* ctx, const int64_t n, const double value, double* y);
  static void Fill(DeviceCtx* ctx, const int64_t n, const int8_t value, int8_t* y);
  static void Fill(DeviceCtx* ctx, const int64_t n, const int32_t value, int32_t* y);
  static void Fill(DeviceCtx* ctx, const int64_t n, const int64_t value, int64_t* y);

  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const float* x, const int64_t x_col_offset, const int64_t x_lda,
                             float* y, const int64_t y_col_offset, const int64_t y_lda);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const double* x, const int64_t x_col_offset, const int64_t x_lda,
                             double* y, const int64_t y_col_offset, const int64_t y_lda);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const float16* x, const int64_t x_col_offset, const int64_t x_lda,
                             float16* y, const int64_t y_col_offset, const int64_t y_lda);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const int8_t* x, const int64_t x_col_offset, const int64_t x_lda,
                             int8_t* y, const int64_t y_col_offset, const int64_t y_lda);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const int32_t* x, const int64_t x_col_offset, const int64_t x_lda,
                             int32_t* y, const int64_t y_col_offset, const int64_t y_lda);
  static void CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                             const int64_t* x, const int64_t x_col_offset, const int64_t x_lda,
                             int64_t* y, const int64_t y_col_offset, const int64_t y_lda);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_ARITHEMETIC_INTERFACE_H_

#endif
