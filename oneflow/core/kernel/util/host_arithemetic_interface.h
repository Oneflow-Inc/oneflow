#ifndef ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_

#include "oneflow/core/kernel/util/arithemetic_interface.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

class Blob;
class ConstantInitializerConf;

template<>
struct ArithemeticIf<DeviceType::kCPU> {
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float* x, float* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const double* x, double* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const int8_t* x, int8_t* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const int32_t* x, int32_t* y);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                        const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const int64_t* x, int64_t* y);

  static void InitializeWithConstConf(DeviceCtx* ctx,
                                      const ConstantInitializerConf& initializer_conf, Blob* blob);

  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const float* x, const float y, float* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const double* x, const double y,
                          double* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t y,
                          int32_t* z);
  static void MulByScalar(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t y,
                          int64_t* z);

  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                             float* z);
  static void MulByScalarPtr(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                             double* z);
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
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int8_t* x, const int8_t* y,
                             int8_t* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int32_t* x, const int32_t* y,
                             int32_t* z);
  static void DivByScalarPtr(DeviceCtx* ctx, const int64_t n, const int64_t* x, const int64_t* y,
                             int64_t* z);

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

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CPU_ARITHEMETIC_INTERFACE_H_
