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
#include "oneflow/core/kernel/util/host_arithemetic_interface.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

namespace {

void ComputeOffset(const int32_t num_axes, const int64_t* shape, const int32_t* permutation,
                   DimVector& offset) {
  offset.resize(num_axes);
  DimVector buff(num_axes);
  int64_t cur_offset = 1;
  for (int32_t i = num_axes - 1; i >= 0; --i) {
    buff[i] = cur_offset;
    cur_offset *= shape[i];
  }
  for (int32_t i = 0; i < num_axes; ++i) { offset[permutation[i]] = buff[i]; }
}

void IncreaseIndex(const int64_t* shape, DimVector& index) {
  for (int32_t i = index.size() - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= shape[i]) {
      index[i] -= shape[i];
    } else {
      break;
    }
  }
}

template<typename T>
void TransposeImpl(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                   const ShapeView& y_shape, const std::vector<int32_t>& permutation,
                   const int64_t elem_cnt, const T* x, T* y) {
  int64_t block_size = 1;
  int32_t shared_idxs_num = 0;
  for (int32_t i = num_axis - 1; i >= 0 && permutation[i] == i; --i) {
    block_size *= y_shape.At(i);
    ++shared_idxs_num;
  }
  if (num_axis < 2 || shared_idxs_num == num_axis) {
    memcpy(y, x, elem_cnt * sizeof(T));
    return;
  }
  int32_t trans_axis = num_axis - shared_idxs_num;
  DimVector x_to_y_offset;
  ComputeOffset(trans_axis, y_shape.ptr(), permutation.data(), x_to_y_offset);
  DimVector x_index_digits(trans_axis, 0);
  int64_t num_blocks = elem_cnt / block_size;
  FOR_RANGE(int64_t, x_idx, 0, num_blocks) {
    int64_t y_idx = std::inner_product(x_to_y_offset.cbegin(), x_to_y_offset.cend(),
                                       x_index_digits.cbegin(), 0);
    if (block_size == 1) {
      y[y_idx] = x[x_idx];
    } else {
      memcpy(y + block_size * y_idx, x + block_size * x_idx, block_size * sizeof(T));
    }
    IncreaseIndex(x_shape.ptr(), x_index_digits);
  }
}

template<typename T>
void ConstantInitializer(const T& value, Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  CHECK(elem_cnt);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
}

}  // namespace

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  TransposeImpl<float>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  TransposeImpl<double>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int8_t* x,
                                                int8_t* y) {
  TransposeImpl<int8_t>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int32_t* x,
                                                int32_t* y) {
  TransposeImpl<int32_t>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int64_t* x,
                                                int64_t* y) {
  TransposeImpl<int64_t>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  TransposeImpl<float>(ctx, num_axis, x_shape, y_shape,
                       std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt,
                       x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  TransposeImpl<double>(ctx, num_axis, x_shape, y_shape,
                        std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt,
                        x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int8_t* x,
                                                int8_t* y) {
  TransposeImpl<int8_t>(ctx, num_axis, x_shape, y_shape,
                        std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt,
                        x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int32_t* x,
                                                int32_t* y) {
  TransposeImpl<int32_t>(ctx, num_axis, x_shape, y_shape,
                         std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt,
                         x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int64_t* x,
                                                int64_t* y) {
  TransposeImpl<int64_t>(ctx, num_axis, x_shape, y_shape,
                         std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt,
                         x, y);
}

void ArithemeticIf<DeviceType::kCPU>::InitializeWithConstConf(
    DeviceCtx* ctx, const ConstantInitializerConf& initializer_conf, Blob* blob) {
  DataType dtype = blob->data_type();
  if (dtype == DataType::kFloat) {
    ConstantInitializer<float>(initializer_conf.value(), blob);
  } else if (dtype == DataType::kDouble) {
    ConstantInitializer<double>(static_cast<double>(initializer_conf.value()), blob);
  } else if (dtype == DataType::kFloat16) {
    ConstantInitializer<float16>(static_cast<float16>(initializer_conf.value()), blob);
  } else {
    UNIMPLEMENTED();
  }
}

#define MUL_BY_SCALAR(T)                                                                         \
  void ArithemeticIf<DeviceType::kCPU>::MulByScalar(DeviceCtx* ctx, const int64_t n, const T* x, \
                                                    const T y, T* z) {                           \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y; }                                         \
  }

MUL_BY_SCALAR(float);
MUL_BY_SCALAR(double);
MUL_BY_SCALAR(int8_t);
MUL_BY_SCALAR(int32_t);
MUL_BY_SCALAR(int64_t);

#undef MUL_BY_SCALAR

#define ADD_BY_SCALAR(T)                                                                         \
  void ArithemeticIf<DeviceType::kCPU>::AddByScalar(DeviceCtx* ctx, const int64_t n, const T* x, \
                                                    const T y, T* z) {                           \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] + y; }                                         \
  }

ADD_BY_SCALAR(float);
ADD_BY_SCALAR(double);
ADD_BY_SCALAR(int8_t);
ADD_BY_SCALAR(int32_t);
ADD_BY_SCALAR(int64_t);

#undef ADD_BY_SCALAR

#define MUL_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::MulByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] * y[0]; }                             \
  }

MUL_BY_SCALAR_PTR(float);
MUL_BY_SCALAR_PTR(double);
MUL_BY_SCALAR_PTR(int8_t);
MUL_BY_SCALAR_PTR(int32_t);
MUL_BY_SCALAR_PTR(int64_t);

#undef MUL_BY_SCALAR_PTR

#define ADD_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::AddByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] + y[0]; }                             \
  }

ADD_BY_SCALAR_PTR(float);
ADD_BY_SCALAR_PTR(double);
ADD_BY_SCALAR_PTR(int8_t);
ADD_BY_SCALAR_PTR(int32_t);
ADD_BY_SCALAR_PTR(int64_t);

#undef ADD_BY_SCALAR_PTR

#define SUB_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::SubByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] - y[0]; }                             \
  }

SUB_BY_SCALAR_PTR(float);
SUB_BY_SCALAR_PTR(double);
SUB_BY_SCALAR_PTR(int8_t);
SUB_BY_SCALAR_PTR(int32_t);
SUB_BY_SCALAR_PTR(int64_t);

#undef SUB_BY_SCALAR_PTR

#define DIV_BY_SCALAR_PTR(T)                                                            \
  void ArithemeticIf<DeviceType::kCPU>::DivByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                       const T* x, const T* y, T* z) {  \
    for (int64_t i = 0; i < n; ++i) { z[i] = x[i] / y[0]; }                             \
  }

DIV_BY_SCALAR_PTR(float);
DIV_BY_SCALAR_PTR(double);
DIV_BY_SCALAR_PTR(int8_t);
DIV_BY_SCALAR_PTR(int32_t);
DIV_BY_SCALAR_PTR(int64_t);

#undef DIV_BY_SCALAR_PTR

#define FILL(T)                                                                              \
  void ArithemeticIf<DeviceType::kCPU>::Fill(DeviceCtx* ctx, const int64_t n, const T value, \
                                             T* y) {                                         \
    std::fill_n(y, n, value);                                                                \
  }

FILL(float);
FILL(double);
FILL(int8_t);
FILL(int32_t);
FILL(int64_t);

#undef FILL

#define COPY_COLS_REGION(T)                                                              \
  void ArithemeticIf<DeviceType::kCPU>::CopyColsRegion(                                  \
      DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,          \
      const int64_t x_col_offset, const int64_t x_lda, T* y, const int64_t y_col_offset, \
      const int64_t y_lda) {                                                             \
    for (int64_t i = 0; i < row_num; ++i) {                                              \
      for (int64_t j = 0; j < col_num; ++j) {                                            \
        y[i * y_lda + y_col_offset + j] = x[i * x_lda + x_col_offset + j];               \
      }                                                                                  \
    }                                                                                    \
  }

COPY_COLS_REGION(float)
COPY_COLS_REGION(double)
COPY_COLS_REGION(int8_t)
COPY_COLS_REGION(int32_t)
COPY_COLS_REGION(int64_t)

#undef COPY_COLS_REGION

}  // namespace oneflow
