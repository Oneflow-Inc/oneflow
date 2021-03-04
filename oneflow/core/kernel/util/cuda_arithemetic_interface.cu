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
#include "oneflow/core/kernel/util/cuda_arithemetic_interface.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/util/host_arithemetic_interface.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

template<int32_t NDIMS>
struct Int32Array {
  int32_t val[NDIMS];
};

template<int32_t NDIMS>
__device__ int32_t GetXIndex(const int32_t* y_shape, const int32_t* x_strides, int32_t y_idx) {
  int32_t x_idx = 0;
  for (int32_t i = NDIMS - 1; i >= 0; --i) {
    const int32_t next_y_idx = y_idx / y_shape[i];
    x_idx += (y_idx - next_y_idx * y_shape[i]) * x_strides[i];
    y_idx = next_y_idx;
  }
  return x_idx;
}

template<int32_t NDIMS, typename T>
__global__ void TransposeGpu(const Int32Array<NDIMS> y_shape, const Int32Array<NDIMS> x_strides,
                             const int32_t elem_cnt, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(y_idx, elem_cnt) {
    const int32_t x_idx = GetXIndex<NDIMS>(y_shape.val, x_strides.val, y_idx);
#if __CUDA_ARCH__ >= 350
    y[y_idx] = __ldg(x + x_idx);
#else
    y[y_idx] = x[x_idx];
#endif
  }
}

template<int32_t NDIMS, typename T>
void TransposeImpl(DeviceCtx* ctx, const ShapeView& x_shape, const ShapeView& y_shape,
                   const std::vector<int32_t>& permutation, const int64_t elem_cnt, const T* x,
                   T* y) {
  CHECK_LE(y_shape.elem_cnt(), GetMaxVal<int32_t>());
  Int32Array<NDIMS> y_shape_struct;
  FOR_RANGE(int32_t, i, 0, NDIMS) { y_shape_struct.val[i] = y_shape.At(i); }
  Int32Array<NDIMS> x_strides;
  int32_t buff[NDIMS];
  int32_t cur_stride = 1;
  for (int32_t i = NDIMS - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= x_shape.At(i);
  }
  for (int32_t i = 0; i < NDIMS; ++i) { x_strides.val[i] = buff[permutation[i]]; }
  TransposeGpu<NDIMS, T>
      <<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          y_shape_struct, x_strides, elem_cnt, x, y);
}

template<typename T>
struct TransposeUtil final {
#define MAKE_TRANSPOSE_SWITCH_ENTRY(func_name, NDIMS) func_name<NDIMS, T>
  DEFINE_STATIC_SWITCH_FUNC(void, TransposeImpl, MAKE_TRANSPOSE_SWITCH_ENTRY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
};

}  // namespace

#define TRANSPOSE_CHECK                               \
  CHECK_LE(y_shape.elem_cnt(), GetMaxVal<int32_t>()); \
  CHECK_EQ(num_axis, y_shape.NumAxes());              \
  CHECK_EQ(num_axis, x_shape.NumAxes())

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<float>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                            permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<double>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                             permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const float16* x,
                                                float16* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<half>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                           elem_cnt, reinterpret_cast<const half*>(x),
                                           reinterpret_cast<half*>(y));
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int8_t* x,
                                                int8_t* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<int8_t>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                             permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int32_t* x,
                                                int32_t* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<int32_t>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                              permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const std::vector<int32_t>& permutation,
                                                const int64_t elem_cnt, const int64_t* x,
                                                int64_t* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<int64_t>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                              permutation, elem_cnt, x, y);
}

#undef TRANSPOSE_CHECK

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float16* x,
                                                float16* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int8_t* x,
                                                int8_t* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int32_t* x,
                                                int32_t* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const ShapeView& x_shape, const ShapeView& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const int64_t* x,
                                                int64_t* y) {
  ArithemeticIf<DeviceType::kGPU>::Transpose(
      ctx, num_axis, x_shape, y_shape,
      std::vector<int32_t>({permutation.cbegin(), permutation.cend()}), elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::InitializeWithConstConf(
    DeviceCtx* ctx, const ConstantInitializerConf& initializer_conf, Blob* blob) {
  WithHostBlobAndStreamSynchronizeEnv(ctx, blob, [&](Blob* host_blob) {
    ArithemeticIf<DeviceType::kCPU>::InitializeWithConstConf(nullptr, initializer_conf, host_blob);
  });
}

namespace {

template<typename T>
struct AddOp {
  __device__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct SubOp {
  __device__ T operator()(T a, T b) const { return a - b; }
};

template<typename T>
struct MulOp {
  __device__ T operator()(T a, T b) const { return a * b; }
};

template<typename T>
struct DivOp {
  __device__ T operator()(T a, T b) const { return a / b; }
};

template<template<typename> typename Op, typename T>
struct UnaryByScalarFunctor {
  __host__ __device__ explicit UnaryByScalarFunctor(T scalar) : scalar(scalar) {}
  __device__ T operator()(T a) const { return Op<T>()(a, scalar); }
  const T scalar;
};

template<template<typename> typename Op, typename T>
struct UnaryByScalarPtrFunctorFactory {
  __host__ __device__ explicit UnaryByScalarPtrFunctorFactory(const T* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ UnaryByScalarFunctor<Op, T> operator()() const {
    return UnaryByScalarFunctor<Op, T>(*scalar_ptr);
  }
  const T* scalar_ptr;
};

template<template<typename> typename Op, typename T>
void LaunchUnaryByScalar(DeviceCtx* ctx, const int64_t n, const T* x, const T y, T* z) {
  OF_CUDA_CHECK(
      (cuda::elementwise::Unary(UnaryByScalarFunctor<Op, T>(y), n, z, x, ctx->cuda_stream())));
}

template<template<typename> typename Op, typename T>
void LaunchUnaryByScalarPtr(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(UnaryByScalarPtrFunctorFactory<Op, T>(y), n, z,
                                                     x, ctx->cuda_stream())));
}

template<typename T>
__global__ void FillGpu(const int64_t n, const T value, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = value; }
}

template<typename T>
__global__ void CopyColsRegionGpu(const int64_t row_num, const int64_t col_num, const T* x,
                                  const int64_t x_col_offset, const int64_t x_lda, T* y,
                                  const int64_t y_col_offset, const int64_t y_lda) {
  CUDA_1D_KERNEL_LOOP(index, row_num * col_num) {
    const int64_t i = index / col_num;
    const int64_t j = index % col_num;
    y[i * y_lda + y_col_offset + j] = x[i * x_lda + x_col_offset + j];
  }
}

}  // namespace

#define OP_BY_SCALAR(op, T)                                                                       \
  void ArithemeticIf<DeviceType::kGPU>::op##ByScalar(DeviceCtx* ctx, const int64_t n, const T* x, \
                                                     const T y, T* z) {                           \
    LaunchUnaryByScalar<op##Op, T>(ctx, n, x, y, z);                                              \
  }

#define OP_BY_SCALAR_HALF(op)                                                                     \
  void ArithemeticIf<DeviceType::kGPU>::op##ByScalar(                                             \
      DeviceCtx* ctx, const int64_t n, const float16* x, const float16 y, float16* z) {           \
    LaunchUnaryByScalar<op##Op, half>(ctx, n, reinterpret_cast<const half*>(x), float16_2half(y), \
                                      reinterpret_cast<half*>(z));                                \
  }

#define DEFINE_OP_BY_SCALAR(op) \
  OP_BY_SCALAR(op, float)       \
  OP_BY_SCALAR(op, double)      \
  OP_BY_SCALAR(op, int8_t)      \
  OP_BY_SCALAR(op, int32_t)     \
  OP_BY_SCALAR(op, int64_t)     \
  OP_BY_SCALAR_HALF(op)

DEFINE_OP_BY_SCALAR(Mul)
DEFINE_OP_BY_SCALAR(Add)

#define OP_BY_SCALAR_PTR(op, T)                                                          \
  void ArithemeticIf<DeviceType::kGPU>::op##ByScalarPtr(DeviceCtx* ctx, const int64_t n, \
                                                        const T* x, const T* y, T* z) {  \
    LaunchUnaryByScalarPtr<op##Op, T>(ctx, n, x, y, z);                                  \
  }

#define OP_BY_SCALAR_PTR_HALF(op)                                                        \
  void ArithemeticIf<DeviceType::kGPU>::op##ByScalarPtr(                                 \
      DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y, float16* z) { \
    LaunchUnaryByScalarPtr<op##Op, half>(ctx, n, reinterpret_cast<const half*>(x),       \
                                         reinterpret_cast<const half*>(y),               \
                                         reinterpret_cast<half*>(z));                    \
  }

#define DEFINE_OP_BY_SCALAR_PTR(op) \
  OP_BY_SCALAR_PTR(op, float)       \
  OP_BY_SCALAR_PTR(op, double)      \
  OP_BY_SCALAR_PTR(op, int8_t)      \
  OP_BY_SCALAR_PTR(op, int32_t)     \
  OP_BY_SCALAR_PTR(op, int64_t)     \
  OP_BY_SCALAR_PTR_HALF(op)

DEFINE_OP_BY_SCALAR_PTR(Mul)
DEFINE_OP_BY_SCALAR_PTR(Add)
DEFINE_OP_BY_SCALAR_PTR(Sub)
DEFINE_OP_BY_SCALAR_PTR(Div)

#define FILL(T)                                                                              \
  void ArithemeticIf<DeviceType::kGPU>::Fill(DeviceCtx* ctx, const int64_t n, const T value, \
                                             T* y) {                                         \
    FillGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>( \
        n, value, y);                                                                        \
  }

FILL(float)
FILL(double)
FILL(int8_t)
FILL(int32_t)
FILL(int64_t)

#undef FILL

void ArithemeticIf<DeviceType::kGPU>::Fill(DeviceCtx* ctx, const int64_t n, const float16 value,
                                           float16* y) {
  FillGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, float16_2half(value), reinterpret_cast<half*>(y));
}

#define COPY_COLS_REGION(T)                                                                    \
  void ArithemeticIf<DeviceType::kGPU>::CopyColsRegion(                                        \
      DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x,                \
      const int64_t x_col_offset, const int64_t x_lda, T* y, const int64_t y_col_offset,       \
      const int64_t y_lda) {                                                                   \
    CopyColsRegionGpu<T><<<BlocksNum4ThreadsNum(row_num* col_num), kCudaThreadsNumPerBlock, 0, \
                           ctx->cuda_stream()>>>(row_num, col_num, x, x_col_offset, x_lda, y,  \
                                                 y_col_offset, y_lda);                         \
  }

COPY_COLS_REGION(float)
COPY_COLS_REGION(double)
COPY_COLS_REGION(int8_t)
COPY_COLS_REGION(int32_t)
COPY_COLS_REGION(int64_t)

#undef COPY_COLS_REGION

void ArithemeticIf<DeviceType::kGPU>::CopyColsRegion(DeviceCtx* ctx, const int64_t row_num,
                                                     const int64_t col_num, const float16* x,
                                                     const int64_t x_col_offset,
                                                     const int64_t x_lda, float16* y,
                                                     const int64_t y_col_offset,
                                                     const int64_t y_lda) {
  CopyColsRegionGpu<half>
      <<<BlocksNum4ThreadsNum(row_num * col_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          row_num, col_num, reinterpret_cast<const half*>(x), x_col_offset, x_lda,
          reinterpret_cast<half*>(y), y_col_offset, y_lda);
}

}  // namespace oneflow
