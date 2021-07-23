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
#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void RsqrtGpu(const int64_t n, T* x, const float epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}

template<typename T>
__global__ void RsqrtGpu(const int64_t n, const T* x, T* y, const float epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}

template<typename T>
__global__ void ExpGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::exp(x[i]); }
}

template<typename T>
__global__ void DivByConstParaPtrGpu(const int64_t n, T* x, const T* alpha_ptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / (*alpha_ptr); }
}

template<typename T>
__global__ void DivGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] / y[i]; }
}

template<typename T>
__global__ void DivByConstParaGpu(const int64_t n, T* x, const T alpha) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / alpha; }
}

template<typename T>
__global__ void ReplicateGpu(const int64_t n, T* y, const T* x) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = *x; }
}

template<typename T>
__global__ void MulGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[i]; }
}

template<typename T>
__global__ void ReciprocalGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = static_cast<T>(1.0) / x[i]; }
}

template<typename T>
__global__ void SquareGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * x[i]; }
}

template<typename T>
__global__ void SqrtGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::sqrt(x[i]); }
}

template<typename T>
__global__ void AxpyGpu(const int n, const T alpha, const T* x, const int incx, T* y,
                        const int incy) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] += alpha * x[i * incx]; }
}

template<typename T>
__global__ void SigmoidForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.0 / (1.0 + std::exp(-x[i])); }
}

template<typename T>
__global__ void SigmoidBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * y[i] * (1.0 - y[i]); }
}

template<typename T>
__global__ void ReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

template<typename T>
__global__ void ReluBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > 0 ? dy[i] : 0; }
}

template<typename T>
__global__ void gpu_assign_add(const int64_t n, T* out, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (in_1[i]) { out[i] += in_1[i]; }
  }
}

template<typename T>
__global__ void gpu_assign_add(const int64_t n, T* out, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] += in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i]; }
}
template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7,
                        const T* in_8) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] =
        in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i] + in_8[i];
  }
}

template<typename T>
__global__ void gpu_set(const T value, T* addr) {
  *addr = value;
}

cublasOperation_t CblasTrans2CublasTrans(CBLAS_TRANSPOSE trans) {
  cublasOperation_t cublas_trans;
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_N;
  } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_T;
  } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_C;
  } else {
    // do nothing
  }
  return cublas_trans;
}

template<int32_t NDIMS>
struct Int32Array {
  int32_t val[NDIMS];
};

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

template<int32_t NDIMS>
__device__ int32_t GetXIndex(const int32_t* y_shape, const int32_t* x_strides, int32_t y_idx) {
  int32_t x_idx = 0;
  for (int32_t i = NDIMS - 1; i >= 0; --i) {
    x_idx += (y_idx % y_shape[i]) * x_strides[i];
    y_idx /= y_shape[i];
  }
  return x_idx;
}

template<int32_t NDIMS, typename T>
__global__ void TransposeGpu(const Int32Array<NDIMS> y_shape, const Int32Array<NDIMS> x_strides,
                             const int32_t elem_cnt, const T* x, T* y) {
  __shared__ int32_t x_strides_shared[NDIMS];
  __shared__ int32_t y_dims_shared[NDIMS];
  const int32_t tid = threadIdx.x;
  if (tid < NDIMS) {
    y_dims_shared[tid] = y_shape.val[tid];
    x_strides_shared[tid] = x_strides.val[tid];
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(y_idx, elem_cnt) {
    const int32_t x_idx = GetXIndex<NDIMS>(y_dims_shared, x_strides_shared, y_idx);
#if __CUDA_ARCH__ >= 350
    y[y_idx] = __ldg(x + x_idx);
#else
    y[y_idx] = x[x_idx];
#endif
  }
}

template<int32_t NDIMS, typename T>
void Transpose(DeviceCtx* ctx, const ShapeView& x_shape, const ShapeView& y_shape,
               const PbRf<int32_t>& permutation, const int64_t elem_cnt, const T* x, T* y) {
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
  DEFINE_STATIC_SWITCH_FUNC(void, Transpose, MAKE_TRANSPOSE_SWITCH_ENTRY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
};

template<typename T, T (*reduce_core_func)(const T, const T)>
__device__ void MatrixShrinkCols(const size_t row_num, const size_t thread_col_num, const T* x,
                                 const size_t x_col_num, const size_t x_lda, T* y,
                                 const size_t y_col_num, const size_t y_lda) {
  const size_t thread_num = blockDim.x * gridDim.x;
  const size_t total_shrink_scale = thread_col_num / y_col_num;
  CUDA_1D_KERNEL_LOOP(index, row_num * thread_col_num) {
    const int32_t thread_col = index % thread_col_num;
    if (((index / thread_num) % total_shrink_scale) != thread_col / y_col_num) { continue; }
    const int32_t row = index / thread_col_num;
    const int32_t col = thread_col % y_col_num;
    const int32_t x_start = row * x_lda + col;
    const int32_t x_end = row * x_lda + x_col_num;
    T reduced = x[x_start];
    for (int32_t x_index = x_start + y_col_num; x_index < x_end; x_index += y_col_num) {
      reduced = reduce_core_func(reduced, x[x_index]);
    }
    y[row * y_lda + col] = reduced;
  }
}

template<typename T, T (*reduce_core_func)(const T, const T), size_t shift_size = 2>
__global__ void MatrixRowReduceGpu(const size_t row_num, const size_t col_num, const T* x, T* y,
                                   T* temp_storage, size_t temp_col_num) {
  const size_t temp_lda = temp_col_num;
  MatrixShrinkCols<T, reduce_core_func>(row_num, temp_lda, x, col_num, col_num, temp_storage,
                                        temp_col_num, temp_lda);
  __syncthreads();
  while (temp_col_num > (1 << shift_size)) {
    size_t new_temp_col_num = temp_col_num >> shift_size;
    MatrixShrinkCols<T, reduce_core_func>(row_num, temp_lda, temp_storage, temp_col_num, temp_lda,
                                          temp_storage, new_temp_col_num, temp_lda);
    temp_col_num = new_temp_col_num;
    __syncthreads();
  }
  MatrixShrinkCols<T, reduce_core_func>(row_num, temp_lda, temp_storage, temp_col_num, temp_lda, y,
                                        1, 1);
}

template<typename T, T (*reduce_core_func)(const T, const T), size_t shift_size = 2>
void MatrixRowReduce(DeviceCtx* ctx, const size_t row_num, const size_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
  CHECK_NOTNULL(temp_storage);
  CHECK_GT(temp_storage_bytes / sizeof(T), row_num);
  const size_t temp_col_num_shift =
      std::floor(std::log2(std::min(temp_storage_bytes / sizeof(T) / row_num, col_num)));
  const size_t temp_col_num = std::min(static_cast<size_t>(kCudaThreadsNumPerBlock),
                                       static_cast<size_t>(1 << temp_col_num_shift));
  MatrixRowReduceGpu<T, reduce_core_func>
      <<<BlocksNum4ThreadsNum(row_num * temp_col_num), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(row_num, col_num, x, y, static_cast<T*>(temp_storage), temp_col_num);
}

template<typename T>
__global__ void AssignStridedAddrGpu(T** dev_ptrs, T* start_ptr, int32_t stride_len,
                                     int32_t stride_num) {
  CUDA_1D_KERNEL_LOOP(i, stride_num) { dev_ptrs[i] = start_ptr + i * stride_len; }
}

template<typename T>
void AssignStridedAddr(DeviceCtx* ctx, T** dev_ptrs, T* start_ptr, int stride_len, int stride_num) {
  AssignStridedAddrGpu<T>
      <<<BlocksNum4ThreadsNum(stride_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          dev_ptrs, start_ptr, stride_len, stride_num);
}

}  // namespace

#define MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY(func_name, T) cub::DeviceReduce::func_name<T*, T*>
DEFINE_STATIC_SWITCH_FUNC(cudaError_t, Sum, MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num) {
  size_t tmp_storage_size;
  SwitchSum(SwitchCase(data_type), nullptr, tmp_storage_size, nullptr, nullptr, sum_elem_num);
  return tmp_storage_size;
}

#undef MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void GpuKernelUtilIf<T, Derived>::

KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  OF_CUDA_CHECK(
      cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, x, max_ptr, n, ctx->cuda_stream()));
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  OF_CUDA_CHECK(
      cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, x, sum_ptr, n, ctx->cuda_stream()));
}
KU_IF_METHOD CopyColsRegion(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num,
                            const T* x, const int64_t x_col_offset, const int64_t x_lda, T* y,
                            const int64_t y_col_offset, const int64_t y_lda) {
  CopyColsRegionGpu<T>
      <<<BlocksNum4ThreadsNum(row_num * col_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          row_num, col_num, x, x_col_offset, x_lda, y, y_col_offset, y_lda);
}
KU_IF_METHOD RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                    void* temp_storage, const size_t temp_storage_bytes) {
  MatrixRowReduce<T, ReduceCoreMax>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);
}
KU_IF_METHOD RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const T* x, T* y,
                    void* temp_storage, const size_t temp_storage_bytes) {
  MatrixRowReduce<T, ReduceCoreAdd>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);
}

KU_IF_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const ShapeView& x_shape,
                       const ShapeView& y_shape, const PbRf<int32_t>& permutation,
                       const int64_t elem_cnt, const T* x, T* y) {
  CHECK_LE(y_shape.elem_cnt(), GetMaxVal<int32_t>());
  CHECK_EQ(num_axis, y_shape.NumAxes());
  CHECK_EQ(num_axis, x_shape.NumAxes());
  TransposeUtil<T>::SwitchTranspose(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                    elem_cnt, x, y);
}

KU_IF_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  WithHostBlobAndStreamSynchronizeEnv(ctx, blob, [&](Blob* host_blob) {
    KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                        host_blob);
  });
}
KU_IF_METHOD Set(DeviceCtx* ctx, const T value, T* addr) {
  gpu_set<T><<<1, 1, 0, ctx->cuda_stream()>>>(value, addr);
}
KU_IF_METHOD Replicate(DeviceCtx* ctx, const int64_t n, T* y, const T* x) {
  ReplicateGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, x);
}

#define KU_FLOATING_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>::

KU_FLOATING_METHOD Dot(DeviceCtx* ctx, const int n, const T* x, const int incx, const T* y,
                       const int incy, T* result) {
  cublas_dot<T>(ctx->cublas_pmd_handle(), n, x, incx, y, incy, result);
}
KU_FLOATING_METHOD Copy(DeviceCtx* ctx, const int n, const T* x, const int incx, T* y,
                        const int incy) {
  cublas_copy<T>(ctx->cublas_pmh_handle(), n, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx,
                        T* y, const int incy) {
  cublas_axpy<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x, const int incx,
                        T* y, const int incy) {
  cublas_axpy<T>(ctx->cublas_pmd_handle(), n, alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx) {
  cublas_scal<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x, const int incx) {
  cublas_scal<T>(ctx->cublas_pmd_handle(), n, alpha, x, incx);
}
KU_FLOATING_METHOD Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m, int n,
                        const T alpha, const T* a, int lda, const T* x, const int incx,
                        const T beta, T* y, const int incy) {
  cublasOperation_t cublas_trans = CblasTrans2CublasTrans(trans);
  cublas_gemv<T>(ctx->cublas_pmh_handle(), cublas_trans, n, m, &alpha, a, lda, x, incx, &beta, y,
                 incy);
}
KU_FLOATING_METHOD Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                        const int m, const int n, const int k, const T alpha, const T* a,
                        const int lda, const T* b, const int ldb, const T beta, T* c,
                        const int ldc) {
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b, ldb,
                 a, lda, &beta, c, ldc);
}
KU_FLOATING_METHOD BatchedGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                               const enum CBLAS_TRANSPOSE trans_a,
                               const enum CBLAS_TRANSPOSE trans_b, int batch_size, int m, int n,
                               int k, const T alpha, const T* a, const T* b, const T beta, T* c,
                               T** buf) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  T** dev_a_ptrs = buf;
  T** dev_b_ptrs = buf + batch_size;
  T** dev_c_ptrs = buf + 2 * batch_size;
  AssignStridedAddr<T>(ctx, dev_a_ptrs, const_cast<T*>(a), a_stride, batch_size);
  AssignStridedAddr<T>(ctx, dev_b_ptrs, const_cast<T*>(b), b_stride, batch_size);
  AssignStridedAddr<T>(ctx, dev_c_ptrs, c, c_stride, batch_size);
#if CUDA_VERSION >= 9010
  cudaDataType_t data_type = CudaDataType<T>::value;
  cublasGemmBatchedEx(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                      reinterpret_cast<const void*>(&alpha),
                      reinterpret_cast<const void**>(const_cast<const T**>(dev_b_ptrs)), data_type,
                      ldb, reinterpret_cast<const void**>(const_cast<const T**>(dev_a_ptrs)),
                      data_type, lda, reinterpret_cast<const void*>(&beta),
                      reinterpret_cast<void**>(dev_c_ptrs), data_type, ldc, batch_size, data_type,
                      CUBLAS_GEMM_DEFAULT);
#else
  cublas_gemmBatched<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha,
                        const_cast<const T**>(dev_b_ptrs), ldb, const_cast<const T**>(dev_a_ptrs),
                        lda, &beta, dev_c_ptrs, ldc, batch_size);
#endif
}

KU_FLOATING_METHOD Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  ExpGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
  DivByConstParaPtrGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, alpha);
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T alpha) {
  DivByConstParaGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, alpha);
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  DivGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z);
}
KU_FLOATING_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  MulGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z);
}
KU_FLOATING_METHOD Reciprocal(DeviceCtx* ctx, const int n, const T* x, T* y) {
  ReciprocalGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Square(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  SquareGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Sqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  SqrtGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon) {
  RsqrtGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, epsilon);
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, T* y, const float epsilon) {
  RsqrtGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y,
                                                                                           epsilon);
}

KU_FLOATING_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  SigmoidForwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

KU_FLOATING_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                   const T* dy, T* dx) {
  SigmoidBackwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

KU_FLOATING_METHOD Relu(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  ReluForwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

KU_FLOATING_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                const T* dy, T* dx) {
  ReluBackwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0) {
  gpu_add<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_0);
}
KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1) {
  if (out == in_0) {
    gpu_assign_add<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, out, in_1);
  } else {
    gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, out, in_0, in_1);
  }
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2) {
  if (out == in_0) {
    gpu_assign_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, out, in_1, in_2);
  } else {
    gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, out, in_0, in_1, in_2);
  }
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3, in_4);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3, in_4, in_5);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6, const T* in_7) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2, const T* in_3, const T* in_4, const T* in_5,
                            const T* in_6, const T* in_7, const T* in_8) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8);
}

#define KU_INTEGRAL_METHOD \
  template<typename T>     \
  void KernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type>::

KU_INTEGRAL_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx,
                        T* y, const int incy) {
  AxpyGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, alpha, x, incx, y, incy);
}

KU_INTEGRAL_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  MulGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z);
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                                \
  template struct GpuKernelUtilIf<type_cpp, KernelUtil<DeviceType::kGPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

template<typename T, typename U>
__global__ void CastOnGpu(const T* in, U* out, int64_t elem_num) {
  CUDA_1D_KERNEL_LOOP(i, elem_num) { out[i] = static_cast<U>(in[i]); }
}

template<>
__global__ void CastOnGpu<float, half>(const float* in, half* out, int64_t elem_num) {
  const int64_t elem_num_2 = elem_num / 2;
  const auto* in_2 = reinterpret_cast<const float2*>(in);
  auto* out_2 = reinterpret_cast<half2*>(out);
  CUDA_1D_KERNEL_LOOP(i, elem_num_2) { out_2[i] = __float22half2_rn(in_2[i]); }
  if (elem_num % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[elem_num - 1] = __float2half(in[elem_num - 1]);
  }
}

template<>
__global__ void CastOnGpu<half, float>(const half* in, float* out, int64_t elem_num) {
  const int64_t elem_num_2 = elem_num / 2;
  const auto* in_2 = reinterpret_cast<const half2*>(in);
  auto* out_2 = reinterpret_cast<float2*>(out);
  CUDA_1D_KERNEL_LOOP(i, elem_num_2) { out_2[i] = __half22float2(in_2[i]); }
  if (elem_num % 2 == 1 && blockIdx.x == 0 && threadIdx.x == 0) {
    out[elem_num - 1] = __half2float(in[elem_num - 1]);
  }
}

template<typename T, typename U>
void CopyElemOnGpu(DeviceCtx* ctx, const T* in_dptr, U* out_dptr, int64_t elem_num) {
  if (std::is_same<T, U>::value) {
    Memcpy<DeviceType::kGPU>(ctx, out_dptr, in_dptr, elem_num * sizeof(T));
  } else {
    CastOnGpu<T, U>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            in_dptr, out_dptr, elem_num);
  }
}

template<>
void CopyElemOnGpu<float, float16>(DeviceCtx* ctx, const float* in_dptr, float16* out_dptr,
                                   int64_t elem_num) {
  CastOnGpu<float, half>
      <<<BlocksNum4ThreadsNum(RoundUp(elem_num, 2) / 2), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(in_dptr, reinterpret_cast<half*>(out_dptr), elem_num);
}

template<>
void CopyElemOnGpu<float16, float>(DeviceCtx* ctx, const float16* in_dptr, float* out_dptr,
                                   int64_t elem_num) {
  CastOnGpu<half, float>
      <<<BlocksNum4ThreadsNum(RoundUp(elem_num, 2) / 2), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(reinterpret_cast<const half*>(in_dptr), out_dptr, elem_num);
}

#define INSTANTIATE_COPY_ELEM_ON_GPU(T, U) \
  template void CopyElemOnGpu(DeviceCtx* ctx, const T* in_dptr, U* out_dptr, int64_t elem_num);

#define MAKE_COPY_ELEM_ON_GPU_ENTRY(TPair, UPair) \
  INSTANTIATE_COPY_ELEM_ON_GPU(OF_PP_PAIR_FIRST(TPair), OF_PP_PAIR_FIRST(UPair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_COPY_ELEM_ON_GPU_ENTRY, POD_DATA_TYPE_SEQ, POD_DATA_TYPE_SEQ)

}  // namespace oneflow
