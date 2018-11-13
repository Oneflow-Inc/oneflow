#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void RsqrtGpu(const int64_t n, T* x, const float epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}

template<typename T>
__global__ void ExpGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::exp(x[i]); }
}

template<typename T>
__global__ void DivGpu(const int64_t n, T* x, const T* alpha_ptr) {
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
__global__ void MulByScalarGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[0]; }
}

template<typename T>
__global__ void ReciprocalGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = static_cast<T>(1.0) / x[i]; }
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
__global__ void TanHForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::tanh(x[i]); }
}

template<typename T>
__global__ void TanHBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * (1.0 - y[i] * y[i]); }
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

const int32_t kMaxDim = OF_PP_SEQ_SIZE(DIM_SEQ);

struct Int32Array {
  int32_t val[kMaxDim];
};

struct Int64Array {
  int64_t val[kMaxDim];
};

__device__ void ComputeOffset(const int32_t num_axis, const int64_t* x_dims,
                              const int32_t* permutation, int64_t* x_strides) {
  int64_t buff[kMaxDim];
  int64_t cur_stride = 1;
  for (int32_t i = num_axis - 1; i >= 0; --i) {
    buff[i] = cur_stride;
#if __CUDA_ARCH__ >= 350
    cur_stride *= __ldg(x_dims + i);
#else
    cur_stride *= x_dims[i];
#endif
  }
  for (int32_t i = 0; i < num_axis; ++i) {
#if __CUDA_ARCH__ >= 350
    x_strides[i] = buff[__ldg(permutation + i)];
#else
    x_strides[i] = buff[permutation[i]];
#endif
  }
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

__device__ int64_t GetXIndex(const int32_t num_axis, const int64_t* y_shape,
                             const int64_t* x_strides, int64_t y_idx) {
  int64_t x_idx = 0;
  for (int32_t i = num_axis - 1; i >= 0 && y_idx > 0; --i) {
    x_idx += (y_idx % y_shape[i]) * x_strides[i];
    y_idx /= y_shape[i];
  }
  return x_idx;
}

template<typename T>
__global__ void TransposeGpu(const int32_t num_axis, const Int64Array x_shape,
                             const Int64Array y_shape, const Int32Array permutation,
                             const int64_t elem_cnt, const T* x, T* y) {
  __shared__ int64_t x_strides[kMaxDim];
  __shared__ int64_t x_dims_shared[kMaxDim];
  __shared__ int64_t y_dims_shared[kMaxDim];
  __shared__ int32_t perm_shared[kMaxDim];
  const int32_t tid = threadIdx.x;
  if (tid < num_axis) {
    x_dims_shared[tid] = x_shape.val[tid];
    y_dims_shared[tid] = y_shape.val[tid];
    perm_shared[tid] = permutation.val[tid];
  }
  __syncthreads();
  if (tid == 0) { ComputeOffset(num_axis, x_dims_shared, perm_shared, x_strides); }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(y_idx, elem_cnt) {
    const int64_t x_idx = GetXIndex(num_axis, y_dims_shared, x_strides, y_idx);
#if __CUDA_ARCH__ >= 350
    y[y_idx] = __ldg(x + x_idx);
#else
    y[y_idx] = x[x_idx];
#endif
  }
}

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

}  // namespace

template<>
void Memcpy<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const void* src, size_t sz,
                              cudaMemcpyKind kind) {
  CudaCheck(cudaMemcpyAsync(dst, src, sz, kind, ctx->cuda_stream()));
}

template<>
void Memset<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const char value, size_t sz) {
  CudaCheck(cudaMemsetAsync(dst, value, sz, ctx->cuda_stream()));
}

#define MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY(func_name, T) cub::DeviceReduce::func_name<T*, T*>
DEFINE_STATIC_SWITCH_FUNC(cudaError_t, Sum, MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num) {
  size_t tmp_storage_size;
  SwitchSum(SwitchCase(data_type), nullptr, tmp_storage_size, nullptr, nullptr, sum_elem_num);
  return tmp_storage_size;
}

#undef MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY

// create temporary host blob store initializer result
#define BEFORE_CPU_INITIALIZE()                                     \
  RtBlobDesc blob_desc(blob->blob_desc().blob_desc_proto());        \
  char* host_raw_dptr = nullptr;                                    \
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->TotalByteSize())); \
  std::unique_ptr<Blob> host_blob;                                  \
  host_blob.reset(new Blob(nullptr, &blob_desc, host_raw_dptr));

// asynchronous copy to device
#define AFTER_CPU_INITIALIZE()                                                          \
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob->dptr(),                    \
                           blob->ByteSizeOfDataContentField(), cudaMemcpyHostToDevice); \
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));                                 \
  CudaCheck(cudaFreeHost(host_raw_dptr));

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void GpuKernelUtilIf<T, Derived>::

KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  CudaCheck(
      cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, x, max_ptr, n, ctx->cuda_stream()));
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr, T* temp_storage,
                 size_t temp_storage_bytes) {
  CudaCheck(
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
KU_IF_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                       const Shape& y_shape, const PbRf<int32_t>& permutation,
                       const int64_t elem_cnt, const T* x, T* y) {
  CHECK_LE(num_axis, kMaxDim);
  Int64Array x_shape_struct;
  Int64Array y_shape_struct;
  Int32Array perm_struct;
  FOR_RANGE(int32_t, i, 0, num_axis) {
    x_shape_struct.val[i] = x_shape.At(i);
    y_shape_struct.val[i] = y_shape.At(i);
    perm_struct.val[i] = permutation[i];
  }
  TransposeGpu<T>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          num_axis, x_shape_struct, y_shape_struct, perm_struct, elem_cnt, x, y);
}

KU_IF_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                      host_blob.get());
  AFTER_CPU_INITIALIZE();
}
KU_IF_METHOD InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob, const std::string& data_format) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                      host_blob.get(), data_format);
  AFTER_CPU_INITIALIZE();
}
KU_IF_METHOD InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                               const std::string& model_dir, Blob* blob,
                               const std::string& bn_in_op, int32_t dim_num,
                               int64_t num_in_each_dim) {
  BEFORE_CPU_INITIALIZE();
  KernelUtil<DeviceType::kCPU, T>::InitializeWithDir(
      ctx, part_id, part_num, model_dir, host_blob.get(), bn_in_op, dim_num, num_in_each_dim);
  AFTER_CPU_INITIALIZE();
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

KU_FLOATING_METHOD Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  ExpGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
  DivGpu<T>
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
KU_FLOATING_METHOD MulByScalar(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, T* z) {
  MulByScalarGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z);
}
KU_FLOATING_METHOD Reciprocal(DeviceCtx* ctx, const int n, const T* x, T* y) {
  ReciprocalGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, T* x, const float epsilon) {
  RsqrtGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, epsilon);
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

KU_FLOATING_METHOD TanH(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  TanHForwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

KU_FLOATING_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                                const T* dy, T* dx) {
  TanHBackwardGpu<T>
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
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1);
}

KU_FLOATING_METHOD Addition(DeviceCtx* ctx, const int64_t n, T* out, const T* in_0, const T* in_1,
                            const T* in_2) {
  gpu_add<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, out, in_0, in_1, in_2);
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

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                                \
  template struct GpuKernelUtilIf<type_cpp, KernelUtil<DeviceType::kGPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

template<>
__device__ float gpu_atomic_add(float* address, const float val) {
  return atomicAdd(address, val);
}

template<>
__device__ double gpu_atomic_add(double* address, const double val) {
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template<>
__device__ float gpu_atomic_max(float* address, const float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

template<>
__device__ double gpu_atomic_max(double* address, const double val) {
  unsigned long long int* address_as_i = (unsigned long long int*)address;
  unsigned long long int old = *address_as_i;
  unsigned long long int assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __double_as_longlong(fmaxf(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

}  // namespace oneflow
