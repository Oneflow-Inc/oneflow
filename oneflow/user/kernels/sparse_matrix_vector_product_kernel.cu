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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

#define BLOCK_SIZE 128u
#define FULL_WARP_MASK 0xffffffff

template<typename T, typename IndexType>
__device__ T warpReduce(T reduction_result) {
  for (IndexType i = 32 / 2; i > 0; i++) {
    reduction_result += __shfl_down_sync(FULL_WARP_MASK, reduction_result, i);
  }
  return reduction_result;
}

template<typename IndexType>
__device__ IndexType PrevPowerOf2(IndexType n) {
  while (n & n - 1) { n = n & n - 1; }
  return n;
}

template<typename T, typename IndexType>
__device__ void CsrStream(const IndexType row_ptrs_begin_idx, const IndexType row_ptrs_end_idx,
                          const IndexType col_ids_begin_idx, const IndexType col_ids_end_idx,
                          const IndexType block_nnz, const IndexType* row_ptrs,
                          const IndexType* col_ids, const T* mat_values, const T* in_vec,
                          T* out_vec, T* cache) {
  /* ========== step 1: product  ========== */
  // obtain index in col_ids of current thread
  const IndexType thread_col_ids_idx = col_ids_begin_idx + threadIdx.x;

  // finish the product workload of current thread, and store the result in shared memory
  if (threadIdx.x < block_nnz) {
    cache[threadIdx.x] = mat_values[thread_col_ids_idx] * in_vec[col_ids[thread_col_ids_idx]];
  }
  __syncthreads();

  /* ========== step 2: reduction ========== */
  // calculate the number of threads for reduction per row (size of a reduction group)
  const IndexType block_num_rows = row_ptrs_end_idx - row_ptrs_begin_idx;
  const IndexType reduction_group_size = PrevPowerOf2<IndexType>(BLOCK_SIZE / block_num_rows);

  if (reduction_group_size > 1) {
    // case: reduce all non zeroes of the row by multiple threads,
    //       imply that the number of processing rows is lees than BLOCK_SIZE

    // obtain the thread index in the located reduction group
    const IndexType thread_in_reduction_group = threadIdx.x % reduction_group_size;

    // obtain the index of target row of current thread
    const IndexType target_row_idx = row_ptrs_begin_idx + threadIdx.x / reduction_group_size;

    // initialize reduction result of current thread
    T reduction_result = 0.0;

    // construct equal-sized reduction group in the shared memory cache
    if (target_row_idx < row_ptrs_end_idx) {
      // obtain the indices inside shared memory cache for reduction of current thread
      const IndexType cache_begin_idx = row_ptrs[target_row_idx] - row_ptrs[row_ptrs_begin_idx];
      const IndexType cache_end_idx = row_ptrs[target_row_idx + 1] - row_ptrs[row_ptrs_begin_idx];

      // merge
      for (IndexType j = cache_begin_idx + thread_in_reduction_group; j < cache_end_idx;
           j += reduction_group_size) {
        reduction_result += cache[j];
      }
    }
    __syncthreads();

    // write back the reduction group result to shared memory cache
    cache[threadIdx.x] = reduction_result;

    // final reduction loop
    for (IndexType j = reduction_group_size / 2; j > 0; j /= 2) {
      __syncthreads();
      const bool use_result = (thread_in_reduction_group < j) && (threadIdx.x + j < BLOCK_SIZE);
      if (use_result) { reduction_result += cache[threadIdx.x + j]; }
      __syncthreads();
      if (use_result) { cache[threadIdx.x] = reduction_result; }
    }

    // write back result
    if (thread_in_reduction_group == 0 && target_row_idx < row_ptrs_end_idx) {
      out_vec[target_row_idx] = reduction_result;
    }
  } else {
    // case: reduce all non zeroes of row/col by a single thread,
    //       imply that the number of processing rows might be larger than BLOCK_SIZE

    // obtain the targeted row/col of current thread
    IndexType target_row_idx = row_ptrs_begin_idx + threadIdx.x;

    while (target_row_idx < row_ptrs_end_idx) {
      // obtain the indices inside shared memory cache for reduction of current thread
      const IndexType cache_begin_idx = row_ptrs[target_row_idx] - row_ptrs[row_ptrs_begin_idx];
      const IndexType cache_end_idx = row_ptrs[target_row_idx + 1] - row_ptrs[row_ptrs_begin_idx];

      // initialize reduction result of current thread
      T reduction_result = 0.0;

      // reduction loop
      for (IndexType j = cache_begin_idx; j < cache_end_idx; j++) { reduction_result += cache[j]; }

      // write back result
      out_vec[target_row_idx] = reduction_result;

      // jump across BLOCK_SIZE
      // prevent the case that the number of merged rows is larger than BLOCK_SIZE
      target_row_idx += BLOCK_SIZE;
    }
  }
}

template<typename T, typename IndexType>
__device__ void CsrVector(const IndexType num_rows, const IndexType target_row_idx,
                          const IndexType col_ids_begin_idx, const IndexType col_ids_end_idx,
                          const IndexType block_nnz, const IndexType* row_ptrs,
                          const IndexType* col_ids, const T* mat_values, const T* in_vec,
                          T* out_vec) {
  // obtain metadata of current warp
  const IndexType warp_id = threadIdx.x / 32;
  const IndexType lane = threadIdx.x % 32;

  // initailize reduction result
  T reduction_result = 0.0;

  if (target_row_idx < num_rows) {  // TODO: might no need to judge here
    // reduction loop
    for (IndexType j = col_ids_begin_idx + lane; j < col_ids_end_idx; j += 32) {
      reduction_result += mat_values[j] * in_vec[col_ids[j]];
    }

    // use warp primitive for reduction within the warp
    reduction_result = warpReduce<T, IndexType>(reduction_result);
  }

  if (lane == 0 && warp_id == 0
      && target_row_idx < num_rows) {  // TODO: might no need the last judgement here
    out_vec[target_row_idx] = reduction_result;
  }
}

template<typename T, typename IndexType>
__device__ void CsrVectorL(const IndexType num_rows, const IndexType target_row_idx,
                           const IndexType col_ids_begin_idx, const IndexType col_ids_end_idx,
                           const IndexType block_nnz, const IndexType* row_ptrs,
                           const IndexType* col_ids, const T* mat_values, const T* in_vec,
                           T* out_vec, T* cache) {
  // obtain metadata of current warp
  const IndexType warp_id = threadIdx.x / 32;
  const IndexType lane = threadIdx.x % 32;

  // initailize reduction result
  T reduction_result = 0.0;

  if (target_row_idx < num_rows) {  // TODO: might no need to judge here
    // reduction loop
    for (IndexType j = col_ids_begin_idx + threadIdx.x; j < col_ids_end_idx; j += BLOCK_SIZE) {
      reduction_result += mat_values[j] * in_vec[col_ids[j]];
    }
  }

  // use warp primitive for reduction within the warp
  reduction_result = warpReduce<T, IndexType>(reduction_result);

  // store the reduction result of current warp to shared memory cache
  if (lane == 0) { cache[warp_id] = reduction_result; }
  __syncthreads();

  // use the first warp to reduce final result
  if (warp_id == 0) {
    reduction_result = 0.0;
    for (IndexType j = lane; j < BLOCK_SIZE / 32; j += 32) { reduction_result += cache[j]; }
    reduction_result = warpReduce<T, IndexType>(reduction_result);

    // use the first lane to write back result
    if (lane == 0 && target_row_idx < num_rows) {  // TODO: might no need the last judgement here
      out_vec[target_row_idx] = reduction_result;
    }
  }
}

// ref:
// https://github.com/senior-zero/matrix_format_performance/blob/master/gpu/csr_adaptive_spmv.cu
template<typename T, typename IndexType>
__global__ void CsrSpMVGpu(const IndexType* row_assignment, const IndexType num_rows,
                           const IndexType* row_ptrs, const IndexType* col_ids, const T* mat_values,
                           const T* in_vec, T* out_vec) {
  // obtain range of row indices for current block
  const IndexType row_ptrs_begin_idx = row_assignment[blockIdx.x];  // start index of row_ptrs
  const IndexType row_ptrs_end_idx =
      row_assignment[blockIdx.x + 1];  // end index of row_ptrs (not included)

  // obtain range of element indices for current block
  const IndexType col_ids_begin_idx = row_ptrs[row_ptrs_begin_idx];  // start index of col_ids
  const IndexType col_ids_end_idx =
      row_ptrs[row_ptrs_end_idx];  // end index of col_ids (not included)

  // obtain number of nnz to be processed for current block
  const IndexType block_nnz = col_ids_end_idx - col_ids_begin_idx;

  // declare shared memory for synchronization among threads in block
  __shared__ T cache[BLOCK_SIZE];

  // invoke different rotinue according to nnz distribution
  if (row_ptrs_end_idx - row_ptrs_begin_idx > 1) {
    // case: more than one row be assigned to current block
    CsrStream<T, IndexType>(row_ptrs_begin_idx, row_ptrs_end_idx, col_ids_begin_idx,
                            col_ids_end_idx, block_nnz, row_ptrs, col_ids, mat_values, in_vec,
                            out_vec, cache);
  } else {
    // case: only one row be assigned to current block
    if (block_nnz <= 64 || BLOCK_SIZE <= 32) {
      // all warps within CsrVector are doing the exact same thing,
      // so we don't want too many threads in it
      CsrVector<T, IndexType>(num_rows, row_ptrs_begin_idx, col_ids_begin_idx, col_ids_end_idx,
                              block_nnz, row_ptrs, col_ids, mat_values, in_vec, out_vec);
    } else {
      CsrVectorL<T, IndexType>(num_rows, row_ptrs_begin_idx, col_ids_begin_idx, col_ids_end_idx,
                               block_nnz, row_ptrs, col_ids, mat_values, in_vec, out_vec, cache);
    }
  }
}

template<typename IndexType>
std::vector<IndexType>* InferRowAssignment(const IndexType num_ptrs, const IndexType* ptrs,
                                           IndexType* num_block) {
  IndexType last_row_id = 0;
  IndexType nnz_sum = 0;
  std::vector<IndexType>* row_assignment = new std::vector<IndexType>;

#define UPDATE_ASSIGNMENT_META(row_id) \
  row_assignment->push_back(row_id);   \
  last_row_id = row_id;                \
  nnz_sum = 0;
  for (IndexType i = 1; i < num_ptrs; i++) {
    nnz_sum += ptrs[i] - ptrs[i - 1];
    if (nnz_sum == BLOCK_SIZE) {
      // case: number of scanned nnz equals to BLOCK_SIZE
      UPDATE_ASSIGNMENT_META(i);
    } else if (nnz_sum > BLOCK_SIZE) {
      // case: number of scanned nnz exceeds BLOCK_SIZE
      if (i - last_row_id > 1) { i--; }
      UPDATE_ASSIGNMENT_META(i);
    } else if (i - last_row_id > BLOCK_SIZE) {
      // case: number of merged rows exceed BLOCK_SIZE
      UPDATE_ASSIGNMENT_META(i);
    }
  }
#undef UPDATE_ASSIGNMENT_META

  *num_block = row_assignment->size();
  row_assignment->push_back(num_ptrs);
  return row_assignment;
}

template<typename T, typename IndexType>
void LaunchCsrSpMVGpuKernel(ep::Stream* stream, const IndexType num_rows, const IndexType nnz,
                            const IndexType* mat_row, const IndexType* mat_col, const T* mat_values,
                            const T* in_vec, T* out_vec) {
  // conduct row assignment and infer kernel shape
  IndexType num_block = 0;
  std::vector<IndexType>* row_assignment =
      InferRowAssignment<IndexType>(num_rows, mat_row, &num_block);

  // copy row_assignment to device memory
  IndexType* d_row_assignment;
  OF_CUDA_CHECK(cudaMalloc(&d_row_assignment, (num_block + 1) * sizeof(IndexType)));
  OF_CUDA_CHECK(cudaMemcpyAsync(d_row_assignment, row_assignment->data(),
                                (num_block + 1) * sizeof(IndexType), cudaMemcpyDefault,
                                stream->As<ep::CudaStream>()->cuda_stream()));

  // launch kernel
  CsrSpMVGpu<T, IndexType>
      <<<num_block, BLOCK_SIZE, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          d_row_assignment, num_rows, mat_row, mat_col, mat_values, in_vec, out_vec);

  // clear memory space for row assignment
  OF_CUDA_CHECK(cudaDeviceSynchronize());
  OF_CUDA_CHECK(cudaFree(d_row_assignment));
  delete row_assignment;
}

template<typename T, typename IndexType>
void DispatchFormat(ep::Stream* stream, const IndexType num_ptrs, const IndexType nnz,
                    const IndexType* mat_row, const IndexType* mat_col, const T* mat_value,
                    const T* vec, T* out, const std::string& format) {
  if (format == "csr") {
    LaunchCsrSpMVGpuKernel<T, IndexType>(stream, num_ptrs, nnz, mat_row, mat_col, mat_value, vec,
                                         out);
  } else if (format == "csc") {
    UNIMPLEMENTED();
  } else if (format == "coo") {
    UNIMPLEMENTED();
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void DispatchIndexType(ep::Stream* stream, const int64_t num_ptrs, const int64_t nnz,
                       const user_op::Tensor* mat_row_tensor, const user_op::Tensor* mat_col_tensor,
                       const user_op::Tensor* mat_value_tensor,
                       const user_op::Tensor* in_vec_tensor, user_op::Tensor* out_vec_tensor,
                       const std::string& format) {
  if (num_ptrs < (1 << 30) && nnz < (1 << 30)) {
    DispatchFormat<T, int32_t>(stream, static_cast<int32_t>(num_ptrs), static_cast<int32_t>(nnz),
                               mat_row_tensor->dptr<int32_t>(), mat_col_tensor->dptr<int32_t>(),
                               mat_value_tensor->dptr<T>(), in_vec_tensor->dptr<T>(),
                               out_vec_tensor->mut_dptr<T>(), format);
  } else {
    DispatchFormat<T, int64_t>(stream, static_cast<int64_t>(num_ptrs), static_cast<int64_t>(nnz),
                               mat_row_tensor->dptr<int64_t>(), mat_col_tensor->dptr<int64_t>(),
                               mat_value_tensor->dptr<T>(), in_vec_tensor->dptr<T>(),
                               out_vec_tensor->mut_dptr<T>(), format);
  }
}

template<typename T>
class GpuSparseMatrixVectorProductKernel final : public user_op::OpKernel {
 public:
  GpuSparseMatrixVectorProductKernel() = default;
  ~GpuSparseMatrixVectorProductKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain tensors from context
    const user_op::Tensor* in_mat_rows = ctx->Tensor4ArgNameAndIndex("mat_rows", 0);
    const user_op::Tensor* in_mat_cols = ctx->Tensor4ArgNameAndIndex("mat_cols", 0);
    const user_op::Tensor* in_mat_values = ctx->Tensor4ArgNameAndIndex("mat_values", 0);
    const user_op::Tensor* in_vec = ctx->Tensor4ArgNameAndIndex("in_vec", 0);
    const std::string format = ctx->Attr<std::string>("format");
    const int64_t num_rows = ctx->Attr<int64_t>("num_rows");
    user_op::Tensor* out_vec = ctx->Tensor4ArgNameAndIndex("out_vec", 0);

    // check attributes
    CHECK(format == "csr" || format == "csc" || format == "coo")
        << "unknown data format " << format;
    CHECK_GT(num_rows, 0) << "invalid number of rows attribute" << num_rows;

    // obtain tensor shapes and number of axes
    const ShapeView& mat_rows_shape = in_mat_rows->shape_view();
    const ShapeView& mat_cols_shape = in_mat_cols->shape_view();
    const ShapeView& mat_values_shape = in_mat_values->shape_view();
    const ShapeView& in_vec_shape = in_vec->shape_view();

    // validate number of axes
    CHECK_EQ(mat_rows_shape.NumAxes(), 1)
        << "number of axes of \'mat_rows\' should be 1, yet get " << mat_rows_shape.NumAxes();
    CHECK_EQ(mat_cols_shape.NumAxes(), 1)
        << "number of axes of \'mat_cols\' should be 1, yet get " << mat_cols_shape.NumAxes();
    CHECK_EQ(mat_values_shape.NumAxes(), 1)
        << "number of axes of \'mat_values\' should be 1, yet get " << mat_values_shape.NumAxes();
    CHECK_EQ(in_vec_shape.NumAxes(), 1)
        << "number of axes of \'in_vec\' should be 1, yet get " << in_vec_shape.NumAxes();

    // check input shape
    const int64_t num_mat_rows = mat_rows_shape.At(0) - 1;
    const int64_t num_mat_cols = mat_cols_shape.At(0);
    const int64_t num_mat_values = mat_values_shape.At(0);
    if (format == "csr") {
      CHECK_EQ(num_mat_cols, num_mat_values)
          << "under CSR format, "
          << "the number of elements in \'mat_cols\'(" << num_mat_cols
          << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
      CHECK_EQ(num_mat_rows, num_rows)
          << "under CSR format, "
          << "the number of elements in \'mat_rows\'(" << num_mat_cols
          << ") should be equal to the given attribute \'num_rows\'(" << num_rows << ")";
    } else if (format == "csc") {
      CHECK_EQ(num_mat_rows, num_mat_values)
          << "under CSC format, "
          << "the number of elements in \'mat_rows\'(" << num_mat_rows
          << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
    } else if (format == "coo") {
      CHECK_EQ(num_mat_rows, num_mat_cols)
          << "under COO format, "
          << "the number of elements in \'mat_rows\'(" << num_mat_rows
          << ") should be equal to the one of \'mat_cols\'(" << num_mat_cols << ")";
      CHECK_EQ(num_mat_rows, num_mat_values)
          << "under COO format, "
          << "the number of elements in \'mat_rows\'(" << num_mat_rows
          << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
    }

    // check whether both mat_cols and mat_rows are index data type
    const DataType mat_rows_dtype = in_mat_rows->data_type();
    const DataType mat_cols_dtype = in_mat_cols->data_type();
    CHECK(IsIndexDataType(mat_rows_dtype))
        << "The dtype of mat_rows must be integer, but found " << DataType_Name(mat_rows_dtype);
    CHECK(IsIndexDataType(mat_cols_dtype))
        << "The dtype of mat_cols must be integer, but found " << DataType_Name(mat_cols_dtype);
    CHECK_EQ(mat_rows_dtype, mat_cols_dtype)
        << "The dtype of mat_rows (" << DataType_Name(mat_rows_dtype) << ")"
        << "is not consistent with"
        << "the dtype of mat_cols (" << DataType_Name(mat_cols_dtype) << ")";

    // check data type of the value of both sparse matrix and vector
    DataType mat_values_dtype = in_mat_values->data_type();
    DataType in_vec_dtype = in_vec->data_type();
    CHECK_EQ(mat_values_dtype, in_vec_dtype)
        << "data type of \'mat_values\' is not consitant with \'in_vec\'";

    // start dispatch process
    DispatchIndexType<T>(ctx->stream(),
                         /*num_ptrs*/ format == "csr" ? num_mat_rows : num_mat_cols,
                         /*nnz*/ format == "csr" ? num_mat_cols : num_mat_rows,
                         /*mat_row_tensor*/ in_mat_rows,
                         /*mat_col_tensor*/ in_mat_cols,
                         /*mat_value_tensor*/ in_mat_values,
                         /*in_vec_tensor*/ in_vec,
                         /*out_vec_tensor*/ out_vec,
                         /*format*/ format);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#undef BLOCK_SIZE

}  // namespace

#define REGISTER_GPU_SPARSE_MATRIX_VECTOR_PRODUCT_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("sparse_matrix_vector_product")                 \
      .SetCreateFn<GpuSparseMatrixVectorProductKernel<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out_vec", 0) == GetDataType<dtype>::value));

REGISTER_GPU_SPARSE_MATRIX_VECTOR_PRODUCT_KERNEL(double)
REGISTER_GPU_SPARSE_MATRIX_VECTOR_PRODUCT_KERNEL(float)
REGISTER_GPU_SPARSE_MATRIX_VECTOR_PRODUCT_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_GPU_SPARSE_MATRIX_VECTOR_PRODUCT_KERNEL(nv_bfloat16)
#endif

}  // namespace oneflow