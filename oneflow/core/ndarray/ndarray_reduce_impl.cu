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
#include "oneflow/core/kernel/util/numerics.cuh"
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/permutation_iterator.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace cub {
struct Prod {
  template<typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a * b;
  }
};
struct Any {
  template<typename T, typename U>
  __host__ __device__ __forceinline__ T operator()(const T& a, const U& b) const {
    return a || b;
  }
};
struct All {
  template<typename T, typename U>
  __host__ __device__ __forceinline__ T operator()(const T& a, const U& b) const {
    return a && b;
  }
};
struct NanSum {
  template<typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    if (oneflow::detail::numerics<T>::isnan(a))
      return oneflow::detail::numerics<T>::isnan(b) ? T{0} : b;
    return oneflow::detail::numerics<T>::isnan(b) ? a : a + b;
  }
};

}  // namespace cub

namespace oneflow {

namespace {

template<template<typename> class R, typename T, typename K, typename RetT>
__global__ void MatrixColReduceBy1ThreadPerColumn(K num_elems, K num_cols, const T* in, RetT* out) {
  CUDA_1D_KERNEL_LOOP_T(K, j, num_cols) {
    K index = j;
    T sum = in[index];
    for (index += num_cols; index < num_elems; index += num_cols) {
      sum = R<T>::Invoke(sum, in[index]);
    }
    out[j] = sum;
  }
}

template<typename T>
struct WithAlign2 {
  union {
    T value;
    int32_t padding;
  };
};

template<template<typename> class R, typename T, typename K, typename RetT>
__global__ void MatrixColReduceByWarpBlock(K num_elems, K num_cols, const T* in, RetT* out) {
  const K thread_col = threadIdx.x % kCudaWarpSize;
  const K thread_row = threadIdx.x / kCudaWarpSize;
  const K thread_dim_row = blockDim.x / kCudaWarpSize;
  const K num_valid_threads = thread_dim_row * num_cols;  // ASSERT: always <= num_elems
  const K col = blockIdx.x * kCudaWarpSize + thread_col;
  __shared__ WithAlign2<T> partial_values[kCudaWarpSize * kCudaWarpSize];
  if (col < num_cols) {
    K index = thread_row * num_cols + col;
    T val = in[index];
    for (index += num_valid_threads; index < num_elems; index += num_valid_threads) {
      val = R<T>::Invoke(val, in[index]);
    }
    partial_values[threadIdx.x].value = val;
  }
  __syncthreads();
  if (col < num_cols && thread_row == 0) {
    int index = thread_col;
    T val = partial_values[index].value;
    for (index += kCudaWarpSize; index < blockDim.x; index += kCudaWarpSize) {
      val = R<T>::Invoke(val, partial_values[index].value);
    }
    out[col] = val;
  }
}

template<template<typename> class R, typename T, typename K, typename RetT>
void MatrixColReduceBy1BlockLayer(ep::Stream* stream, K num_elems, K num_cols, const T* in,
                                  RetT* out) {
  CHECK_LE(num_cols, kCudaMaxBlocksNum * kCudaWarpSize);
  const K num_rows = num_elems / num_cols;
  CHECK_GT(num_rows, 0);
  if (num_rows < kCudaWarpSize) {
    RUN_CUDA_KERNEL((MatrixColReduceBy1ThreadPerColumn<R, T, K, RetT>), stream, num_cols, num_elems,
                    num_cols, in, out);
  } else {
    const int num_blocks = (num_cols + kCudaWarpSize - 1) / kCudaWarpSize;
    const int num_threads = kCudaWarpSize * kCudaWarpSize;
    auto Reduce = &MatrixColReduceByWarpBlock<R, T, K, RetT>;
    Reduce<<<num_blocks, num_threads, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        num_elems, num_cols, in, out);
  }
}

const static int32_t kNumRows4OneBlockLayer = kCudaWarpSize * kCudaWarpSize;
const static int32_t kNumCols4OneBlockLayer = kCudaMaxBlocksNum * kCudaWarpSize / 2;

template<template<typename> class R, typename T, typename K>
void MatrixColReduceK(ep::Stream* stream, K num_rows, K num_cols, const T* in,
                      typename BinaryFuncTrait<R, T>::return_type* out, T* tmp) {
  K num_elems = num_rows * num_cols;
  if (num_rows < kNumRows4OneBlockLayer || num_cols > kNumCols4OneBlockLayer) {
    MatrixColReduceBy1BlockLayer<R, T, K, typename BinaryFuncTrait<R, T>::return_type>(
        stream, num_elems, num_cols, in, out);
  } else {
    int scale_shift = 1;
    for (; true; ++scale_shift) {
      if ((num_rows >> scale_shift) < kNumRows4OneBlockLayer) { break; }
      if ((num_cols << scale_shift) > kNumCols4OneBlockLayer) { break; }
    }
    MatrixColReduceBy1BlockLayer<R, T, K, T>(stream, num_elems, (num_cols << scale_shift), in, tmp);
    // recursively calls MatrixColReduceK(...) log32(num_rows) times at most
    MatrixColReduceK<R, T, K>(stream, (1 << scale_shift), num_cols, tmp, out, tmp);
  }
}

template<template<typename> class R, typename T>
void MatrixColReduce(ep::Stream* stream, int64_t num_rows, int64_t num_cols, const T* in,
                     typename BinaryFuncTrait<R, T>::return_type* out, T* tmp) {
  if (IsKernelSafeInt32(num_rows * num_cols)) {
    return MatrixColReduceK<R, T, int32_t>(stream, num_rows, num_cols, in, out, tmp);
  } else {
    return MatrixColReduceK<R, T, int64_t>(stream, num_rows, num_cols, in, out, tmp);
  }
}

}  // namespace

template<typename T, template<typename> class binary_func>
struct CubFunctor4BianryFunc;

#define SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC(func_name)          \
  template<typename T>                                           \
  struct CubFunctor4BianryFunc<T, BinaryFunc##func_name> final { \
    using type = cub::func_name;                                 \
  };
OF_PP_FOR_EACH_ATOMIC(SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ(NanSum));
#undef SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC

struct RowOffsetFunctor final {
  OF_DEVICE_FUNC explicit RowOffsetFunctor(int32_t num_cols) : num_cols_(num_cols) {}
  OF_DEVICE_FUNC int32_t operator()(const int32_t& x) const { return x * num_cols_; }
  int32_t num_cols_;
};

template<typename T, template<typename> class binary_func>
struct NdarrayScalarReduce<DeviceType::kCUDA, T, binary_func> final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    return y.shape().ElemNum() == 1;
  }

  static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& y,
                     const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    size_t x_size = x.shape().ElemNum();
    size_t tmp_storage_bytes = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode = cub::DeviceReduce::Reduce(
          tmp_storage_ptr, tmp_storage_bytes, x.ptr(), y.ptr(), x_size,
          typename CubFunctor4BianryFunc<T, binary_func>::type(),
          UnitOfBinaryFunc<T, binary_func>::Val(), stream->As<ep::CudaStream>()->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayMatrixRowReduce<DeviceType::kCUDA, T, binary_func> final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 2) { return false; }
    if (y.shape().NumAxes() != 2) { return false; }
    return x.shape().At(0) == y.shape().At(0) && y.shape().At(1) == 1;
  }

  static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& y,
                     const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int32_t num_rows = y.shape().ElemNum();
    int32_t num_cols = x.shape().ElemNum() / y.shape().ElemNum();
    RowOffsetFunctor get_row_offset(num_cols);
    cub::CountingInputIterator<int32_t> counting_intput_it(0);
    cub::TransformInputIterator<int32_t, RowOffsetFunctor, cub::CountingInputIterator<int32_t>>
        transform_input_iter(counting_intput_it, get_row_offset);
    size_t tmp_storage_bytes = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode = cub::DeviceSegmentedReduce::Reduce(
          tmp_storage_ptr, tmp_storage_bytes, x.ptr(), y.ptr(), num_rows, transform_input_iter,
          transform_input_iter + 1, typename CubFunctor4BianryFunc<T, binary_func>::type(),
          UnitOfBinaryFunc<T, binary_func>::Val(), stream->As<ep::CudaStream>()->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayMatrixColReduce<DeviceType::kCUDA, T, binary_func> final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 2) { return false; }
    if (y.shape().NumAxes() != 2) { return false; }
    return y.shape().At(0) == 1 && x.shape().At(1) == y.shape().At(1);
  }

  struct XY2YXFunctor final {
    __host__ __device__ XY2YXFunctor(int32_t dim_x, int32_t dim_y) : dim_x_(dim_x), dim_y_(dim_y) {}

    __host__ __device__ int32_t operator()(const int32_t& idx) const {
      const int32_t y = idx / dim_x_;
      const int32_t x = idx % dim_x_;
      return x * dim_y_ + y;
    }

    int32_t dim_x_;
    int32_t dim_y_;
  };

  static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& y,
                     const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int64_t num_rows = x.shape().At(0);
    int64_t num_cols = x.shape().At(1);
    if (num_cols < kNumCols4OneBlockLayer) {
      return MatrixColReduce<binary_func, T>(stream, num_rows, num_cols, x.host_ptr(), y.host_ptr(),
                                             tmp_storage.host_ptr());
    }
    RowOffsetFunctor get_row_offset(num_rows);
    cub::CountingInputIterator<int32_t> counting_intput_it(0);
    cub::TransformInputIterator<int32_t, RowOffsetFunctor, cub::CountingInputIterator<int32_t>>
        transform_input_iter(counting_intput_it, get_row_offset);

    XY2YXFunctor xy2yx(x.shape().At(0), x.shape().At(1));
    using XY2YxIndexIter =
        cub::TransformInputIterator<int32_t, XY2YXFunctor, cub::CountingInputIterator<int32_t>>;
    XY2YxIndexIter xy2yx_iter(counting_intput_it, xy2yx);
    PermutationIterator<const T, const T*, XY2YxIndexIter> x_iter(x.ptr(), xy2yx_iter);
    size_t tmp_storage_bytes = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode = cub::DeviceSegmentedReduce::Reduce(
          tmp_storage_ptr, tmp_storage_bytes, x_iter, y.ptr(), num_cols, transform_input_iter,
          transform_input_iter + 1, typename CubFunctor4BianryFunc<T, binary_func>::type(),
          UnitOfBinaryFunc<T, binary_func>::Val(), stream->As<ep::CudaStream>()->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayXYZCubeXZReduce<DeviceType::kCUDA, T, binary_func> final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static bool Matched(const XpuVarNdarray<RetT>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 3) { return false; }
    if (y.shape().NumAxes() != 3) { return false; }
    return y.shape().At(0) == 1 && x.shape().At(1) == y.shape().At(1) && y.shape().At(2) == 1;
  }

  struct XYZ2YxzFunctor final {
    __host__ __device__ XYZ2YxzFunctor(int32_t dim_x, int32_t dim_y, int32_t dim_z)
        : dim_z_(dim_z), dim_xz_(dim_x * dim_z), dim_yz_(dim_y * dim_z) {}

    __host__ __device__ int32_t operator()(const int32_t& idx) const {
      const int32_t y = idx / dim_xz_;
      const int32_t xz_idx = idx % dim_xz_;
      const int32_t x = xz_idx / dim_z_;
      const int32_t z = xz_idx % dim_z_;
      return x * dim_yz_ + y * dim_z_ + z;
    }

    int32_t dim_z_;
    int32_t dim_xz_;
    int32_t dim_yz_;
  };

  static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& y,
                     const XpuVarNdarray<const T>& x, const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int32_t num_rows = y.shape().ElemNum();
    int32_t num_cols = x.shape().ElemNum() / y.shape().ElemNum();

    RowOffsetFunctor get_row_offset(num_cols);
    cub::CountingInputIterator<int32_t> counting_intput_it(0);
    cub::TransformInputIterator<int32_t, RowOffsetFunctor, cub::CountingInputIterator<int32_t>>
        transform_input_iter(counting_intput_it, get_row_offset);

    XYZ2YxzFunctor xyz2yxz(x.shape().At(0), x.shape().At(1), x.shape().At(2));
    using XYZ2YxzIndexIter =
        cub::TransformInputIterator<int32_t, XYZ2YxzFunctor, cub::CountingInputIterator<int32_t>>;
    XYZ2YxzIndexIter xyz2yxz_iter(counting_intput_it, xyz2yxz);
    PermutationIterator<const T, const T*, XYZ2YxzIndexIter> x_iter(x.ptr(), xyz2yxz_iter);
    size_t tmp_storage_bytes = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode = cub::DeviceSegmentedReduce::Reduce(
          tmp_storage_ptr, tmp_storage_bytes, x_iter, y.ptr(), num_rows, transform_input_iter,
          transform_input_iter + 1, typename CubFunctor4BianryFunc<T, binary_func>::type(),
          UnitOfBinaryFunc<T, binary_func>::Val(), stream->As<ep::CudaStream>()->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

namespace {

template<typename T, int NDIMS, template<typename> class binary_func>
__global__ void NdarrayReduceGpuInplaceReduceAxis(const XpuReducedNdarray<T, NDIMS> dst_reduced,
                                                  const XpuReducedNdarray<T, NDIMS> x, int axis) {
  NdarrayReduceCore<T, NDIMS, binary_func>::ReduceAxis(dst_reduced, x, axis);
}

}  // namespace

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayReduceCoreWrapper<DeviceType::kCUDA, T, NDIMS, binary_func> final {
  static void ReduceAxis(ep::Stream* stream, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis) {
    size_t n = x.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayReduceGpuInplaceReduceAxis<T, NDIMS, binary_func>), stream, n,
                    dst_reduced, x, axis);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE_IMPL(dtype, binary_func)                                        \
  template struct NdarrayScalarReduce<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype), binary_func>;    \
  template struct NdarrayMatrixRowReduce<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayMatrixColReduce<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayXYZCubeXZReduce<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype), binary_func>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 ARITHMETIC_REDUCE_BINARY_FUNC_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL, FLOATING_DATA_TYPE_SEQ,
                                 NANSUM_REDUCE_BINARY_FUNC_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL,
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ,
                                 LOGICAL_REDUCE_BINARY_FUNC_SEQ);

#define INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER(dtype_pair, NDIMS, binary_func)                    \
  template struct NdarrayReduceCoreWrapper<DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, \
                                           binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 DIM_SEQ, ARITHMETIC_REDUCE_BINARY_FUNC_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER, FLOATING_DATA_TYPE_SEQ,
                                 DIM_SEQ, NANSUM_REDUCE_BINARY_FUNC_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER,
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ,
                                 DIM_SEQ, LOGICAL_REDUCE_BINARY_FUNC_SEQ);

}  // namespace oneflow
