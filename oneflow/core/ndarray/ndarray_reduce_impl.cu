#include <cub/cub.cuh>
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/permutation_iterator.h"

namespace oneflow {

template<typename T, template<typename> class binary_func>
struct CubFunctor4BianryFunc;

#define SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC(func_name)          \
  template<typename T>                                           \
  struct CubFunctor4BianryFunc<T, BinaryFunc##func_name> final { \
    using type = cub::func_name;                                 \
  };
OF_PP_FOR_EACH_ATOMIC(SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC, REDUCE_BINARY_FUNC_NAME_SEQ);
#undef SPECIALIZE_CUB_FUNCTOR_4_BINARY_FUNC

namespace {

template<typename T, template<typename> class binary_func>
void __global__ NdarrayMatrixColReduceNaiveCudaKernel(T* y_ptr, const T* x_ptr, int32_t num_rows,
                                                      int32_t num_cols) {
  CUDA_1D_KERNEL_LOOP(j, num_cols) {
    T reduced = x_ptr[j];
    FOR_RANGE(int32_t, i, 1, num_rows) {
      reduced = binary_func<T>::Invoke(reduced, x_ptr[i * num_cols + j]);
    }
    y_ptr[j] = reduced;
  }
}

}  // namespace

struct RowOffsetFunctor final {
  OF_DEVICE_FUNC explicit RowOffsetFunctor(int32_t num_cols) : num_cols_(num_cols) {}
  OF_DEVICE_FUNC int32_t operator()(const int32_t& x) const { return x * num_cols_; }
  int32_t num_cols_;
};

template<typename T, template<typename> class binary_func>
struct NdarrayScalarReduce<DeviceType::kGPU, T, binary_func> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return y.shape().ElemNum() == 1;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    size_t x_size = x.shape().ElemNum();
    size_t tmp_storage_bytes = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode =
          cub::DeviceReduce::Reduce(tmp_storage_ptr, tmp_storage_bytes, x.ptr(), y.ptr(), x_size,
                                    typename CubFunctor4BianryFunc<T, binary_func>::type(),
                                    UnitOfBinaryFunc<T, binary_func>::Val(), ctx->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayMatrixRowReduce<DeviceType::kGPU, T, binary_func> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 2) { return false; }
    if (y.shape().NumAxes() != 2) { return false; }
    return x.shape().At(0) == y.shape().At(0) && y.shape().At(1) == 1;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
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
          UnitOfBinaryFunc<T, binary_func>::Val(), ctx->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_bytes);
    DoReduce(tmp_storage.ptr());
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayMatrixColReduce<DeviceType::kGPU, T, binary_func> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 2) { return false; }
    if (y.shape().NumAxes() != 2) { return false; }
    return y.shape().At(0) == 1 && x.shape().At(1) == y.shape().At(1);
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int32_t num_rows = x.shape().ElemNum() / y.shape().ElemNum();
    int32_t num_cols = y.shape().ElemNum();
    RUN_CUDA_KERNEL((NdarrayMatrixColReduceNaiveCudaKernel<T, binary_func>), ctx, num_cols, y.ptr(),
                    x.ptr(), num_rows, num_cols);
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayXYZCubeYReduce<DeviceType::kGPU, T, binary_func> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return false;
    if (y.shape().ElemNum() > GetMaxVal<int32_t>()) { return false; }
    if (x.shape().NumAxes() != 3) { return false; }
    if (y.shape().NumAxes() != 3) { return false; }
    return x.shape().At(0) == y.shape().At(0) && y.shape().At(1) == 1
           && x.shape().At(2) == y.shape().At(2);
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    UNIMPLEMENTED();
  }
};

template<typename T, template<typename> class binary_func>
struct NdarrayXYZCubeXZReduce<DeviceType::kGPU, T, binary_func> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
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

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
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
          UnitOfBinaryFunc<T, binary_func>::Val(), ctx->cuda_stream());
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
struct NdarrayReduceCoreWrapper<DeviceType::kGPU, T, NDIMS, binary_func> final {
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis) {
    size_t n = x.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayReduceGpuInplaceReduceAxis<T, NDIMS, binary_func>), ctx, n, dst_reduced,
                    x, axis);
  }
};

#define INSTANTIATE_NDARRAY_REDUCE_IMPL(dtype, binary_func)                                       \
  template struct NdarrayScalarReduce<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype), binary_func>;    \
  template struct NdarrayMatrixRowReduce<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayMatrixColReduce<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype), binary_func>; \
  template struct NdarrayXYZCubeYReduce<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype), binary_func>;  \
  template struct NdarrayXYZCubeXZReduce<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype), binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 REDUCE_BINARY_FUNC_SEQ);

#define INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER(dtype_pair, NDIMS, binary_func)                   \
  template struct NdarrayReduceCoreWrapper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, \
                                           binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_CORE_WRAPPER,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, DIM_SEQ,
                                 REDUCE_BINARY_FUNC_SEQ);

}  // namespace oneflow
