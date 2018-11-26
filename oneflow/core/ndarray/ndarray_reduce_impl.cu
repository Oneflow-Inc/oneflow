#include <cub/cub.cuh>
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

struct RowOffsetFunctor final {
  OF_DEVICE_FUNC explicit RowOffsetFunctor(int32_t num_cols) : num_cols_(num_cols) {}
  OF_DEVICE_FUNC int32_t operator()(const int32_t& x) const { return x * num_cols_; }
  int32_t num_cols_;
};

template<typename T>
struct NdarrayMatrixRowReduce<DeviceType::kGPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    if (y.shape().ElemNum() > MaxVal<int32_t>()) { return false; }
    const auto& x_squeezed = SqueezeRight(x.shape());
    const auto& y_squeezed = SqueezeRight(y.shape());
    if (x_squeezed.NumAxes() == 0) { return false; }
    for (int i = 0; i < x_squeezed.NumAxes(); ++i) {
      if (x_squeezed.At(i) != y_squeezed.At(i)) { return false; }
    }
    CHECK_EQ(y.shape().ElemNum() % x.shape().ElemNum(), 0);
    return true;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int32_t num_rows = x.shape().ElemNum();
    int32_t num_cols = y.shape().ElemNum() / x.shape().ElemNum();
    RowOffsetFunctor get_row_offset(num_cols);
    cub::CountingInputIterator<int32_t> counting_intput_it(0);
    cub::TransformInputIterator<int32_t, RowOffsetFunctor, cub::CountingInputIterator<int32_t>>
        transform_input_iter(counting_intput_it, get_row_offset);
    size_t tmp_storage_size = 0;
    auto DoReduce = [&](T* tmp_storage_ptr) {
      int retcode = cub::DeviceSegmentedReduce::Reduce(
          tmp_storage_ptr, tmp_storage_size, x.ptr(), y.ptr(), num_rows, transform_input_iter,
          transform_input_iter + 1, cub::Sum(), ZeroVal<T>::value, ctx->cuda_stream());
      CHECK_EQ(retcode, 0) << "cub::DeviceSegmentedReduce::Reduce error";
    };
    DoReduce(nullptr);
    CHECK_GE(tmp_storage.shape().ElemNum() * sizeof(T), tmp_storage_size);
    DoReduce(tmp_storage.ptr());
  }

 private:
  static XpuShape SqueezeRight(const XpuShape& shape) {
    TODO();
    return shape;
  }
};

template<typename T>
struct NdarrayMatrixColReduce<DeviceType::kGPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    return false;
    TODO();
    const auto& x_squeezed = SqueezeLeft(x.shape());
    const auto& y_squeezed = SqueezeLeft(y.shape());
    if (x_squeezed.NumAxes() == 0) { return false; }
    for (int i = 0; i < x_squeezed.NumAxes(); ++i) {
      if (x_squeezed.At(x_squeezed.NumAxes() - 1 - i)
          != y_squeezed.At(y_squeezed.NumAxes() - 1 - i)) {
        return false;
      }
    }
    return true;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
  }

 private:
  static XpuShape SqueezeLeft(const XpuShape& shape) {
    TODO();
    return shape;
  }
};

#define INSTANTIATE_NDARRAY_REDUCE_IMPL(type_cpp, type_proto)         \
  template struct NdarrayMatrixRowReduce<DeviceType::kGPU, type_cpp>; \
  template struct NdarrayMatrixColReduce<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
