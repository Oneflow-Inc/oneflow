#include "oneflow/core/ndarray/ndarray_reduce_impl.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

template<typename T>
struct NdarrayScalarReduce<DeviceType::kCPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) { return false; }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    UNIMPLEMENTED();
  }
};

template<typename T>
struct NdarrayMatrixRowReduce<DeviceType::kCPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) { return false; }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    UNIMPLEMENTED();
  }
};

template<typename T>
struct NdarrayMatrixColReduce<DeviceType::kCPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) { return false; }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_NDARRAY_REDUCE_IMPL(type_cpp, type_proto)         \
  template struct NdarrayScalarReduce<DeviceType::kCPU, type_cpp>;    \
  template struct NdarrayMatrixRowReduce<DeviceType::kCPU, type_cpp>; \
  template struct NdarrayMatrixColReduce<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_REDUCE_IMPL, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
