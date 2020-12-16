#include "oneflow/user/kernels/ndindex_to_offset_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct NdIndexToOffsetFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int32_t in_num,
                  int32_t ndim, const T* index, const T* dims_tensor, T* out) {
    DoIndexToOffset<T>(in_num, ndim, index, dims_tensor, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDINDEX_TO_OFFSET_FUNCTOR, (DeviceType::kCPU),
                                 NDINDEX_TO_OFFSET_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
