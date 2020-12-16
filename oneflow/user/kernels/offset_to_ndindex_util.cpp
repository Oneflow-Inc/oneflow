#include "oneflow/user/kernels/offset_to_ndindex_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct OffsetToNdIndexFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int32_t in_num,
                  int32_t ndim, const T* index, T* dims_tensor, T* out) {
    DoOffsetToIndex<T>(in_num, ndim, index, dims_tensor, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_OFFSET_TO_NDINDEX__FUNCTOR, (DeviceType::kCPU),
                                 OFFSET_TO_NDINDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
