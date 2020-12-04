#include "oneflow/user/kernels/ravel_multi_index_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct RavelMultiIndexFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int32_t n, int32_t in_num,
                  const RavelMultiIndexHelper<T> helper, std::vector<const T*> in_dptrs, 
                  T* out) {
    DoIndexToOffset<T>(n, in_num, helper, in_dptrs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_MULTI_INDEX_FUNCTOR, (DeviceType::kCPU),
                                 RAVEL_MULTI_INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
