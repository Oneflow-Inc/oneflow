#include "oneflow/user/kernels/ravel_index_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct RavelIndexFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int32_t in_num,
                  int32_t ndim, const T* index, const T* dims_tensor, T* out) {
    std::cout<<"CPP Enter DOIndexToOffset, ndim is: "<<ndim<<std::endl;
    DoIndexToOffset<T>(in_num, ndim, index, dims_tensor, out);

  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_INDEX_FUNCTOR, (DeviceType::kCPU),
                                 RAVEL_INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
