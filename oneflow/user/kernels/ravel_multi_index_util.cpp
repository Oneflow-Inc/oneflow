#include "oneflow/user/kernels/ravel_multi_index_util.h"

namespace oneflow {

namespace user_op {
template<typename T>
struct RavelMultiIndexFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, user_op::KernelComputeContext* kernel_ctx, int32_t n, int32_t in_num,
                  int32_t ndim, const Tensor* dims_tensor, T* out) {
    const T* dims = dims_tensor->dptr<T>();
    std::cout<<"Helper Ndim is: "<<ndim<<std::endl;
    RavelMultiIndexHelper<T> helper(dims, ndim);

    const T* in_dptrs[6]; // The max input num is 6
    // TODO: Add a check to promise the input num is less than the max legal input num
    for (int32_t i = 0; i < in_num; ++i) {
      std::cout<<"Current Loop idx is: "<<i<<std::endl;
      // Can simplize
      in_dptrs[i] = kernel_ctx->Tensor4ArgNameAndIndex("multi_index", i)->dptr<T>();
    }

    DoIndexToOffset<T>(n, in_num, helper, in_dptrs, out);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_MULTI_INDEX_FUNCTOR, (DeviceType::kCPU),
                                 RAVEL_MULTI_INDEX_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
