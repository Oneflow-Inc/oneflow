// TODO： std::vector Bug


// #ifdef WITH_CUDA
// #include "oneflow/core/framework/framework.h"
// #include "oneflow/user/kernels/ravel_multi_index_util.h"

// namespace oneflow {

// namespace user_op {

// template<typename T>
// __global__ void RavelMultiIndexForwardGpuKernel(int32_t n, int32_t in_num, 
//     const RavelMultiIndexHelper<T> helper, std::vector<const T*> in_dptrs, T* out) {
//   DoIndexToOffset<T>(n, in_num, helper, in_dptrs, out);
// }

// template<typename T>
// struct RavelMultiIndexFunctor<DeviceType::kGPU, T> final {
//   void operator()(DeviceCtx* ctx, int32_t n, int32_t in_num,
//                 const RavelMultiIndexHelper<T> helper, std::vector<const T*> in_dptrs, 
//                 T* out) {
//     RUN_CUDA_KERNEL((RavelMultiIndexForwardGpuKernel<T>), ctx, in_num, n, in_num, helper, in_dptrs, out);
//   }
// };


// OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_RAVEL_MULTI_INDEX_FUNCTOR, (DeviceType::kGPU),
//                                  RAVEL_MULTI_INDEX_DATA_TYPE_SEQ);
// }  // namespace user_op
// }  // namespace oneflow

// #endif  // End WITH_CUDA