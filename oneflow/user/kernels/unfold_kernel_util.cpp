#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T, typename INDEX_T, int NDIM, int SDIM>
struct UnfoldKernelUtil<DeviceType::kCPU, T, INDEX_T, NDIM, SDIM> {
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  static void Forward(DeviceCtx* ctx, const UnfoldParams<INDEX_T, NDIM, SDIM>* raw_params, const T* input_ptr, T* output_ptr) {
    // const auto* params = static_cast<const ParamType*>(raw_params);
    for (INDEX_T out_offset = 0; out_offset < raw_params->out_elem_cnt; ++out_offset) {
      using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
      INDEX_T in_index[ParamType::kInputNDim] = {0};
      INDEX_T out_index[ParamType::kOutputNDim] = {0};
      raw_params->out_index_helper.OffsetToNdIndex(out_offset, out_index);
      if (!UnfoldIndexTransform<INDEX_T, NDIM, SDIM>(*raw_params, out_index, in_index)) {
        INDEX_T in_offset = raw_params->in_index_helper.NdIndexToOffset(in_index);
        output_ptr[out_offset] = input_ptr[in_offset];
      } else {
        output_ptr[out_offset] = static_cast<T>(kUnfoldPaddingValue);
      }
    }
  }
};

INSTANTIATE_UNFOLD_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCPU)

}  // namespace user_op

}  // namespace oneflow