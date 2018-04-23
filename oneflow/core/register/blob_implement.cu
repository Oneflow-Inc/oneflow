#include "oneflow/core/register/blob_implement.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T, int32_t NDIMS>
struct BlobImplUtil<DeviceType::kGPU, T, NDIMS> {
  static void DoTranspose(DeviceCtx* ctx, EigenTensor<T, NDIMS>* tensor,
                          EigenConstTensor<T, NDIMS>* const_tensor,
                          const PbRf<int32_t>& permutation) {
    Eigen::array<int32_t, NDIMS> p;
    for (int32_t i = 0; i < NDIMS; ++i) { p[i] = permutation[i]; }
    tensor->device(ctx->eigen_gpu_device()) = const_tensor->shuffle(p);
  }
};

#define INSTANTIATE_CPU_BLOB_IMPL_UTIL(data_type_pair, ndims) \
  template struct BlobImplUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), ndims>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CPU_BLOB_IMPL_UTIL, ALL_DATA_TYPE_SEQ, DIM_SEQ)
}  // namespace oneflow
