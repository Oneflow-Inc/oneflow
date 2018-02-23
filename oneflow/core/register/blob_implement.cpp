#include "oneflow/core/register/blob_implement.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T, int32_t NDIMS>
class BlobImplUtil<DeviceType::kCPU, T, NDIMS> final {
 public:
  static void DoTranspose(DeviceCtx* ctx, EigenTensor<T, NDIMS>* tensor,
                          EigenConstTensor<T, NDIMS>* const_tensor,
                          Eigen::array<int32_t, NDIMS>* p) {
    *tensor = const_tensor->shuffle((*p));
  }
};

#define INSTANTIATE_CPU_BLOB_IMPL_UTIL(data_type_pair, ndims) \
  template class BlobImplUtil<DeviceType::kCPU,               \
                              OF_PP_PAIR_FIRST(data_type_pair), ndims>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CPU_BLOB_IMPL_UTIL,
                                 ALL_DATA_TYPE_SEQ, DIM_SEQ)

}  // namespace oneflow
