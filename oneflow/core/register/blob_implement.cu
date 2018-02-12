#include "oneflow/core/register/blob_implement.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

#define INSTANTIATE_GPU_BLOB_IMPL(data_type_pair, ndims)           \
  template class BlobImpl<OF_PP_PAIR_FIRST(data_type_pair), ndims, \
                          DeviceType::kGPU>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GPU_BLOB_IMPL, ALL_DATA_TYPE_SEQ,
                                 DIM_SEQ)

}  // namespace oneflow
