#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

#ifdef WITH_CUDA

void NcclCheck(ncclResult_t error) { CHECK_EQ(error, ncclSuccess) << ncclGetErrorString(error); }

std::string NcclUniqueIdToString(const ncclUniqueId& unique_id) {
  return std::string(unique_id.internal, NCCL_UNIQUE_ID_BYTES);
}

void NcclUniqueIdFromString(const std::string& str, ncclUniqueId* unique_id) {
  CHECK_EQ(str.size(), NCCL_UNIQUE_ID_BYTES);
  memcpy(unique_id->internal, str.data(), NCCL_UNIQUE_ID_BYTES);
}

#endif  // WITH_CUDA

}  // namespace oneflow
