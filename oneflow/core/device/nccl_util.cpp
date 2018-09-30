#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

#ifdef WITH_NCCL
void NcclCheck(ncclResult_t error) { CHECK_EQ(error, ncclSuccess) << ncclGetErrorString(error); }
#endif  // WITH_NCCL

}  // namespace oneflow
