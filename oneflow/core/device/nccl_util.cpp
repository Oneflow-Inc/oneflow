#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

void NcclCheck(ncclResult_t error) { CHECK_EQ(error, ncclSuccess) << ncclGetErrorString(error); }

}  // namespace oneflow
