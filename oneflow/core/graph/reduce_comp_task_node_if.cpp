#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/graph/reduce_comp_task_node_if.h"

namespace oneflow {
int64_t InferRegstSize(const RegstDesc& regst) {
  return RtBlobDesc(*(regst.GetBlobDesc(GenPackedLbi()))).ByteSizeOfDataContentField();
}

}  // namespace oneflow