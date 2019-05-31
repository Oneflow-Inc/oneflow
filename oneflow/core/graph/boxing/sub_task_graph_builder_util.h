#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

struct SubTskGphBuilderUtil {
  static bool IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc);
  static ParallelDesc CloneParallelDescWithNewDeviceType(const ParallelDesc& src,
                                                         DeviceType device_type);
  static std::vector<TensorSliceView> GetTensorSliceView(int64_t parallel_num,
                                                         const SbpParallel& sbp_parallel,
                                                         const BlobDesc& blob_desc);
  static TensorSliceView GetBroadcastTensorSliceView(const BlobDesc& blob_desc);
  static bool IsOnSameGPU(const TaskNode* lhs, const TaskNode* rhs);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_
