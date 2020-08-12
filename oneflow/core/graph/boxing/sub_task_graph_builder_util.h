/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

struct SubTskGphBuilderStatus;

struct SubTskGphBuilderUtil {
  static constexpr int64_t kDistanceSameDevice = 0;
  static constexpr int64_t kDistanceSameMachine = 1;
  static constexpr int64_t kDistanceDiffMachine = 2;
  static constexpr int64_t kDistanceMax = 3;

  static bool IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc);
  static std::vector<TensorSliceView> GetTensorSliceView(int64_t parallel_num,
                                                         const SbpParallel& sbp_parallel,
                                                         const BlobDesc& blob_desc);
  static TensorSliceView GetBroadcastTensorSliceView(const BlobDesc& blob_desc);
  static bool HasEmptySliceIfSplit(int64_t parallel_num, const SbpParallel& sbp_parallel,
                                   const BlobDesc& blob_desc);
  static bool IsOnSameGPU(const TaskNode* lhs, const TaskNode* rhs);
  static bool IsBoxingS2S(const SbpParallel& src, const SbpParallel& dst);
  static bool IsBoxingS2B(const SbpParallel& src, const SbpParallel& dst);
  static bool IsBoxingP2S(const SbpParallel& src, const SbpParallel& dst);
  static bool IsBoxingP2B(const SbpParallel& src, const SbpParallel& dst);
  static bool IsBoxingB2B(const SbpParallel& src, const SbpParallel& dst);
  static bool IsBoxingB2S(const SbpParallel& src, const SbpParallel& dst);
  static bool BlobHasDynamicShape(const BlobDesc& blob_desc);
  static bool IsErrorBoxingNotSupported(const ErrorProto& error);
  static int64_t GetDistance(const TaskNode* src, const TaskNode* dst);
  static std::string SerializeSbpParallel(const SbpParallel& sbp_parallel);
  static std::string SerializeParallelDesc(const ParallelDesc& parallel_desc);
  static std::string SerializeLogicalBlobId(const LogicalBlobId& lbi);
  static std::string GetBlobInfo4LogicalBlobDesc(const BlobDesc& blob_desc);
  static std::string SubTskGphBuilderStatus2String(const SubTskGphBuilderStatus& status);
  static Maybe<SubTskGphBuilderStatus> BuildBoxingLogInfo(
      const CompTaskNode* src_node, const CompTaskNode* dst_node,
      const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
      const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc, const std::string& boxing_type);
  template<typename NodeType>
  static int64_t FindNearestNodeIndex(const std::vector<NodeType*> from_nodes,
                                      const NodeType* to_node) {
    CHECK(!from_nodes.empty());
    int64_t nearest_from_node_idx = -1;
    int64_t nearest_distance = SubTskGphBuilderUtil::kDistanceMax;
    for (int64_t i = 0; i < from_nodes.size(); ++i) {
      NodeType* from_node = from_nodes.at(i);
      int64_t distance = SubTskGphBuilderUtil::GetDistance(from_node, to_node);
      if (distance < nearest_distance) {
        nearest_from_node_idx = i;
        nearest_distance = distance;
      }
    }
    return nearest_from_node_idx;
  }

  template<typename NodeType>
  static NodeType* FindNearestNode(const std::vector<NodeType*> from_nodes,
                                   const NodeType* to_node) {
    const int64_t idx = FindNearestNodeIndex<NodeType>(from_nodes, to_node);
    return from_nodes.at(idx);
  }
};

struct SubTskGphBuilderStatus {
  std::string src_op_name_;
  std::string dst_op_name_;
  std::string src_parallel_conf_;
  std::string dst_parallel_conf_;
  std::string src_spb_parallel_;
  std::string dst_sbp_parallel_;
  std::string lbi_info_;
  std::string logical_blob_desc_info_;
  std::string boxing_type_;
};

#define OF_BOXING_LOGGER_COLNUM_NAME_FIELD                            \
  "src_op_name,src_parallel_conf,src_sbp_conf,lbi,logical_blob_desc," \
  "boxing_type,dst_op_name,dst_parallel_conf,dst_sbp_conf\n"

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_
