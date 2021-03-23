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

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

struct SubTskGphBuilderUtil {
  static constexpr int64_t kDistanceSameDevice = 0;
  static constexpr int64_t kDistanceSameMachine = 1;
  static constexpr int64_t kDistanceDiffMachine = 2;
  static constexpr int64_t kDistanceMax = 3;

  static bool IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc);
  static std::vector<TensorSliceView> GetTensorSliceView(int64_t parallel_num,
                                                         const SbpParallel& sbp_parallel,
                                                         const BlobDesc& blob_desc);
  static std::vector<TensorSliceView> GetTensorSliceView(
      const Shape& parallel_hierarchy, const ParallelDistribution& parallel_distribution,
      const Shape& logical_shape);
  static TensorSliceView GetTensorSliceView4ParallelRank(
      const Shape& parallel_hierarchy, const ParallelDistribution& parallel_distribution,
      const Shape& logical_shape, const std::vector<int64_t>& parallel_rank);
  static TensorSliceView GetTensorSliceView4ParallelId(
      const Shape& parallel_hierarchy, const ParallelDistribution& parallel_distribution,
      const Shape& logical_shape, int64_t parallel_id);
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
  static bool IsErrorBoxingNotSupported(const cfg::ErrorProto& error);
  static int64_t GetDistance(int64_t src_machine_id, int64_t src_dev_phy_id,
                             DeviceType src_device_type, int64_t dst_machine_id,
                             int64_t dst_dev_phy_id, DeviceType dst_device_type);
  static int64_t GetDistance(const ParallelDesc& src_parallel_desc, int64_t src_parallel_id,
                             const ParallelDesc& dst_parallel_desc, int64_t dst_parallel_id);
  static int64_t GetDistance(const TaskNode* src, const TaskNode* dst);

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

  static int64_t FindNearestSrcParallelId(const ParallelDesc& from_parallel_desc,
                                          const ParallelDesc& to_parallel_desc,
                                          int64_t to_parallel_id);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_UTIL_H_
