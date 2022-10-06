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
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/ep/include/primitive/copy_nd.h"

namespace oneflow {

namespace {

bool IsCopyNdPrimitiveSupported(DeviceType device_type, int64_t ndims) {
  auto primitive = ep::primitive::NewPrimitive<ep::primitive::CopyNdFactory>(device_type, ndims);
  return primitive.operator bool();
}

}  // namespace

Maybe<SubTskGphBuilderStatus> SliceBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if (!IsCopyNdPrimitiveSupported(in_parallel_desc.device_type(),
                                  logical_blob_desc.shape().NumAxes())) {
    return Error::BoxingNotSupportedError();
  }
  if (!IsCopyNdPrimitiveSupported(out_parallel_desc.device_type(),
                                  logical_blob_desc.shape().NumAxes())) {
    return Error::BoxingNotSupportedError();
  }
  if (SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)) {
    return Error::BoxingNotSupportedError();
  }
  if (SubTskGphBuilderUtil::HasEmptySliceIfSplit(in_parallel_desc.parallel_num(), in_sbp_parallel,
                                                 logical_blob_desc)) {
    return Error::BoxingNotSupportedError();
  }
  if (SubTskGphBuilderUtil::HasEmptySliceIfSplit(out_parallel_desc.parallel_num(), out_sbp_parallel,
                                                 logical_blob_desc)) {
    return Error::BoxingNotSupportedError();
  }
  if (!(SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)
        || SubTskGphBuilderUtil::IsBoxingS2S(in_sbp_parallel, out_sbp_parallel)
        || SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)
        || SubTskGphBuilderUtil::IsBoxingP2B(in_sbp_parallel, out_sbp_parallel)
        || SubTskGphBuilderUtil::IsBoxingB2S(in_sbp_parallel, out_sbp_parallel))) {
    return Error::BoxingNotSupportedError();
  }

  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
  const auto CreateSliceBoxingNode =
      [&ctx, &lbi](const ParallelDesc& pd, const int64_t parallel_id, const TensorSliceView& slice,
                   SliceBoxingTaskMode mode) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    const int64_t machine_id = CHECK_JUST(pd.MachineId4ParallelId(parallel_id));
    int64_t device_index = (pd.device_type() == DeviceType::kCPU)
                               ? 0
                               : CHECK_JUST(pd.DeviceId4ParallelId(parallel_id));
    int64_t thrd_id = EncodeStreamIdToInt64(
        GenerateComputeTaskStreamId(machine_id, pd.device_type(), device_index));
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  const auto GetSliceCopyNode = [&CreateSliceBoxingNode, &NewEdge](
                                    TaskNode* in_node, const TensorSliceView& in_slice,
                                    const ParallelDesc& in_pd, const int64_t in_id,
                                    const TensorSliceView& intersection) -> TaskNode* {
    if (in_slice == intersection) {
      return in_node;
    } else {
      SliceBoxingTaskNode* slice_copy_node =
          CreateSliceBoxingNode(in_pd, in_id, intersection, kSliceBoxingTaskModeCopy);
      slice_copy_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
      return slice_copy_node;
    }
  };
  const auto BuildSubTaskGphS2B =
      [&ctx, &CreateSliceBoxingNode, &NewEdge, &lbi](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const SbpParallel& in_sbp,
          const SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingS2B(in_sbp, out_sbp));
        const std::vector<TensorSliceView> in_slices =
            GetTensorSliceView(in_pd.parallel_num(), in_sbp, blob_desc);
        const TensorSliceView& out_slice = GetBroadcastTensorSliceView(blob_desc);
        FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
          SliceBoxingTaskNode* out_node =
              CreateSliceBoxingNode(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
          FOR_RANGE(int64_t, in_id, 0, in_pd.parallel_num()) {
            const TensorSliceView& in_slice = in_slices.at(in_id);
            TaskNode* in_node = in_nodes.at(in_id);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                in_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), in_slice);
          }
          out_nodes->emplace_back(out_node);
        }
      };
  const auto BuildSubTaskGphS2S = [&ctx, &lbi, &CreateSliceBoxingNode, &GetSliceCopyNode, &NewEdge](
                                      const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                                      const SbpParallel& in_sbp, const SbpParallel& out_sbp,
                                      const BlobDesc& blob_desc,
                                      const std::vector<TaskNode*>& in_nodes,
                                      std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingS2S(in_sbp, out_sbp));
    const std::vector<TensorSliceView> in_slices =
        GetTensorSliceView(in_pd.parallel_num(), in_sbp, blob_desc);
    const std::vector<TensorSliceView> out_slices =
        GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
    for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      SliceBoxingTaskNode* out_node =
          CreateSliceBoxingNode(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
      for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
        const TensorSliceView& in_slice = in_slices.at(in_id);
        const TensorSliceView& intersection = out_slice.Intersect(in_slice);
        if (intersection.IsEmpty()) { continue; }
        TaskNode* in_node = in_nodes.at(in_id);
        TaskNode* slice_copy_node = GetSliceCopyNode(in_node, in_slice, in_pd, in_id, intersection);
        TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            slice_copy_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
        out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
      }
      out_nodes->emplace_back(out_node);
    }
  };
  const auto BuildSubTaskGphP2S = [&ctx, &lbi, &CreateSliceBoxingNode, &GetSliceCopyNode, &NewEdge](
                                      const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                                      const SbpParallel& in_sbp, const SbpParallel& out_sbp,
                                      const BlobDesc& blob_desc,
                                      const std::vector<TaskNode*>& in_nodes,
                                      std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingP2S(in_sbp, out_sbp));
    const TensorSliceView& in_slice = GetBroadcastTensorSliceView(blob_desc);
    const std::vector<TensorSliceView> out_slices =
        GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
    for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      SliceBoxingTaskNode* out_node =
          CreateSliceBoxingNode(out_pd, out_id, out_slice, kSliceBoxingTaskModeAdd);
      for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
        const TensorSliceView& intersection = out_slice.Intersect(in_slice);
        if (intersection.IsEmpty()) { continue; }
        TaskNode* in_node = in_nodes.at(in_id);
        TaskNode* slice_copy_node = GetSliceCopyNode(in_node, in_slice, in_pd, in_id, intersection);
        TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
            slice_copy_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
        out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
      }
      out_nodes->emplace_back(out_node);
    }
  };

  const auto BuildSubTaskGphP2B =
      [&ctx, &lbi, &CreateSliceBoxingNode, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const SbpParallel& in_sbp,
          const SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingP2B(in_sbp, out_sbp));
        const TensorSliceView& slice = GetBroadcastTensorSliceView(blob_desc);
        for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
          SliceBoxingTaskNode* out_node =
              CreateSliceBoxingNode(out_pd, out_id, slice, kSliceBoxingTaskModeAdd);
          for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
            TaskNode* in_node = in_nodes.at(in_id);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                in_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), slice);
          }
          out_nodes->emplace_back(out_node);
        }
      };

  const auto BuildSubTaskGphB2S =
      [&ctx, &lbi, &CreateSliceBoxingNode, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const SbpParallel& in_sbp,
          const SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingB2S(in_sbp, out_sbp));
        const TensorSliceView& in_slice = GetBroadcastTensorSliceView(blob_desc);
        const std::vector<TensorSliceView> out_slices =
            GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
        FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
          const TensorSliceView& out_slice = out_slices.at(out_id);
          const int64_t nearest_idx =
              SubTskGphBuilderUtil::FindNearestSrcParallelId(in_pd, out_pd, out_id);
          TaskNode* in_node = in_nodes.at(nearest_idx);
          SliceBoxingTaskNode* slice_node =
              CreateSliceBoxingNode(in_pd, nearest_idx, out_slice, kSliceBoxingTaskModeCopy);
          slice_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
          TaskNode* out_node = ctx->task_graph()->GetProxyNode(slice_node, lbi, out_pd, out_id);

          out_nodes->emplace_back(out_node);
        }
      };

  std::string comment;
  if (SubTskGphBuilderUtil::IsBoxingS2B(in_sbp_parallel, out_sbp_parallel)) {
    BuildSubTaskGphS2B(in_parallel_desc, out_parallel_desc, in_sbp_parallel, out_sbp_parallel,
                       logical_blob_desc, sorted_in_tasks, sorted_out_tasks);
    comment = "BuildSubTaskGphS2B";
  } else if (SubTskGphBuilderUtil::IsBoxingS2S(in_sbp_parallel, out_sbp_parallel)) {
    BuildSubTaskGphS2S(in_parallel_desc, out_parallel_desc, in_sbp_parallel, out_sbp_parallel,
                       logical_blob_desc, sorted_in_tasks, sorted_out_tasks);
    comment = "BuildSubTaskGphS2S";
  } else if (SubTskGphBuilderUtil::IsBoxingP2S(in_sbp_parallel, out_sbp_parallel)) {
    BuildSubTaskGphP2S(in_parallel_desc, out_parallel_desc, in_sbp_parallel, out_sbp_parallel,
                       logical_blob_desc, sorted_in_tasks, sorted_out_tasks);
    comment = "BuildSubTaskGphP2S";
  } else if (SubTskGphBuilderUtil::IsBoxingP2B(in_sbp_parallel, out_sbp_parallel)) {
    if (logical_blob_desc.shape().elem_cnt() < out_parallel_desc.parallel_num()) {
      BuildSubTaskGphP2B(in_parallel_desc, out_parallel_desc, in_sbp_parallel, out_sbp_parallel,
                         logical_blob_desc, sorted_in_tasks, sorted_out_tasks);
      comment = "BuildSubTaskGphP2B";
    } else {
      BlobDesc flat_blob_desc(logical_blob_desc.data_type());
      flat_blob_desc.set_shape(Shape({logical_blob_desc.shape().elem_cnt()}));
      std::vector<TaskNode*> middle_nodes;
      SbpParallel middle_sbp;
      middle_sbp.mutable_split_parallel()->set_axis(0);
      BuildSubTaskGphP2S(in_parallel_desc, out_parallel_desc, in_sbp_parallel, middle_sbp,
                         flat_blob_desc, sorted_in_tasks, &middle_nodes);
      BuildSubTaskGphS2B(out_parallel_desc, out_parallel_desc, middle_sbp, out_sbp_parallel,
                         flat_blob_desc, middle_nodes, sorted_out_tasks);
      comment = "BuildSubTaskGphP2S->BuildSubTaskGphS2B";
      for (TaskNode* out_node : *sorted_out_tasks) {
        auto* slice_boxing_node = dynamic_cast<SliceBoxingTaskNode*>(out_node);
        CHECK_NOTNULL(slice_boxing_node);
        slice_boxing_node->SetOutShape(logical_blob_desc.shape());
      }
    }

  } else if (SubTskGphBuilderUtil::IsBoxingB2S(in_sbp_parallel, out_sbp_parallel)) {
    BuildSubTaskGphB2S(in_parallel_desc, out_parallel_desc, in_sbp_parallel, out_sbp_parallel,
                       logical_blob_desc, sorted_in_tasks, sorted_out_tasks);
    comment = "BuildSubTaskGphB2S";
  } else {
    UNIMPLEMENTED();
  }
  return TRY(BuildSubTskGphBuilderStatus("SliceBoxingSubTskGphBuilder", comment));
}

}  // namespace oneflow
