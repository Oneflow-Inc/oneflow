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
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/device/cpu_stream_index.h"
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_stream_index.h"
#endif

namespace oneflow {

namespace {

bool ContainsEmptySlice(const std::vector<TensorSliceView>& slices) {
  return std::any_of(slices.cbegin(), slices.cend(),
                     [](const TensorSliceView& slice) { return slice.IsEmpty(); });
}

bool IsCopyContiguous(const TensorSliceView& src, const TensorSliceView& dst) {
  CHECK_EQ(src.range_vec().size(), dst.range_vec().size());
  FOR_RANGE(int64_t, i, 1, src.range_vec().size()) {
    if (src.range_vec().at(i) != dst.range_vec().at(i)) { return false; }
  }
  return true;
}

bool IsSameDevice(const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                  const int64_t in_parallel_id, const int64_t out_parallel_id) {
  return in_pd.device_type() == out_pd.device_type()
         && CHECK_JUST(in_pd.DeviceId4ParallelId(in_parallel_id))
                == CHECK_JUST(out_pd.DeviceId4ParallelId(out_parallel_id));
}

}  // namespace

Maybe<SubTskGphBuilderStatus> SliceBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const cfg::SbpParallel& in_sbp_parallel,
    const cfg::SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if (SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)) {
    return Error::BoxingNotSupportedError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(in_parallel_desc)) {
    return Error::BoxingNotSupportedError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(out_parallel_desc)) {
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

  const auto GetBoxingGpuThrdId = [](int64_t machine_id, int64_t dev_id,
                                     const std::string& stream_name) -> int64_t {
    int64_t thrd_id = -1;
#ifdef WITH_CUDA
    DeviceId device_id{static_cast<DeviceId::rank_t>(machine_id), DeviceType::kGPU,
                       static_cast<DeviceId::device_index_t>(dev_id)};
    auto* generator = dynamic_cast<CudaStreamIndexGenerator*>(
        Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
    CHECK_NOTNULL(generator);
    StreamId::stream_index_t stream_index = 0;
    if (stream_name == "H2D") {
      stream_index = generator->GenerateH2DStreamIndex();
    } else if (stream_name == "D2H") {
      stream_index = generator->GenerateD2HStreamIndex();
    } else if (stream_name == "MIX") {
      stream_index = generator->GenerateNamedStreamIndex("MIX");
    } else {
      UNIMPLEMENTED();
    }
    thrd_id = SerializeStreamIdToInt64(StreamId{device_id, stream_index});
#else
    UNIMPLEMENTED();
#endif
    return thrd_id;
  };

  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
  const auto CreateBoxingNode121 = [&ctx, &lbi, &GetBoxingGpuThrdId](
                                       const ParallelDesc& pd, const int64_t parallel_id,
                                       const TensorSliceView& slice,
                                       SliceBoxingTaskMode mode) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    const int64_t machine_id = CHECK_JUST(pd.MachineId4ParallelId(parallel_id));
    int64_t thrd_id = -1;
    if (pd.device_type() == DeviceType::kCPU) {
      thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(machine_id);
    } else if (pd.device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
      int64_t dev_id = CHECK_JUST(pd.DeviceId4ParallelId(parallel_id));
      thrd_id = GetBoxingGpuThrdId(machine_id, dev_id, "H2D");
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  const auto BuildSubTaskGphS2B =
      [&ctx, &CreateBoxingNode121, &NewEdge, &lbi](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const cfg::SbpParallel& in_sbp,
          const cfg::SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingS2B(in_sbp, out_sbp));
        const std::vector<TensorSliceView> in_slices =
            GetTensorSliceView(in_pd.parallel_num(), in_sbp, blob_desc);
        CHECK(!ContainsEmptySlice(in_slices));
        const TensorSliceView& out_slice = GetBroadcastTensorSliceView(blob_desc);
        FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
          SliceBoxingTaskNode* out_node =
              CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
          FOR_RANGE(int64_t, in_id, 0, in_pd.parallel_num()) {
            const TensorSliceView& in_slice = in_slices.at(in_id);
            TaskNode* in_node = in_nodes.at(in_id);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                in_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), in_slice);
          }
          out_nodes->push_back(out_node);
        }
      };
  const auto BuildSubTaskGphS2S =
      [&ctx, &lbi, &CreateBoxingNode121, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const cfg::SbpParallel& in_sbp,
          const cfg::SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingS2S(in_sbp, out_sbp));
        const std::vector<TensorSliceView> in_slices =
            GetTensorSliceView(in_pd.parallel_num(), in_sbp, blob_desc);
        CHECK(!ContainsEmptySlice(in_slices));
        const std::vector<TensorSliceView> out_slices =
            GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
        CHECK(!ContainsEmptySlice(out_slices));
        for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
          const TensorSliceView& out_slice = out_slices.at(out_id);
          SliceBoxingTaskNode* out_node =
              CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
          for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
            const TensorSliceView& in_slice = in_slices.at(in_id);
            const TensorSliceView& intersection = out_slice.Intersect(in_slice);
            if (intersection.IsEmpty()) { continue; }
            TaskNode* in_node = in_nodes.at(in_id);
            SliceBoxingTaskNode* scatter_node =
                CreateBoxingNode121(in_pd, in_id, intersection, kSliceBoxingTaskModeCopy);
            scatter_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                scatter_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
          }
          out_nodes->push_back(out_node);
        }
      };
  const auto BuildSubTaskGphP2S =
      [&ctx, &lbi, &CreateBoxingNode121, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const cfg::SbpParallel& in_sbp,
          const cfg::SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingP2S(in_sbp, out_sbp));
        const TensorSliceView& in_slice = GetBroadcastTensorSliceView(blob_desc);
        const std::vector<TensorSliceView> out_slices =
            GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
        CHECK(!ContainsEmptySlice(out_slices));
        for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
          const TensorSliceView& out_slice = out_slices.at(out_id);
          SliceBoxingTaskNode* out_node =
              CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeAdd);
          for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
            const TensorSliceView& intersection = out_slice.Intersect(in_slice);
            if (intersection.IsEmpty()) { continue; }
            TaskNode* in_node = in_nodes.at(in_id);
            SliceBoxingTaskNode* scatter_node =
                CreateBoxingNode121(in_pd, in_id, intersection, kSliceBoxingTaskModeCopy);
            scatter_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                scatter_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), intersection);
          }
          out_nodes->push_back(out_node);
        }
      };

  const auto BuildSubTaskGphP2B =
      [&ctx, &lbi, &CreateBoxingNode121, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const cfg::SbpParallel& in_sbp,
          const cfg::SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingP2B(in_sbp, out_sbp));
        const TensorSliceView& slice = GetBroadcastTensorSliceView(blob_desc);
        for (int64_t out_id = 0; out_id < out_pd.parallel_num(); ++out_id) {
          SliceBoxingTaskNode* out_node =
              CreateBoxingNode121(out_pd, out_id, slice, kSliceBoxingTaskModeAdd);
          for (int64_t in_id = 0; in_id < in_pd.parallel_num(); ++in_id) {
            TaskNode* in_node = in_nodes.at(in_id);
            TaskNode* proxy_node = ctx->task_graph()->GetProxyNode(
                in_node, lbi, dynamic_cast<TaskNode*>(out_node)->MemZoneId121());
            out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), slice);
          }
          out_nodes->push_back(out_node);
        }
      };

  const auto BuildSubTaskGphB2S =
      [&ctx, &lbi, &CreateBoxingNode121, &NewEdge](
          const ParallelDesc& in_pd, const ParallelDesc& out_pd, const cfg::SbpParallel& in_sbp,
          const cfg::SbpParallel& out_sbp, const BlobDesc& blob_desc,
          const std::vector<TaskNode*>& in_nodes, std::vector<TaskNode*>* out_nodes) {
        CHECK(SubTskGphBuilderUtil::IsBoxingB2S(in_sbp, out_sbp));
        const TensorSliceView& in_slice = GetBroadcastTensorSliceView(blob_desc);
        const std::vector<TensorSliceView> out_slices =
            GetTensorSliceView(out_pd.parallel_num(), out_sbp, blob_desc);
        CHECK(!ContainsEmptySlice(out_slices));
        FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
          const TensorSliceView& out_slice = out_slices.at(out_id);
          const int64_t nearest_idx =
              SubTskGphBuilderUtil::FindNearestSrcParallelId(in_pd, out_pd, out_id);
          TaskNode* in_node = in_nodes.at(nearest_idx);
          SliceBoxingTaskNode* slice_node =
              CreateBoxingNode121(in_pd, nearest_idx, out_slice, kSliceBoxingTaskModeCopy);
          slice_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
          TaskNode* out_node = ctx->task_graph()->GetProxyNode(slice_node, lbi, out_pd, out_id);

          out_nodes->push_back(out_node);
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
      flat_blob_desc.mut_shape() = Shape({logical_blob_desc.shape().elem_cnt()});
      std::vector<TaskNode*> middle_nodes;
      cfg::SbpParallel middle_sbp;
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
