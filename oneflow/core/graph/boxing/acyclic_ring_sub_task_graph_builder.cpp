#include "oneflow/core/graph/boxing/acyclic_ring_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus AcyclicRingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!dst_parallel_desc.EqualsIgnoringPolicy(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const ParallelDesc& parallel_desc = src_parallel_desc;
  if (parallel_desc.device_type() != DeviceType::kGPU) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (parallel_desc.parallel_num() <= 1) { return SubTskGphBuilderStatus::MakeStatusError(); }
  const auto RingNextParallelId = [&parallel_desc](int64_t parallel_id) -> int64_t {
    return (parallel_id + 1) % parallel_desc.parallel_num();
  };
  const auto GetBoxingGpuThrdId = [](const int64_t dev_id, CudaWorkType work_type) -> int64_t {
    if (work_type == CudaWorkType::kBoxingH2D) {
      return Global<IDMgr>::Get()->GetGpuBoxingH2DThrdId(dev_id);
    } else if (work_type == CudaWorkType::kBoxingD2H) {
      return Global<IDMgr>::Get()->GetGpuBoxingD2HThrdId(dev_id);
    } else {
      return Global<IDMgr>::Get()->GetGpuMixThrdId(dev_id);
    }
  };
  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
  const auto CreateBoxingNode121 = [&ctx, &lbi, &GetBoxingGpuThrdId](
                                       const ParallelDesc& pd, const int64_t parallel_id,
                                       const TensorSliceView& slice,
                                       SliceBoxingTaskMode mode) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    const int64_t machine_id = pd.MachineIdForParallelId(parallel_id);
    int64_t thrd_id = -1;
    if (pd.device_type() == DeviceType::kCPU) {
      thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(machine_id);
    } else if (pd.device_type() == DeviceType::kGPU) {
      thrd_id = GetBoxingGpuThrdId(pd.DeviceIdForParallelId(parallel_id), CudaWorkType::kBoxingH2D);
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  const auto BuildSubTaskGphS2B = [&ctx, &CreateBoxingNode121, &NewEdge, &RingNextParallelId,
                                   &parallel_desc](const SbpParallel& in_sbp,
                                                   const SbpParallel& out_sbp,
                                                   const BlobDesc& blob_desc,
                                                   const std::vector<TaskNode*> in_nodes,
                                                   std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingS2B(in_sbp, out_sbp));
    const std::vector<TensorSliceView> in_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(parallel_desc.parallel_num(), in_sbp, blob_desc);
    const TensorSliceView out_slice = SubTskGphBuilderUtil::GetBroadcastTensorSliceView(blob_desc);
    std::vector<SliceBoxingTaskNode*> out_boxing_nodes;
    FOR_RANGE(int64_t, out_id, 0, parallel_desc.parallel_num()) {
      SliceBoxingTaskNode* out_node =
          CreateBoxingNode121(parallel_desc, out_id, out_slice, kSliceBoxingTaskModeCopy);
      out_boxing_nodes.push_back(out_node);
      out_nodes->push_back(out_node);
    }
    FOR_RANGE(int64_t, in_id, 0, parallel_desc.parallel_num()) {
      TaskNode* in_node = in_nodes.at(in_id);
      const TensorSliceView& in_slice = in_slices.at(in_id);
      out_boxing_nodes.at(in_id)->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
      int64_t send_id = in_id;
      int64_t recv_id = RingNextParallelId(send_id);
      TaskNode* send_node = in_node;
      while (recv_id != in_id) {
        SliceBoxingTaskNode* out_boxing_node = out_boxing_nodes.at(recv_id);
        if (send_node->machine_id() == out_boxing_node->machine_id()) {
          SliceBoxingTaskNode* recv_node =
              CreateBoxingNode121(parallel_desc, recv_id, in_slice, kSliceBoxingTaskModeCopy);
          recv_node->ConnectToSrcNodeWithSlice(send_node, NewEdge(), in_slice);
          out_boxing_node->ConnectToSrcNodeWithSlice(recv_node, NewEdge(), in_slice);
          send_node = recv_node;
        } else {
          TaskNode* recv_node =
              ctx->GetProxyNode(send_node, GetDefaultMemCase(send_node),
                                out_boxing_node->machine_id(), GetDefaultMemCase(out_boxing_node));
          out_boxing_node->ConnectToSrcNodeWithSlice(recv_node, NewEdge(), in_slice);
          send_node = recv_node;
        }
        send_id = recv_id;
        recv_id = RingNextParallelId(send_id);
      }
    }
  };
  const auto BuildSubTaskGphP2S = [&ctx, &CreateBoxingNode121, &NewEdge, &RingNextParallelId,
                                   &parallel_desc](const SbpParallel& in_sbp,
                                                   const SbpParallel& out_sbp,
                                                   const BlobDesc& blob_desc,
                                                   const std::vector<TaskNode*> in_nodes,
                                                   std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingP2S(in_sbp, out_sbp));
    const TensorSliceView in_slice = SubTskGphBuilderUtil::GetBroadcastTensorSliceView(blob_desc);
    const std::vector<TensorSliceView> out_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(parallel_desc.parallel_num(), out_sbp, blob_desc);
    FOR_RANGE(int64_t, out_id, 0, parallel_desc.parallel_num()) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      int64_t send_id = RingNextParallelId(out_id);
      SliceBoxingTaskNode* first_boxing_node =
          CreateBoxingNode121(parallel_desc, send_id, out_slice, kSliceBoxingTaskModeAdd);
      first_boxing_node->ConnectToSrcNodeWithSlice(in_nodes.at(send_id), NewEdge(), in_slice);
      TaskNode* send_node = first_boxing_node;
      while (send_id != out_id) {
        int64_t recv_id = RingNextParallelId(send_id);
        SliceBoxingTaskNode* recv_boxing_node =
            CreateBoxingNode121(parallel_desc, recv_id, out_slice, kSliceBoxingTaskModeAdd);
        recv_boxing_node->ConnectToSrcNodeWithSlice(in_nodes.at(recv_id), NewEdge(), in_slice);
        if (send_node->machine_id() == recv_boxing_node->machine_id()) {
          recv_boxing_node->ConnectToSrcNodeWithSlice(send_node, NewEdge(), out_slice);
        } else {
          TaskNode* proxy_node = ctx->GetProxyNode(send_node, GetDefaultMemCase(send_node),
                                                   recv_boxing_node->machine_id(),
                                                   GetDefaultMemCase(recv_boxing_node));
          recv_boxing_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), out_slice);
        }
        send_node = recv_boxing_node;
        send_id = recv_id;
      }
      out_nodes->push_back(send_node);
    }
  };
  std::vector<TaskNode*> in_nodes;
  in_nodes.assign(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());
  std::vector<TaskNode*> out_nodes;
  if (SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)
      && logical_blob_desc.shape().At(src_sbp_parallel.split_parallel().axis())
             >= parallel_desc.parallel_num()) {
    BuildSubTaskGphS2B(src_sbp_parallel, dst_sbp_parallel, logical_blob_desc, in_nodes, &out_nodes);
  } else if (SubTskGphBuilderUtil::IsBoxingP2S(src_sbp_parallel, dst_sbp_parallel)
             && logical_blob_desc.shape().At(dst_sbp_parallel.split_parallel().axis())
                    >= parallel_desc.parallel_num()) {
    BuildSubTaskGphP2S(src_sbp_parallel, dst_sbp_parallel, logical_blob_desc, in_nodes, &out_nodes);
  } else if (SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)
             && logical_blob_desc.shape().elem_cnt() >= parallel_desc.parallel_num()) {
    BlobDesc flat_blob_desc;
    flat_blob_desc.set_data_type(logical_blob_desc.data_type());
    flat_blob_desc.mut_shape() = Shape({logical_blob_desc.shape().elem_cnt()});
    std::vector<TaskNode*> middle_nodes;
    SbpParallel middle_sbp;
    middle_sbp.mutable_split_parallel()->set_axis(0);
    BuildSubTaskGphP2S(src_sbp_parallel, middle_sbp, flat_blob_desc, in_nodes, &middle_nodes);
    BuildSubTaskGphS2B(middle_sbp, dst_sbp_parallel, flat_blob_desc, middle_nodes, &out_nodes);
    for (TaskNode* out_node : out_nodes) {
      auto* slice_boxing_node = dynamic_cast<SliceBoxingTaskNode*>(out_node);
      CHECK_NOTNULL(slice_boxing_node);
      slice_boxing_node->SetOutShape(logical_blob_desc.shape());
    }
  } else {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  ctx->NaiveConnectAll121(out_nodes, sorted_dst_comp_tasks);
  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
