#include "oneflow/core/graph/boxing/inter_node_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

namespace {

void GroupParallelIdByMachine(const ParallelDesc& pd,
                              HashMap<int64_t, std::vector<int64_t>>* machine_id2parallel_ids) {
  FOR_RANGE(int64_t, parallel_id, 0, pd.parallel_num()) {
    (*machine_id2parallel_ids)[pd.MachineIdForParallelId(parallel_id)].push_back(parallel_id);
  }
}

}  // namespace

SubTskGphBuilderStatus InterNodeSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(dst_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const auto GetBoxingGpuThrdId = [](const int64_t dev_id) -> int64_t {
    return Global<IDMgr>::Get()->GetGpuMixThrdId(dev_id);
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
      thrd_id = GetBoxingGpuThrdId(pd.DeviceIdForParallelId(parallel_id));
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  const auto CreateBoxingNodeToHost =
      [&ctx, &lbi, &GetBoxingGpuThrdId, &NewEdge](
          TaskNode* src_node, const TensorSliceView& src_slice,
          const TensorSliceView& dst_slice) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* dst_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    int64_t thrd_id = -1;
    if (src_node->device_type() == DeviceType::kCPU) {
      thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(src_node->machine_id());
    } else if (src_node->device_type() == DeviceType::kGPU) {
      thrd_id = GetBoxingGpuThrdId(src_node->GpuPhyId());
    } else {
      UNIMPLEMENTED();
    }
    dst_node->Init(lbi, dst_slice, kSliceBoxingTaskModeCopy, src_node->machine_id(), thrd_id,
                   MakeHostMemCase());
    dst_node->ConnectToSrcNodeWithSlice(src_node, NewEdge(), src_slice);
    return dst_node;
  };
  const auto BuildSubTaskGphS2B = [&ctx, &logical_blob_desc, &CreateBoxingNode121, &NewEdge](
                                      const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                                      const SbpParallel& in_sbp, const SbpParallel& out_sbp,
                                      const std::vector<TaskNode*> in_nodes,
                                      std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingS2B(in_sbp, out_sbp));
    const std::vector<TensorSliceView> in_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(in_pd.parallel_num(), in_sbp, logical_blob_desc);
    const TensorSliceView out_slice =
        SubTskGphBuilderUtil::GetBroadcastTensorSliceView(logical_blob_desc);
    FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
      SliceBoxingTaskNode* out_node =
          CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
      FOR_RANGE(int64_t, in_id, 0, in_pd.parallel_num()) {
        const TensorSliceView& in_slice = in_slices.at(in_id);
        TaskNode* in_node = in_nodes.at(in_id);
        if (SubTskGphBuilderUtil::IsOnSameGPU(in_node, out_node)) {
          out_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
        } else {
          TaskNode* proxy_node = ctx->GetProxyNode(in_node, GetDefaultMemCase(in_node),
                                                   out_node->machine_id(), MakeHostMemCase());
          out_node->ConnectToSrcNodeWithSlice(proxy_node, NewEdge(), in_slice);
        }
      }
      out_nodes->push_back(out_node);
    }
  };
  const auto BuildSubTaskGphS2S = [&ctx, &lbi, &logical_blob_desc, &CreateBoxingNode121,
                                   &CreateBoxingNodeToHost, &GetBoxingGpuThrdId,
                                   &NewEdge](const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                                             const SbpParallel& in_sbp, const SbpParallel& out_sbp,
                                             const std::vector<TaskNode*> in_nodes,
                                             std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingS2S(in_sbp, out_sbp));
    const std::vector<TensorSliceView> in_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(in_pd.parallel_num(), in_sbp, logical_blob_desc);
    const std::vector<TensorSliceView> out_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(out_pd.parallel_num(), out_sbp, logical_blob_desc);
    HashMap<int64_t, std::vector<int64_t>> machine_id2in_parallel_ids;
    GroupParallelIdByMachine(in_pd, &machine_id2in_parallel_ids);
    FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      SliceBoxingTaskNode* out_node =
          CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeCopy);
      for (const auto& pair : machine_id2in_parallel_ids) {
        const int64_t in_machine_id = pair.first;
        const std::vector<int64_t>& in_parallel_ids = pair.second;
        if (out_node->machine_id() == in_machine_id) {
          for (const int64_t in_id : in_parallel_ids) {
            const TensorSliceView& in_slice = in_slices.at(in_id);
            TaskNode* in_node = in_nodes.at(in_id);
            if (in_pd.device_type() == DeviceType::kGPU
                && out_pd.device_type() == DeviceType::kCPU) {
              const TensorSliceView& intersection = out_slice.Intersect(in_slice);
              SliceBoxingTaskNode* copy_to_host =
                  CreateBoxingNodeToHost(in_node, in_slice, intersection);
              out_node->ConnectToSrcNodeWithSlice(copy_to_host, NewEdge(), intersection);
            } else {
              out_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
            }
          }
        } else {
          std::vector<TensorSliceView> intersections;
          for (const int64_t in_id : in_parallel_ids) {
            intersections.push_back(out_slice.Intersect(in_slices.at(in_id)));
          }
          const TensorSliceView concat_slice =
              TensorSliceView::Concatenate(intersections, in_sbp.split_parallel().axis());
          SliceBoxingTaskNode* local_concat_node =
              ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
          int64_t local_concat_thrd_id = -1;
          if (in_pd.device_type() == DeviceType::kCPU) {
            local_concat_thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(in_machine_id);
          } else if (in_pd.device_type() == DeviceType::kGPU) {
            local_concat_thrd_id = GetBoxingGpuThrdId(
                in_nodes.at(in_parallel_ids.at(out_id % in_parallel_ids.size()))->GpuPhyId());
          }
          local_concat_node->Init(lbi, concat_slice, kSliceBoxingTaskModeCopy, in_machine_id,
                                  local_concat_thrd_id, MakeHostMemCase());
          for (const int64_t in_id : in_parallel_ids) {
            local_concat_node->ConnectToSrcNodeWithSlice(in_nodes.at(in_id), NewEdge(),
                                                         in_slices.at(in_id));
          }
          TaskNode* local_add_proxy_node = ctx->GetProxyNode(
              local_concat_node, MakeHostMemCase(), out_node->machine_id(), MakeHostMemCase());
          out_node->ConnectToSrcNodeWithSlice(local_add_proxy_node, NewEdge(), concat_slice);
        }
      }
      out_nodes->push_back(out_node);
    }
  };
  const auto BuildSubTaskGphP2S = [&ctx, &lbi, &logical_blob_desc, &CreateBoxingNode121,
                                   &CreateBoxingNodeToHost, &GetBoxingGpuThrdId,
                                   &NewEdge](const ParallelDesc& in_pd, const ParallelDesc& out_pd,
                                             const SbpParallel& in_sbp, const SbpParallel& out_sbp,
                                             const std::vector<TaskNode*> in_nodes,
                                             std::vector<TaskNode*>* out_nodes) {
    CHECK(SubTskGphBuilderUtil::IsBoxingP2S(in_sbp, out_sbp));
    const TensorSliceView in_slice =
        SubTskGphBuilderUtil::GetBroadcastTensorSliceView(logical_blob_desc);
    const std::vector<TensorSliceView> out_slices =
        SubTskGphBuilderUtil::GetTensorSliceView(out_pd.parallel_num(), out_sbp, logical_blob_desc);
    HashMap<int64_t, std::vector<int64_t>> machine_id2in_parallel_ids;
    GroupParallelIdByMachine(in_pd, &machine_id2in_parallel_ids);
    FOR_RANGE(int64_t, out_id, 0, out_pd.parallel_num()) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      SliceBoxingTaskNode* out_node =
          CreateBoxingNode121(out_pd, out_id, out_slice, kSliceBoxingTaskModeAdd);
      for (const auto& pair : machine_id2in_parallel_ids) {
        const int64_t in_machine_id = pair.first;
        const std::vector<int64_t>& in_parallel_ids = pair.second;
        if (out_node->machine_id() == in_machine_id) {
          for (const int64_t in_id : in_parallel_ids) {
            TaskNode* in_node = in_nodes.at(in_id);
            if (in_pd.device_type() == DeviceType::kGPU
                && out_pd.device_type() == DeviceType::kCPU) {
              SliceBoxingTaskNode* copy_to_host =
                  CreateBoxingNodeToHost(in_node, in_slice, out_slice);
              out_node->ConnectToSrcNodeWithSlice(copy_to_host, NewEdge(), out_slice);
            } else {
              out_node->ConnectToSrcNodeWithSlice(in_node, NewEdge(), in_slice);
            }
          }
        } else {
          SliceBoxingTaskNode* local_add_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
          int64_t local_add_thrd_id = -1;
          if (in_pd.device_type() == DeviceType::kCPU) {
            local_add_thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(in_machine_id);
          } else if (in_pd.device_type() == DeviceType::kGPU) {
            local_add_thrd_id = GetBoxingGpuThrdId(
                in_nodes.at(in_parallel_ids.at(out_id % in_parallel_ids.size()))->GpuPhyId());
          }
          local_add_node->Init(lbi, out_slice, kSliceBoxingTaskModeAdd, in_machine_id,
                               local_add_thrd_id, MakeHostMemCase());
          for (const int64_t in_id : in_parallel_ids) {
            local_add_node->ConnectToSrcNodeWithSlice(in_nodes.at(in_id), NewEdge(), in_slice);
          }
          TaskNode* local_add_proxy_node = ctx->GetProxyNode(
              local_add_node, MakeHostMemCase(), out_node->machine_id(), MakeHostMemCase());
          out_node->ConnectToSrcNodeWithSlice(local_add_proxy_node, NewEdge(), out_slice);
        }
      }
      out_nodes->push_back(out_node);
    }
  };

  std::vector<TaskNode*> in_nodes;
  in_nodes.assign(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());
  std::vector<TaskNode*> out_nodes;
  if (SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)) {
    BuildSubTaskGphS2B(src_parallel_desc, dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                       in_nodes, &out_nodes);
  } else if (SubTskGphBuilderUtil::IsBoxingS2S(src_sbp_parallel, dst_sbp_parallel)) {
    BuildSubTaskGphS2S(src_parallel_desc, dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                       in_nodes, &out_nodes);
  } else if (SubTskGphBuilderUtil::IsBoxingP2S(src_sbp_parallel, dst_sbp_parallel)) {
    BuildSubTaskGphP2S(src_parallel_desc, dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                       in_nodes, &out_nodes);
  } else if (SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
    std::vector<TaskNode*> middle_nodes;
    SbpParallel middle_sbp;
    middle_sbp.mutable_split_parallel()->set_axis(0);
    BuildSubTaskGphP2S(src_parallel_desc, dst_parallel_desc, src_sbp_parallel, middle_sbp, in_nodes,
                       &middle_nodes);
    BuildSubTaskGphS2B(dst_parallel_desc, dst_parallel_desc, middle_sbp, dst_sbp_parallel,
                       middle_nodes, &out_nodes);
  } else {
    UNIMPLEMENTED();
  }
  CHECK_EQ(out_nodes.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int64_t, i, 0, sorted_dst_comp_tasks.size()) {
    Connect<TaskNode>(out_nodes.at(i), NewEdge(), sorted_dst_comp_tasks.at(i));
  }
  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
