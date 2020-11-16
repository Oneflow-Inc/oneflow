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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing_s2s_all2all_pack_compute_task_node.h"
#include "oneflow/core/graph/boxing_s2s_all2all_unpack_compute_task_node.h"
#ifdef WITH_CUDA
#include <nccl.h>
#endif

namespace oneflow {

using namespace boxing::collective;

namespace {

void NcclInitCollectiveNode(CollectiveBoxingGenericTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
  OperatorConf op_conf;
  op_conf.set_name(name);
  op_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(DeviceType::kGPU)));
  CollectiveBoxingGenericOpConf* conf = op_conf.mutable_collective_boxing_generic_conf();
  *conf->mutable_lbi() = lbi;
  RankDesc* rank_desc = conf->mutable_rank_desc();
  OpDesc* op_desc = rank_desc->mutable_op_desc();
  op_desc->set_name(name);
  op_desc->set_op_type(op_type);
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeReduceScatter
      || op_type == OpType::kOpTypeReduce) {
    op_desc->set_reduce_method(ReduceMethod::kReduceMethodSum);
  }
  op_desc->set_data_type(logical_blob_desc.data_type());
  logical_blob_desc.shape().ToProto(op_desc->mutable_shape());
  op_desc->set_num_ranks(parallel_desc.parallel_num());
  if (op_type == OpType::kOpTypeBroadcast || op_type == OpType::kOpTypeReduce) {
    CHECK_GE(root, 0);
    CHECK_LT(root, parallel_desc.parallel_num());
    op_desc->set_root(root);
  } else {
    CHECK_EQ(root, -1);
  }
  op_desc->set_backend(Backend::kBackendNCCL);
  rank_desc->set_rank(parallel_id);

  const int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
  const int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
  const int64_t thrd_id = Global<IDMgr>::Get()->GetGpuNcclThrdId(device_id);
  node->Init(machine_id, thrd_id, NewAreaId(), op_conf);
}

int64_t FindRootParallelId(const ParallelDesc& multi_device, const ParallelDesc& sole_device) {
  CHECK_EQ(sole_device.parallel_num(), 1);
  const int64_t root_machine_id = CHECK_JUST(sole_device.MachineId4ParallelId(0));
  const int64_t root_device_id = CHECK_JUST(sole_device.DeviceId4ParallelId(0));
  int64_t root_parallel_id = -1;
  FOR_RANGE(int64_t, i, 0, multi_device.parallel_num()) {
    if (CHECK_JUST(multi_device.MachineId4ParallelId(i)) == root_machine_id
        && CHECK_JUST(multi_device.DeviceId4ParallelId(i)) == root_device_id) {
      root_parallel_id = i;
      break;
    }
  }
  return root_parallel_id;
}

bool IsSourceTimeShape(const Shape& shape) {
  return shape.elem_cnt() == GlobalJobDesc().TotalBatchNum() * GlobalJobDesc().NumOfPiecesInBatch();
}

class NcclCollectiveBoxingAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAllReduceSubTskGphBuilder);
  NcclCollectiveBoxingAllReduceSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingAllReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (dst_parallel_desc.Equals(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllReduce-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, src_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllReduce, -1);
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingAllReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class NcclCollectiveBoxingReduceScatterSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingReduceScatterSubTskGphBuilder);
  NcclCollectiveBoxingReduceScatterSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingReduceScatterSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (dst_parallel_desc.Equals(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0
        && SubTskGphBuilderUtil::IsBoxingP2S(src_sbp_parallel, dst_sbp_parallel)
        && dst_sbp_parallel.split_parallel().axis() == 0) {
      const std::string op_name =
          "System-Boxing-NcclCollectiveBoxingReduceScatter-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, src_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduceScatter, -1);
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class NcclCollectiveBoxingAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAllGatherSubTskGphBuilder);
  NcclCollectiveBoxingAllGatherSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (dst_parallel_desc.EqualsIgnoringDeviceType(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(src_parallel_desc)
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0
        && SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)
        && src_sbp_parallel.split_parallel().axis() == 0) {
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllGather-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_comp_task = sorted_src_comp_tasks.at(i);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
        TaskNode* src_node = ctx->GetProxyNode(src_comp_task, src_comp_task->MemZoneId121(),
                                               dst_node->machine_id(), dst_node->MemZoneId121());
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, dst_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllGather, -1);
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingAllGatherSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class NcclCollectiveBoxingReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingReduceSubTskGphBuilder);
  NcclCollectiveBoxingReduceSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (src_parallel_desc.parallel_num() > 1 && dst_parallel_desc.parallel_num() == 1
        && src_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && src_sbp_parallel.has_partial_sum_parallel()) {
      const int64_t root_parallel_id = FindRootParallelId(src_parallel_desc, dst_parallel_desc);
      if (root_parallel_id == -1) { return Error::BoxingNotSupportedError(); }

      const std::string op_name = "System-Boxing-NcclCollectiveBoxingReduce-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, src_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeReduce, root_parallel_id);
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), collective_node);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.front();
        if (i == root_parallel_id) {
          Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
        } else {
          collective_node->BuildCtrlRegstDesc(dst_node);
          Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
        }
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingReduceSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder);
  CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder() = default;
  ~CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (src_parallel_desc.parallel_num() == 1 && dst_parallel_desc.parallel_num() > 1
        && src_parallel_desc.device_type() == DeviceType::kCPU
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && logical_blob_desc.shape().elem_cnt() >= 1024
        && dst_sbp_parallel.has_broadcast_parallel()
        // a potential optimization: flat the blob and then relax this requirement
        && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0) {
      const TensorSliceView in_slice =
          SubTskGphBuilderUtil::GetBroadcastTensorSliceView(logical_blob_desc);
      SbpParallel split_sbp_parallel;
      split_sbp_parallel.mutable_split_parallel()->set_axis(0);
      std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensorSliceView(
          dst_parallel_desc.parallel_num(), split_sbp_parallel, logical_blob_desc);
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllGather-" + NewUniqueId();
      FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
        const TensorSliceView& out_slice = out_slices.at(out_id);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(out_id);
        CompTaskNode* src_node =
            SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
        SliceBoxingTaskNode* slice_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
        // slice on cpu
        const auto src_machine_id = CHECK_JUST(src_parallel_desc.MachineId4ParallelId(0));
        slice_node->Init(lbi, out_slice, kSliceBoxingTaskModeCopy, src_machine_id,
                         Global<IDMgr>::Get()->PickCpuThrdIdEvenly(src_machine_id));
        slice_node->ConnectToSrcNodeWithSlice(src_node, ctx->task_graph()->NewEdge(), in_slice);
        // copy to dst gpu
        TaskNode* slice_node_proxy =
            ctx->GetProxyNode(slice_node, slice_node->MemZoneId121(), dst_node->machine_id(),
                              dst_node->MemZoneId121());
        // allgather
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, dst_parallel_desc, out_id, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllGather, -1);
        Connect<TaskNode>(slice_node_proxy, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  };
};

class NcclCollectiveBoxingBroadcastSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingBroadcastSubTskGphBuilder);
  NcclCollectiveBoxingBroadcastSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingBroadcastSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (src_parallel_desc.parallel_num() == 1 && dst_parallel_desc.parallel_num() > 1
        && (src_parallel_desc.device_type() == DeviceType::kGPU
            || (src_parallel_desc.device_type() == DeviceType::kCPU
                && logical_blob_desc.shape().elem_cnt() >= 1024))
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && dst_sbp_parallel.has_broadcast_parallel()) {
      TaskNode* gpu_src_node = nullptr;
      int64_t root_parallel_id = -1;
      if (src_parallel_desc.device_type() == DeviceType::kCPU) {
        auto* cpu_src_node = sorted_src_comp_tasks.front();
        root_parallel_id =
            SubTskGphBuilderUtil::FindNearestNodeIndex(sorted_dst_comp_tasks, cpu_src_node);
        auto* nearest_dst_node = sorted_dst_comp_tasks.at(root_parallel_id);
        gpu_src_node =
            ctx->GetProxyNode(cpu_src_node, cpu_src_node->MemZoneId121(),
                              nearest_dst_node->machine_id(), nearest_dst_node->MemZoneId121());
      } else if (src_parallel_desc.device_type() == DeviceType::kGPU) {
        root_parallel_id = FindRootParallelId(dst_parallel_desc, src_parallel_desc);
        gpu_src_node = sorted_src_comp_tasks.front();
      } else {
        return Error::BoxingNotSupportedError();
      }
      if (root_parallel_id == -1) { return Error::BoxingNotSupportedError(); }

      const std::string op_name = "System-Boxing-NcclCollectiveBoxingBroadcast-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, dst_parallel_desc.parallel_num()) {
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, dst_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeBroadcast, root_parallel_id);
        if (i == root_parallel_id) {
          Connect<TaskNode>(gpu_src_node, ctx->task_graph()->NewEdge(), collective_node);
        } else {
          gpu_src_node->BuildCtrlRegstDesc(collective_node);
          Connect<TaskNode>(gpu_src_node, ctx->task_graph()->NewEdge(), collective_node);
        }
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingBroadcastSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

class NcclCollectiveBoxingAll2AllSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAll2AllSubTskGphBuilder);
  NcclCollectiveBoxingAll2AllSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingAll2AllSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                                      const ParallelDesc& src_parallel_desc,
                                      const ParallelDesc& dst_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const SbpParallel& src_sbp_parallel,
                                      const SbpParallel& dst_sbp_parallel) const override {
    if (dst_parallel_desc.EqualsIgnoringDeviceType(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && src_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && logical_blob_desc.shape().At(src_sbp_parallel.split_parallel().axis())
                   % src_parallel_desc.parallel_num()
               == 0
        && logical_blob_desc.shape().At(dst_sbp_parallel.split_parallel().axis())
                   % dst_parallel_desc.parallel_num()
               == 0
        && src_sbp_parallel.split_parallel().axis() != dst_sbp_parallel.split_parallel().axis()
        && SubTskGphBuilderUtil::IsBoxingS2S(src_sbp_parallel, dst_sbp_parallel)) {
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAll2All-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);

        BoxingS2SAll2AllPackCompTaskNode* pack_node =
            ctx->task_graph()->NewNode<BoxingS2SAll2AllPackCompTaskNode>();
        pack_node->Init(src_node, lbi, dst_sbp_parallel.split_parallel().axis());
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), pack_node);

        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, dst_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAll2All, -1);
        Connect<TaskNode>(pack_node, ctx->task_graph()->NewEdge(), collective_node);

        BoxingS2SAll2AllUnpackCompTaskNode* unpack_node =
            ctx->task_graph()->NewNode<BoxingS2SAll2AllUnpackCompTaskNode>();
        unpack_node->Init(src_node, lbi, logical_blob_desc.shape(),
                          src_sbp_parallel.split_parallel().axis(),
                          dst_sbp_parallel.split_parallel().axis());

        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), unpack_node);
        Connect<TaskNode>(unpack_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return TRY(BuildSubTskGphBuilderStatus(
          sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
          dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
          "NcclCollectiveBoxingAll2AllSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
};

}  // namespace

CollectiveBoxingSubTskGphBuilder::CollectiveBoxingSubTskGphBuilder() {
  const CollectiveBoxingConf collective_boxing_conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new NcclCollectiveBoxingAllReduceSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingReduceScatterSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingAllGatherSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingReduceSubTskGphBuilder());
  builders.emplace_back(new CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingBroadcastSubTskGphBuilder());
  if (collective_boxing_conf.nccl_enable_all_to_all()) {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
    builders.emplace_back(new NcclCollectiveBoxingAll2AllSubTskGphBuilder());
#else
    LOG(WARNING) << "nccl_enable_all_to_all is unavailable unless NCCL_VERSION > 2.7.0";
#endif
  }
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> CollectiveBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!GlobalJobDesc().Bool("__is_user_function__")) { return Error::BoxingNotSupportedError(); }
  if (!IsSourceTimeShape(*sorted_src_comp_tasks.front()->logical_node()->out_blob_time_shape())) {
    return Error::BoxingNotSupportedError();
  }
  return chain_builder_->Build(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks, src_parallel_desc,
                               dst_parallel_desc, lbi, logical_blob_desc, src_sbp_parallel,
                               dst_sbp_parallel);
}

}  // namespace oneflow
