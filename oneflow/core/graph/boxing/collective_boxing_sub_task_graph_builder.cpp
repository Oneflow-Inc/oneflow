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

  const int64_t machine_id = parallel_desc.MachineIdForParallelId(parallel_id);
  const int64_t device_id = parallel_desc.DeviceIdForParallelId(parallel_id);
  const int64_t thrd_id = Global<IDMgr>::Get()->GetGpuNcclThrdId(device_id);
  node->Init(machine_id, thrd_id, NewAreaId(), op_conf);
}

int64_t FindRootParallelId(const ParallelDesc& multi_device, const ParallelDesc& sole_device) {
  CHECK_EQ(sole_device.parallel_num(), 1);
  const int64_t root_machine_id = sole_device.MachineIdForParallelId(0);
  const int64_t root_device_id = sole_device.DeviceIdForParallelId(0);
  int64_t root_parallel_id = -1;
  FOR_RANGE(int64_t, i, 0, multi_device.parallel_num()) {
    if (multi_device.MachineIdForParallelId(i) == root_machine_id
        && multi_device.DeviceIdForParallelId(i) == root_device_id) {
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
      return Error::BoxingNotSupported();
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
      return Error::BoxingNotSupported();
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
          "NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupported();
    }
  }
};

void SplitAxisChange(const int32_t axis_pre, const int32_t axis_after,
                     const std::vector<TensorSliceView>& pre_slices,
                     std::vector<TensorSliceView>& slices, TensorSliceView concat_slice) {
  // 2->0  (0,4)(0,16)(0,8)  (0,4)(0,16)(8,16) -> (0,4)(0,16)(0,8) (4,8)(0,16)(0,8)
  // 0->2  (0,4)(0,16)(0,8) (4,8)(0,16)(0,8) -> (0,4)(0,16)(0,8) (0,4)(0,16)(8,16)
  TensorSliceView slice_0 = pre_slices.at(0);
  std::vector<Range> ranges = slice_0.range_vec();
  FOR_RANGE(int64_t, i, 0, pre_slices.size()) {
    ranges[axis_pre].mut_begin() = slice_0.range_vec().at(axis_pre).begin();
    ranges[axis_pre].mut_end() = slice_0.range_vec().at(axis_pre).end();
    ranges[axis_after].mut_begin() =
        slice_0.range_vec().at(axis_after).begin() + i * slice_0.shape().At(i);
    ranges[axis_after].mut_end() =
        slice_0.range_vec().at(axis_after).end() + i * slice_0.shape().At(i);
    slices.push_back(TensorSliceView(ranges));
  }
  ranges[axis_pre].mut_begin() = slice_0.range_vec().at(axis_pre).begin();
  ranges[axis_pre].mut_end() = slice_0.range_vec().at(axis_pre).end();
  ranges[axis_after].mut_begin() = slice_0.range_vec().at(axis_after).begin();
  ranges[axis_after].mut_end() = slice_0.range_vec().at(axis_after).begin()
                                 + pre_slices.size() * slice_0.shape().At(axis_after);
  concat_slice = TensorSliceView(ranges);
}

class NcclCollectiveBoxingAll2AllSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAll2AllSubTskGphBuilder);
  NcclCollectiveBoxingAll2AllSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingAll2AllSubTskGphBuilder() override = default;

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
                    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                    const SbpParallel& src_sbp_parallel,
                    const SbpParallel& dst_sbp_parallel) const override {
    const auto GetBoxingGpuThrdId = [](const int64_t dev_id, CudaWorkType work_type) -> int64_t {
#ifdef WITH_CUDA
      if (work_type == CudaWorkType::kCopyH2D) {
        return Global<IDMgr>::Get()->GetGpuH2DThrdId(dev_id);
      } else if (work_type == CudaWorkType::kCopyD2H) {
        return Global<IDMgr>::Get()->GetGpuD2HThrdId(dev_id);
      } else {
        return Global<IDMgr>::Get()->GetGpuMixThrdId(dev_id);
      }
#else
      UNIMPLEMENTED();
#endif
    };

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
#ifdef WITH_CUDA
        thrd_id = GetBoxingGpuThrdId(pd.DeviceIdForParallelId(parallel_id), CudaWorkType::kCopyH2D);
#else
        UNIMPLEMENTED();
#endif
      } else {
        UNIMPLEMENTED();
      }
      node->Init(lbi, slice, mode, machine_id, thrd_id);
      return node;
    };

    if (dst_parallel_desc.EqualsIgnoringDeviceType(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(src_parallel_desc)
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0
        && SubTskGphBuilderUtil::IsBoxingS2S(src_sbp_parallel, dst_sbp_parallel)) {
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAll2All-" + NewUniqueId();
      const std::vector<TensorSliceView> in_slices = SubTskGphBuilderUtil::GetTensorSliceView(
          src_parallel_desc.parallel_num(), src_sbp_parallel, logical_blob_desc);
      const std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensorSliceView(
          dst_parallel_desc.parallel_num(), dst_sbp_parallel, logical_blob_desc);

      const int32_t parallel_num = src_parallel_desc.parallel_num();

      std::vector<std::vector<TensorSliceView>> intersections;
      FOR_RANGE(int64_t, in_id, 0, src_parallel_desc.parallel_num()) {
        const TensorSliceView& in_slice = in_slices.at(in_id);
        std::vector<TensorSliceView> intersection;
        FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
          intersection.push_back(in_slice.Intersect(out_slices.at(out_id)));
        }
        intersections.push_back(intersection);
      }

      std::vector<std::vector<TensorSliceView>> intersections2;
      FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
        const TensorSliceView& out_slice = out_slices.at(out_id);
        std::vector<TensorSliceView> intersection;
        FOR_RANGE(int64_t, in_id, 0, src_parallel_desc.parallel_num()) {
          intersection.push_back(out_slice.Intersect(in_slices.at(in_id)));
        }
        intersections2.push_back(intersection);
      }

      std::vector<TaskNode*> in_nodes;
      in_nodes.assign(sorted_src_comp_tasks.begin(), sorted_src_comp_tasks.end());

      const int32_t src_split_axis = src_sbp_parallel.split_parallel().axis();

      FOR_RANGE(int64_t, i, 0, parallel_num) {  // for gpus

        TaskNode* in_node = in_nodes.at(i);
        SliceBoxingTaskNode* intersection_copy_to_s0;
        TensorSliceView collective_in_slice;
        // src not contiguous s0
        if (src_split_axis != logical_blob_desc.shape().NumAxes() - 1) {
          std::vector<TensorSliceView> intersection_slices;
          TensorSliceView concat_slice;
          SplitAxisChange(logical_blob_desc.shape().NumAxes() - 1, 0, intersections[i],
                          intersection_slices,
                          concat_slice);  // intersections split2, intersection_slices split0
          intersection_copy_to_s0 =
              CreateBoxingNode121(src_parallel_desc, i, concat_slice, kSliceBoxingTaskModeCopy);
          FOR_RANGE(int64_t, j, 0, parallel_num) {  // for each gpus slices
            // split to intersections
            SliceBoxingTaskNode* copy_to_intersection = CreateBoxingNode121(
                src_parallel_desc, i, intersections[i][j], kSliceBoxingTaskModeCopy);
            copy_to_intersection->ConnectToSrcNodeWithSlice(in_node, ctx->task_graph()->NewEdge(),
                                                            in_slices.at(i));
            // intersections concat to split 0
            intersection_copy_to_s0->ConnectToSrcNodeWithSlice(
                copy_to_intersection, ctx->task_graph()->NewEdge(), intersection_slices[j]);
          }
          collective_in_slice = concat_slice;
        } else {
          // useless copy, to rm
          intersection_copy_to_s0 =
              CreateBoxingNode121(src_parallel_desc, i, in_slices.at(i), kSliceBoxingTaskModeCopy);
          intersection_copy_to_s0->ConnectToSrcNodeWithSlice(in_node, ctx->task_graph()->NewEdge(),
                                                             in_slices.at(i));
          collective_in_slice = in_slices.at(i);
        }

        // nccl node
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        SliceBoxingTaskNode* collective_out = CreateBoxingNode121(
            dst_parallel_desc, i, collective_in_slice, kSliceBoxingTaskModeCopy);
        NcclInitCollectiveNode(collective_node, src_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAll2All, -1);
        Connect<TaskNode>(intersection_copy_to_s0, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), collective_out);

        std::vector<TaskNode*> out_nodes;
        const int32_t dst_split_axis = dst_sbp_parallel.split_parallel().axis();
        SliceBoxingTaskNode* out_node =
            CreateBoxingNode121(dst_parallel_desc, i, out_slices.at(i), kSliceBoxingTaskModeCopy);
        if (dst_split_axis != logical_blob_desc.shape().NumAxes() - 1) {  // dst not contiguous s0
          std::vector<TensorSliceView> intersection_slices;
          TensorSliceView concat_slice;
          SplitAxisChange(logical_blob_desc.shape().NumAxes() - 1, 0, intersections2[i],
                          intersection_slices,
                          concat_slice);  // intersections split2, intersection_slices split0
          FOR_RANGE(int64_t, j, 0, parallel_num) {  // for each gpus slices
            // split to intersections
            SliceBoxingTaskNode* copy_to_intersection = CreateBoxingNode121(
                dst_parallel_desc, i, intersection_slices[j], kSliceBoxingTaskModeCopy);
            copy_to_intersection->ConnectToSrcNodeWithSlice(
                collective_out, ctx->task_graph()->NewEdge(), concat_slice);
            // intersections concat to split 0
            out_node->ConnectToSrcNodeWithSlice(copy_to_intersection, ctx->task_graph()->NewEdge(),
                                                intersections2[i][j]);
          }
        } else {
          // useless copy, to rm
          out_node->ConnectToSrcNodeWithSlice(collective_out, ctx->task_graph()->NewEdge(),
                                              out_slices.at(i));
        }
        out_nodes.push_back(out_node);
        ctx->ConnectAll121(out_nodes, sorted_dst_comp_tasks);
      }
      return Maybe<void>::Ok();
    } else {
      return Error::BoxingNotSupported();
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
      if (root_parallel_id == -1) { return Error::BoxingNotSupported(); }

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
          "NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupported();
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
        const auto src_machine_id = src_parallel_desc.MachineIdForParallelId(0);
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
          "NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupported();
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
        return Error::BoxingNotSupported();
      }
      if (root_parallel_id == -1) { return Error::BoxingNotSupported(); }

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
          "NcclCollectiveBoxingReduceScatterSubTskGphBuilder", ""));
    } else {
      return Error::BoxingNotSupported();
    }
  }
};
}  // namespace

CollectiveBoxingSubTskGphBuilder::CollectiveBoxingSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new NcclCollectiveBoxingAllReduceSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingReduceScatterSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingAllGatherSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingAll2AllSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingReduceSubTskGphBuilder());
  builders.emplace_back(new CollectiveBoxingScatterThenNcclAllGatherSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingBroadcastSubTskGphBuilder());
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> CollectiveBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!GlobalJobDesc().Bool("__is_user_function__")) { return Error::BoxingNotSupported(); }
  if (!IsSourceTimeShape(*sorted_src_comp_tasks.front()->logical_node()->out_blob_time_shape())) {
    return Error::BoxingNotSupported();
  }
  return chain_builder_->Build(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks, src_parallel_desc,
                               dst_parallel_desc, lbi, logical_blob_desc, src_sbp_parallel,
                               dst_sbp_parallel);
}

}  // namespace oneflow
