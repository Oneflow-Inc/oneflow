#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/collective_boxing_task_node.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"

namespace oneflow {

using namespace boxing::collective;

namespace {

void NcclInitCollectiveNode(CollectiveBoxingGenericTaskNode* node,
                            const ParallelDesc& parallel_desc, int64_t parallel_id,
                            const std::string& name, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, OpType op_type, int64_t root) {
  OperatorConf op_conf;
  op_conf.set_name(name);
  op_conf.set_device_type(DeviceType::kGPU);
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

int64_t FindRootParallelId(const ParallelDesc& multi_device, const int64_t root_machine_id,
                           const int64_t root_device_id) {
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

int64_t FindRootParallelId(const ParallelDesc& multi_device, const ParallelDesc& sole_device) {
  CHECK_EQ(sole_device.parallel_num(), 1);
  const int64_t root_machine_id = sole_device.MachineIdForParallelId(0);
  const int64_t root_device_id = sole_device.DeviceIdForParallelId(0);
  return FindRootParallelId(multi_device, root_machine_id, root_device_id);
}

bool IsSourceTimeShape(const Shape& shape) {
  return shape.elem_cnt() == GlobalJobDesc().TotalBatchNum() * GlobalJobDesc().NumOfPiecesInBatch();
}

class NcclCollectiveBoxingAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingAllReduceSubTskGphBuilder);
  NcclCollectiveBoxingAllReduceSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingAllReduceSubTskGphBuilder() override = default;

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
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
      return Maybe<void>::Ok();
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

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
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
      return Maybe<void>::Ok();
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

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
                    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                    const SbpParallel& src_sbp_parallel,
                    const SbpParallel& dst_sbp_parallel) const override {
    if (dst_parallel_desc.Equals(src_parallel_desc)
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && dst_parallel_desc.parallel_num() > 1
        && SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)
        && src_sbp_parallel.split_parallel().axis() == 0) {
      const std::string op_name = "System-Boxing-NcclCollectiveBoxingAllGather-" + NewUniqueId();
      FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
        CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
        CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
        auto* collective_node = ctx->task_graph()->NewNode<CollectiveBoxingGenericTaskNode>();
        NcclInitCollectiveNode(collective_node, src_parallel_desc, i, op_name, lbi,
                               logical_blob_desc, OpType::kOpTypeAllGather, -1);
        Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), collective_node);
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
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

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
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
        if (i == root_parallel_id) {
          CompTaskNode* dst_node = sorted_dst_comp_tasks.front();
          Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
        }
      }
      return Maybe<void>::Ok();
    } else {
      return Error::BoxingNotSupported();
    }
  }
};

class NcclCollectiveBoxingBroadcastSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclCollectiveBoxingBroadcastSubTskGphBuilder);
  NcclCollectiveBoxingBroadcastSubTskGphBuilder() = default;
  ~NcclCollectiveBoxingBroadcastSubTskGphBuilder() override = default;

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
                    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                    const SbpParallel& src_sbp_parallel,
                    const SbpParallel& dst_sbp_parallel) const override {
    if (src_parallel_desc.parallel_num() == 1 && dst_parallel_desc.parallel_num() > 1
        && dst_parallel_desc.device_type() == DeviceType::kGPU
        && !SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)
        && dst_sbp_parallel.has_broadcast_parallel()) {
      TaskNode* gpu_src_node;
      int64_t root_parallel_id;
      if (src_parallel_desc.device_type() == DeviceType::kCPU) {
        auto* cpu_src_node = sorted_src_comp_tasks.front();
        const int64_t gpu_id_copied_to = sorted_dst_comp_tasks.front()->GpuPhyId();
        CopyHdTaskNode* copy_task = ctx->task_graph()->NewNode<CopyHdTaskNode>();
        copy_task->Init(CopyHdOpConf::H2D, cpu_src_node->machine_id(), gpu_id_copied_to);
        Connect<TaskNode>(cpu_src_node, ctx->task_graph()->NewEdge(), copy_task);
        gpu_src_node = copy_task;
        root_parallel_id =
            FindRootParallelId(dst_parallel_desc, cpu_src_node->machine_id(), gpu_id_copied_to);
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
        }
        Connect<TaskNode>(collective_node, ctx->task_graph()->NewEdge(), dst_node);
      }
      return Maybe<void>::Ok();
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
  builders.emplace_back(new NcclCollectiveBoxingReduceSubTskGphBuilder());
  builders.emplace_back(new NcclCollectiveBoxingBroadcastSubTskGphBuilder());
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<void> CollectiveBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!IsSourceTimeShape(*sorted_src_comp_tasks.front()->logical_node()->out_blob_time_shape())) {
    return Error::BoxingNotSupported();
  }
  return chain_builder_->Build(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks, src_parallel_desc,
                               dst_parallel_desc, lbi, logical_blob_desc, src_sbp_parallel,
                               dst_sbp_parallel);
}

}  // namespace oneflow
