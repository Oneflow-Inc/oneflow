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

#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/fallback_to_cpu_slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/graph/nccl_send_recv_boxing_task_node.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

std::shared_ptr<ChainSubTskGphBuilder> Make1DSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  if (!Singleton<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  }
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new FallbackToCpuSliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  return std::make_shared<ChainSubTskGphBuilder>(builders);
}

void MergeParallelConf(const ParallelDesc& parallel_desc_0, const ParallelDesc& parallel_desc_1,
                       ParallelConf* parallel_conf) {
  CHECK_EQ(parallel_desc_0.device_tag(), parallel_desc_1.device_tag());
  std::set<std::pair<int64_t, int64_t>> machine_device_ids;
  for (int64_t machine_id : parallel_desc_0.sorted_machine_ids()) {
    for (int64_t device_id : parallel_desc_0.sorted_dev_phy_ids(machine_id)) {
      machine_device_ids.insert(std::make_pair(machine_id, device_id));
    }
  }
  for (int64_t machine_id : parallel_desc_1.sorted_machine_ids()) {
    for (int64_t device_id : parallel_desc_1.sorted_dev_phy_ids(machine_id)) {
      machine_device_ids.insert(std::make_pair(machine_id, device_id));
    }
  }
  parallel_conf->set_device_tag(parallel_desc_0.device_tag());
  for (const auto& pair : machine_device_ids) {
    parallel_conf->add_device_name("@" + std::to_string(pair.first) + ":"
                                   + std::to_string(pair.second));
  }
}

inline std::string NewUniqueIdGbc() {
  static std::atomic<int64_t> counter(0);
  static std::atomic<int64_t> curr_job_id(0);
  if (curr_job_id != GlobalJobDesc().job_id()) {
    curr_job_id = GlobalJobDesc().job_id();
    counter = 0;
  }
  return std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace

class FlatSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FlatSubTskGphBuilder);
  FlatSubTskGphBuilder() { sub_tsk_gph_builder_ = Make1DSubTskGphBuilder(); }
  ~FlatSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.hierarchy()->NumAxes() == 1
        && out_parallel_desc.hierarchy()->NumAxes() == 1) {
      return sub_tsk_gph_builder_->Build(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                                         in_parallel_desc, out_parallel_desc, lbi,
                                         logical_blob_desc, in_nd_sbp.sbp_parallel(0),
                                         out_nd_sbp.sbp_parallel(0), time_shape);
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

class NDNcclSendRecvBoxingSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NDNcclSendRecvBoxingSubTskGphBuilder);
  NDNcclSendRecvBoxingSubTskGphBuilder() {}
  ~NDNcclSendRecvBoxingSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.device_type() == DeviceType::kCUDA
        && out_parallel_desc.device_type() == DeviceType::kCUDA
        && !NdSbpHasPartialParallel(out_nd_sbp)) {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
      ParallelConf merged_parallel_conf;
      MergeParallelConf(in_parallel_desc.parallel_conf(), out_parallel_desc.parallel_conf(),
                        &merged_parallel_conf);
      ParallelDesc merged_parallel_desc(merged_parallel_conf);
      TaskNode* first_in_node = sorted_in_tasks.front();
      sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
      std::string stream_name = "NCCL_SEND_RECV_BOXING" + NewUniqueIdGbc();
      FOR_RANGE(int64_t, id, 0, merged_parallel_desc.parallel_num()) {
        NcclSendRecvBoxingTaskNode* node = ctx->task_graph()->NewNode<NcclSendRecvBoxingTaskNode>();
        const int64_t machine_id = JUST(merged_parallel_desc.MachineId4ParallelId(id));
        int64_t device_index = JUST(merged_parallel_desc.DeviceId4ParallelId(id));
        int64_t thrd_id = EncodeStreamIdToInt64(GenerateNamedTaskStreamId(
            machine_id, merged_parallel_desc.device_type(), device_index, stream_name));
        bool has_input = in_parallel_desc.Containing(machine_id, device_index);
        bool has_output = out_parallel_desc.Containing(machine_id, device_index);
        node->Init(machine_id, thrd_id, lbi, logical_blob_desc.shape(),
                   logical_blob_desc.data_type(), in_nd_sbp, out_nd_sbp, in_parallel_desc,
                   out_parallel_desc, id, merged_parallel_desc, has_input, has_output, stream_name);
        if (has_input) {
          int64_t in_id =
              JUST(in_parallel_desc.ParallelId4MachineDeviceId(machine_id, device_index));
          ctx->task_graph()->ConnectWithLbi(sorted_in_tasks.at(in_id), node, lbi);
        } else {
          // TODO: find nearest
          std::string regst_desc_name;
          first_in_node->BuildCtrlRegstDesc(node, &regst_desc_name);
          TaskEdge* edge = ctx->task_graph()->NewEdge();
          Connect<TaskNode>(first_in_node, edge, node);
          first_in_node->BindEdgeWithProducedRegst(edge, regst_desc_name);
        }
        if (has_output) { sorted_out_tasks->push_back(node); }
      }
      return BuildSubTskGphBuilderStatus("NDNcclSendRecvBoxingSubTskGphBuilder", "");
#else
      return Error::BoxingNotSupportedError() << "No CUDA or low NCCL version";
#endif
    } else {
      return Error::BoxingNotSupportedError()
             << "Partial SBP in the consumer or not running on CUDA";
    }
  }
};

class IntraGroupSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IntraGroupSubTskGphBuilder);
  IntraGroupSubTskGphBuilder() { sub_tsk_gph_builder_ = Make1DSubTskGphBuilder(); }
  ~IntraGroupSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy()
        && in_parallel_desc.hierarchy()->NumAxes() == 2
        && in_nd_sbp.sbp_parallel(0) == out_nd_sbp.sbp_parallel(0)
        && in_nd_sbp.sbp_parallel(1) != out_nd_sbp.sbp_parallel(1)) {
      const auto& hierarchy = in_parallel_desc.hierarchy();
      std::vector<SubTskGphBuilderStatus> status;
      const int64_t num_groups = hierarchy->At(0);
      const int64_t group_size = hierarchy->At(1);
      status.reserve(num_groups);
      sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
      sorted_out_tasks->resize(out_parallel_desc.parallel_num());
      FOR_RANGE(int64_t, i, 0, num_groups) {
        std::vector<TaskNode*> in_tasks;
        std::vector<TaskNode*> out_tasks;
        std::vector<std::vector<TaskNode*>> ctrl_tasks;
        ParallelConf in_parallel_conf;
        in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
        in_parallel_conf.mutable_hierarchy()->add_dim(group_size);
        ParallelConf out_parallel_conf;
        out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
        out_parallel_conf.mutable_hierarchy()->add_dim(group_size);
        FOR_RANGE(int64_t, j, 0, group_size) {
          const int64_t parallel_id = i * group_size + j;
          in_tasks.emplace_back(sorted_in_tasks.at(parallel_id));
          in_parallel_conf.add_device_name(
              "@" + std::to_string(JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
          out_parallel_conf.add_device_name(
              "@" + std::to_string(JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
        }
        DimVector dim_vec = logical_blob_desc.shape().dim_vec();
        if (in_nd_sbp.sbp_parallel(0).has_split_parallel()) {
          const int64_t axis = in_nd_sbp.sbp_parallel(0).split_parallel().axis();
          dim_vec.at(axis) /= hierarchy->At(0);
        }
        BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
        std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
            JUST(sub_tsk_gph_builder_->Build(
                ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
                ParallelDesc(out_parallel_conf), lbi, new_blob_desc, in_nd_sbp.sbp_parallel(1),
                out_nd_sbp.sbp_parallel(1), time_shape));
        status.emplace_back(*boxing_builder_status);
        CHECK_EQ_OR_RETURN(out_tasks.size(), group_size);
        FOR_RANGE(int64_t, j, 0, group_size) {
          const int64_t parallel_id = i * group_size + j;
          sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
          if (!ctrl_tasks.empty()) {
            for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
              sorted_ctrl_tasks->at(parallel_id).emplace_back(ctrl_node);
            }
          }
        }
      }
      return MakeComposedSubTskGphBuilderStatus(status);
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

class InterGroupSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InterGroupSubTskGphBuilder);
  InterGroupSubTskGphBuilder() { sub_tsk_gph_builder_ = Make1DSubTskGphBuilder(); }
  ~InterGroupSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy()
        && in_parallel_desc.hierarchy()->NumAxes() == 2
        && in_nd_sbp.sbp_parallel(1) == out_nd_sbp.sbp_parallel(1)
        && in_nd_sbp.sbp_parallel(0) != out_nd_sbp.sbp_parallel(0)
        && !NdSbpAllSameSplitParallel(in_nd_sbp) && !NdSbpAllSameSplitParallel(out_nd_sbp)) {
      const auto& hierarchy = in_parallel_desc.hierarchy();
      std::vector<SubTskGphBuilderStatus> status;
      const int64_t num_groups = hierarchy->At(0);
      const int64_t group_size = hierarchy->At(1);
      status.reserve(group_size);
      sorted_ctrl_tasks->resize(out_parallel_desc.parallel_num());
      sorted_out_tasks->resize(out_parallel_desc.parallel_num());
      FOR_RANGE(int64_t, i, 0, group_size) {
        std::vector<TaskNode*> in_tasks;
        std::vector<TaskNode*> out_tasks;
        std::vector<std::vector<TaskNode*>> ctrl_tasks;
        ParallelConf in_parallel_conf;
        in_parallel_conf.set_device_tag(in_parallel_desc.device_tag());
        in_parallel_conf.mutable_hierarchy()->add_dim(num_groups);
        ParallelConf out_parallel_conf;
        out_parallel_conf.set_device_tag(out_parallel_desc.device_tag());
        out_parallel_conf.mutable_hierarchy()->add_dim(num_groups);
        FOR_RANGE(int64_t, j, 0, num_groups) {
          const int64_t parallel_id = j * group_size + i;
          in_tasks.emplace_back(sorted_in_tasks.at(parallel_id));
          in_parallel_conf.add_device_name(
              "@" + std::to_string(JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
          out_parallel_conf.add_device_name(
              "@" + std::to_string(JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
        }
        DimVector dim_vec = logical_blob_desc.shape().dim_vec();
        if (in_nd_sbp.sbp_parallel(1).has_split_parallel()) {
          const int64_t axis = in_nd_sbp.sbp_parallel(1).split_parallel().axis();
          dim_vec.at(axis) /= hierarchy->At(1);
        }
        BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
        std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
            JUST(sub_tsk_gph_builder_->Build(
                ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
                ParallelDesc(out_parallel_conf), lbi, new_blob_desc, in_nd_sbp.sbp_parallel(0),
                out_nd_sbp.sbp_parallel(0), time_shape));
        status.emplace_back(*boxing_builder_status);
        CHECK_EQ_OR_RETURN(out_tasks.size(), num_groups);
        FOR_RANGE(int64_t, j, 0, num_groups) {
          const int64_t parallel_id = j * group_size + i;
          sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
          if (!ctrl_tasks.empty()) {
            for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
              sorted_ctrl_tasks->at(parallel_id).emplace_back(ctrl_node);
            }
          }
        }
      }
      return MakeComposedSubTskGphBuilderStatus(status);
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::shared_ptr<SubTskGphBuilder> sub_tsk_gph_builder_;
};

class Dim0NdSbpMismatchedSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dim0NdSbpMismatchedSubTskGphBuilder);
  Dim0NdSbpMismatchedSubTskGphBuilder() {
    inter_group_sub_tsk_gph_builder_.reset(new InterGroupSubTskGphBuilder());
  }
  ~Dim0NdSbpMismatchedSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.hierarchy()->NumAxes() == 2
        && (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy())
        && in_nd_sbp.sbp_parallel(0) != out_nd_sbp.sbp_parallel(0)
        && in_nd_sbp.sbp_parallel(1) == out_nd_sbp.sbp_parallel(1)
        && !(NdSbpAllSameSplitParallel(in_nd_sbp) || NdSbpAllSameSplitParallel(out_nd_sbp))) {
      return inter_group_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
    } else {
      return nd_nccl_send_recv_boxing_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
    }
  }

 private:
  std::unique_ptr<InterGroupSubTskGphBuilder> inter_group_sub_tsk_gph_builder_;
  std::unique_ptr<NDNcclSendRecvBoxingSubTskGphBuilder>
      nd_nccl_send_recv_boxing_sub_tsk_gph_builder_;
};

class Same2DHierarchySubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Same2DHierarchySubTskGphBuilder);
  Same2DHierarchySubTskGphBuilder() {
    intra_group_sub_tsk_gph_builder_.reset(new IntraGroupSubTskGphBuilder());
    dim0_nd_sbp_mismatched_sub_tsk_gph_builder_.reset(new Dim0NdSbpMismatchedSubTskGphBuilder());
  }
  ~Same2DHierarchySubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.hierarchy()->NumAxes() == 2
        && (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy())) {
      if (in_nd_sbp.sbp_parallel(0) == out_nd_sbp.sbp_parallel(0)) {
        return intra_group_sub_tsk_gph_builder_->Build(
            ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
            out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
      } else {
        return dim0_nd_sbp_mismatched_sub_tsk_gph_builder_->Build(
            ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
            out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
      }
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::unique_ptr<IntraGroupSubTskGphBuilder> intra_group_sub_tsk_gph_builder_;
  std::unique_ptr<Dim0NdSbpMismatchedSubTskGphBuilder> dim0_nd_sbp_mismatched_sub_tsk_gph_builder_;
};

struct DispatchHierarchicalSubTskGphBuilder::Impl {
  Impl();
  std::unique_ptr<FlatSubTskGphBuilder> flat_sub_tsk_gph_builder_;
  std::unique_ptr<Same2DHierarchySubTskGphBuilder> same_2d_hierarchy_sub_tsk_gph_builder_;
  std::unique_ptr<NDNcclSendRecvBoxingSubTskGphBuilder>
      nd_nccl_send_recv_boxing_sub_tsk_gph_builder_;
};

DispatchHierarchicalSubTskGphBuilder::Impl::Impl() {
  flat_sub_tsk_gph_builder_.reset(new FlatSubTskGphBuilder());
  same_2d_hierarchy_sub_tsk_gph_builder_.reset(new Same2DHierarchySubTskGphBuilder());
  nd_nccl_send_recv_boxing_sub_tsk_gph_builder_.reset(new NDNcclSendRecvBoxingSubTskGphBuilder());
}

DispatchHierarchicalSubTskGphBuilder::DispatchHierarchicalSubTskGphBuilder() {
  impl_.reset(new Impl());
}

DispatchHierarchicalSubTskGphBuilder::~DispatchHierarchicalSubTskGphBuilder() = default;

Maybe<SubTskGphBuilderStatus> DispatchHierarchicalSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
    const Shape& time_shape) const {
  ParallelDesc reduced_in_parallel_desc = in_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = out_parallel_desc;
  NdSbp reduced_in_nd_sbp;
  NdSbp reduced_out_nd_sbp;
  // The 1d to 2d and 2d to 1d cases are consider in this function
  // If it gives out 1d sbp and 2d sbp simultaneously, then that the 2d sbp can not be converted
  // to 1d sbp and 1d sbp can not be expanded to 2d sbp.
  InOutParallelDimReduce(in_parallel_desc, out_parallel_desc, in_nd_sbp, out_nd_sbp,
                         &reduced_in_parallel_desc, &reduced_out_parallel_desc, &reduced_in_nd_sbp,
                         &reduced_out_nd_sbp, logical_blob_desc.shape());
  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
  if ((in_hierarchy->NumAxes() > 2 || out_hierarchy->NumAxes() > 2)
      && reduced_in_parallel_desc.device_type() == DeviceType::kCUDA
      && reduced_out_parallel_desc.device_type() == DeviceType::kCUDA) {
    return impl_->nd_nccl_send_recv_boxing_sub_tsk_gph_builder_->Build(
        ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
        reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_nd_sbp, reduced_out_nd_sbp,
        time_shape);
  }
  if (in_hierarchy->NumAxes() <= 2 && out_hierarchy->NumAxes() <= 2) {
    if (in_hierarchy->NumAxes() == 1 && out_hierarchy->NumAxes() == 1) {
      return impl_->flat_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_nd_sbp, reduced_out_nd_sbp,
          time_shape);
    } else if ((in_hierarchy->NumAxes() == 2) && (*in_hierarchy == *out_hierarchy)) {
      return impl_->same_2d_hierarchy_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_nd_sbp, reduced_out_nd_sbp,
          time_shape);
    } else if (reduced_in_parallel_desc.device_type() == DeviceType::kCUDA
               && reduced_out_parallel_desc.device_type() == DeviceType::kCUDA) {
      return impl_->nd_nccl_send_recv_boxing_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_nd_sbp, reduced_out_nd_sbp,
          time_shape);
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
  return Error::BoxingNotSupportedError();
}

}  // namespace oneflow
