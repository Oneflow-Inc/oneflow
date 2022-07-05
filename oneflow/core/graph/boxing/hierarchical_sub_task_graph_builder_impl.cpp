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
        && in_nd_sbp.sbp_parallel(1) == out_nd_sbp.sbp_parallel(1)) {
      if (!(NdSbpAllSameSplitParallel(in_nd_sbp) || NdSbpAllSameSplitParallel(out_nd_sbp))) {
        return inter_group_sub_tsk_gph_builder_->Build(
            ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
            out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
      } else {
        return Error::BoxingNotSupportedError();
      }
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::unique_ptr<InterGroupSubTskGphBuilder> inter_group_sub_tsk_gph_builder_;
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
      } else if (in_nd_sbp.sbp_parallel(1) == out_nd_sbp.sbp_parallel(1)) {
        return dim0_nd_sbp_mismatched_sub_tsk_gph_builder_->Build(
            ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
            out_parallel_desc, lbi, logical_blob_desc, in_nd_sbp, out_nd_sbp, time_shape);
      } else {
        return Error::BoxingNotSupportedError();
      }
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::unique_ptr<IntraGroupSubTskGphBuilder> intra_group_sub_tsk_gph_builder_;
  std::unique_ptr<Dim0NdSbpMismatchedSubTskGphBuilder> dim0_nd_sbp_mismatched_sub_tsk_gph_builder_;
};

class ExpandToSame2DHierarchySubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpandToSame2DHierarchySubTskGphBuilder);
  ExpandToSame2DHierarchySubTskGphBuilder() {
    same_2d_hierarchy_sub_tsk_gph_builder_.reset(new Same2DHierarchySubTskGphBuilder());
  }
  ~ExpandToSame2DHierarchySubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.hierarchy()->elem_cnt() == out_parallel_desc.hierarchy()->elem_cnt()
        && in_parallel_desc.hierarchy()->NumAxes() == 1
        && out_parallel_desc.hierarchy()->NumAxes() == 2) {
      ParallelConf intermediate_parallel_conf = in_parallel_desc.parallel_conf();
      out_parallel_desc.hierarchy()->ToProto(intermediate_parallel_conf.mutable_hierarchy());
      NdSbp intermediate_nd_sbp;
      *intermediate_nd_sbp.add_sbp_parallel() = in_nd_sbp.sbp_parallel(0);
      *intermediate_nd_sbp.add_sbp_parallel() = in_nd_sbp.sbp_parallel(0);
      return same_2d_hierarchy_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
          ParallelDesc(intermediate_parallel_conf), out_parallel_desc, lbi, logical_blob_desc,
          intermediate_nd_sbp, out_nd_sbp, time_shape);
    } else if (in_parallel_desc.hierarchy()->elem_cnt() == out_parallel_desc.hierarchy()->elem_cnt()
               && in_parallel_desc.hierarchy()->NumAxes() == 2
               && out_parallel_desc.hierarchy()->NumAxes() == 1) {
      ParallelConf intermediate_parallel_conf = out_parallel_desc.parallel_conf();
      in_parallel_desc.hierarchy()->ToProto(intermediate_parallel_conf.mutable_hierarchy());
      NdSbp intermediate_nd_sbp;
      *intermediate_nd_sbp.add_sbp_parallel() = out_nd_sbp.sbp_parallel(0);
      *intermediate_nd_sbp.add_sbp_parallel() = out_nd_sbp.sbp_parallel(0);
      return same_2d_hierarchy_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
          ParallelDesc(intermediate_parallel_conf), lbi, logical_blob_desc, in_nd_sbp,
          intermediate_nd_sbp, time_shape);
    } else {
      return Error::BoxingNotSupportedError();
    }
  }

 private:
  std::unique_ptr<Same2DHierarchySubTskGphBuilder> same_2d_hierarchy_sub_tsk_gph_builder_;
};

struct DispatchHierarchicalSubTskGphBuilder::Impl {
  Impl();
  std::unique_ptr<FlatSubTskGphBuilder> flat_sub_tsk_gph_builder_;
  std::unique_ptr<Same2DHierarchySubTskGphBuilder> same_2d_hierarchy_sub_tsk_gph_builder_;
  std::unique_ptr<ExpandToSame2DHierarchySubTskGphBuilder>
      expand_to_same_2d_hierarchy_sub_tsk_gph_builder_;
};

DispatchHierarchicalSubTskGphBuilder::Impl::Impl() {
  flat_sub_tsk_gph_builder_.reset(new FlatSubTskGphBuilder());
  same_2d_hierarchy_sub_tsk_gph_builder_.reset(new Same2DHierarchySubTskGphBuilder());
  expand_to_same_2d_hierarchy_sub_tsk_gph_builder_.reset(
      new ExpandToSame2DHierarchySubTskGphBuilder());
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
  InOutParallelDimReduce(in_parallel_desc, out_parallel_desc, in_nd_sbp, out_nd_sbp,
                         &reduced_in_parallel_desc, &reduced_out_parallel_desc, &reduced_in_nd_sbp,
                         &reduced_out_nd_sbp);
  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
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
    } else if (in_hierarchy->elem_cnt() == out_hierarchy->elem_cnt()
               && ((in_hierarchy->NumAxes() == 1 && out_hierarchy->NumAxes() == 2)
                   || (in_hierarchy->NumAxes() == 2 && out_hierarchy->NumAxes() == 1))) {
      return impl_->expand_to_same_2d_hierarchy_sub_tsk_gph_builder_->Build(
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
