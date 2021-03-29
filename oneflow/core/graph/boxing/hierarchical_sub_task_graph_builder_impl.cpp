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
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

namespace {

void ParallelDimReduce(const ParallelDesc& parallel_desc,
                       const ParallelDistribution& parallel_distribution,
                       ParallelDesc* reduced_parallel_desc,
                       ParallelDistribution* reduced_parallel_distribution) {
  const auto& hierarchy = parallel_desc.hierarchy();
  DimVector reduced_hierarchy;
  reduced_hierarchy.push_back(hierarchy->At(0));
  *reduced_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(0);
  FOR_RANGE(int64_t, i, 1, hierarchy->NumAxes()) {
    if (parallel_distribution.sbp_parallel(i) == parallel_distribution.sbp_parallel(i - 1)) {
      reduced_hierarchy.back() *= hierarchy->At(i);
    } else {
      reduced_hierarchy.push_back(hierarchy->At(i));
      *reduced_parallel_distribution->add_sbp_parallel() = parallel_distribution.sbp_parallel(i);
    }
  }
  ParallelConf reduced_parallel_conf = parallel_desc.parallel_conf();
  Shape(reduced_hierarchy).ToProto(reduced_parallel_conf.mutable_hierarchy());
  *reduced_parallel_desc = ParallelDesc(reduced_parallel_conf);
}

void CollaborativeParallelDimReduce(const ParallelDesc& in_parallel_desc,
                                    const ParallelDesc& out_parallel_desc,
                                    const ParallelDistribution& in_parallel_distribution,
                                    const ParallelDistribution& out_parallel_distribution,
                                    ParallelDesc* reduced_in_parallel_desc,
                                    ParallelDesc* reduced_out_parallel_desc,
                                    ParallelDistribution* reduced_in_parallel_distribution,
                                    ParallelDistribution* reduced_out_parallel_distribution) {
  const auto& in_hierarchy = in_parallel_desc.hierarchy();
  const auto& out_hierarchy = out_parallel_desc.hierarchy();
  CHECK_EQ(in_hierarchy->NumAxes(), out_hierarchy->NumAxes());

  DimVector reduced_in_hierarchy;
  reduced_in_hierarchy.push_back(in_hierarchy->At(0));
  *reduced_in_parallel_distribution->add_sbp_parallel() = in_parallel_distribution.sbp_parallel(0);

  DimVector reduced_out_hierarchy;
  reduced_out_hierarchy.push_back(out_hierarchy->At(0));
  *reduced_out_parallel_distribution->add_sbp_parallel() =
      out_parallel_distribution.sbp_parallel(0);

  FOR_RANGE(int64_t, i, 1, in_hierarchy->NumAxes()) {
    if ((in_parallel_distribution.sbp_parallel(i) == in_parallel_distribution.sbp_parallel(i - 1))
        && (out_parallel_distribution.sbp_parallel(i)
            == out_parallel_distribution.sbp_parallel(i - 1))) {
      reduced_in_hierarchy.back() *= in_hierarchy->At(i);
      reduced_out_hierarchy.back() *= out_hierarchy->At(i);
    } else {
      reduced_in_hierarchy.push_back(in_hierarchy->At(i));
      *reduced_in_parallel_distribution->add_sbp_parallel() =
          in_parallel_distribution.sbp_parallel(i);

      reduced_out_hierarchy.push_back(out_hierarchy->At(i));
      *reduced_out_parallel_distribution->add_sbp_parallel() =
          out_parallel_distribution.sbp_parallel(i);
    }
  }

  ParallelConf reduced_in_parallel_conf = in_parallel_desc.parallel_conf();
  Shape(reduced_in_hierarchy).ToProto(reduced_in_parallel_conf.mutable_hierarchy());
  *reduced_in_parallel_desc = ParallelDesc(reduced_in_parallel_conf);

  ParallelConf reduced_out_parallel_conf = out_parallel_desc.parallel_conf();
  Shape(reduced_out_hierarchy).ToProto(reduced_out_parallel_conf.mutable_hierarchy());
  *reduced_out_parallel_desc = ParallelDesc(reduced_out_parallel_conf);
}

void InOutParallelDimReduce(const ParallelDesc& in_parallel_desc,
                            const ParallelDesc& out_parallel_desc,
                            const ParallelDistribution& in_parallel_distribution,
                            const ParallelDistribution& out_parallel_distribution,
                            ParallelDesc* reduced_in_parallel_desc,
                            ParallelDesc* reduced_out_parallel_desc,
                            ParallelDistribution* reduced_in_parallel_distribution,
                            ParallelDistribution* reduced_out_parallel_distribution) {
  const int64_t in_hierarchy_axes = in_parallel_desc.hierarchy()->NumAxes();
  const int64_t out_hierarchy_axes = out_parallel_desc.hierarchy()->NumAxes();
  if (in_hierarchy_axes == 1 && out_hierarchy_axes == 1) {
    *reduced_in_parallel_desc = in_parallel_desc;
    *reduced_out_parallel_desc = out_parallel_desc;
    *reduced_in_parallel_distribution = in_parallel_distribution;
    *reduced_out_parallel_distribution = out_parallel_distribution;
  } else if (in_hierarchy_axes != out_hierarchy_axes) {
    ParallelDimReduce(in_parallel_desc, in_parallel_distribution, reduced_in_parallel_desc,
                      reduced_in_parallel_distribution);
    ParallelDimReduce(out_parallel_desc, out_parallel_distribution, reduced_out_parallel_desc,
                      reduced_out_parallel_distribution);
  } else {
    CollaborativeParallelDimReduce(in_parallel_desc, out_parallel_desc, in_parallel_distribution,
                                   out_parallel_distribution, reduced_in_parallel_desc,
                                   reduced_out_parallel_desc, reduced_in_parallel_distribution,
                                   reduced_out_parallel_distribution);
  }
}

std::shared_ptr<ChainSubTskGphBuilder> Make1DSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  if (!Global<ResourceDesc, ForSession>::Get()->nccl_use_compute_stream()) {
    builders.emplace_back(new CollectiveBoxingSubTskGphBuilder());
  }
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  return std::make_shared<ChainSubTskGphBuilder>(builders);
}

bool ParallelDistributionAllSameSplitParallel(const ParallelDistribution& parallel_distribution) {
  CHECK_GT(parallel_distribution.sbp_parallel_size(), 0);
  const SbpParallel& first_sbp = parallel_distribution.sbp_parallel(0);
  if (!first_sbp.has_split_parallel()) { return false; }
  FOR_RANGE(int64_t, i, 1, parallel_distribution.sbp_parallel_size()) {
    if (parallel_distribution.sbp_parallel(i) != first_sbp) { return false; }
  }
  return true;
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
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    if (in_parallel_desc.hierarchy()->NumAxes() == 1
        && out_parallel_desc.hierarchy()->NumAxes() == 1) {
      return sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, in_parallel_distribution.sbp_parallel(0),
          out_parallel_distribution.sbp_parallel(0), time_shape);
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
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    if (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy()
        && in_parallel_desc.hierarchy()->NumAxes() == 2
        && in_parallel_distribution.sbp_parallel(0) == out_parallel_distribution.sbp_parallel(0)
        && in_parallel_distribution.sbp_parallel(1) != out_parallel_distribution.sbp_parallel(1)) {
      const auto& hierarchy = in_parallel_desc.hierarchy();
      std::vector<SubTskGphBuilderStatus> status;
      const int64_t num_groups = hierarchy->At(0);
      const int64_t group_size = hierarchy->At(1);
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
          in_tasks.push_back(sorted_in_tasks.at(parallel_id));
          in_parallel_conf.add_device_name(
              std::to_string(JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
          out_parallel_conf.add_device_name(
              std::to_string(JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
        }
        DimVector dim_vec = logical_blob_desc.shape().dim_vec();
        if (in_parallel_distribution.sbp_parallel(0).has_split_parallel()) {
          const int64_t axis = in_parallel_distribution.sbp_parallel(0).split_parallel().axis();
          dim_vec.at(axis) /= hierarchy->At(0);
        }
        BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
        std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
            JUST(sub_tsk_gph_builder_->Build(
                ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
                ParallelDesc(out_parallel_conf), lbi, new_blob_desc,
                in_parallel_distribution.sbp_parallel(1), out_parallel_distribution.sbp_parallel(1),
                time_shape));
        status.push_back(*boxing_builder_status);
        CHECK_EQ_OR_RETURN(out_tasks.size(), group_size);
        FOR_RANGE(int64_t, j, 0, group_size) {
          const int64_t parallel_id = i * group_size + j;
          sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
          if (!ctrl_tasks.empty()) {
            for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
              sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
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
                                      const ParallelDistribution& in_parallel_distribution,
                                      const ParallelDistribution& out_parallel_distribution,
                                      const Shape& time_shape) const override {
    if (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy()
        && in_parallel_desc.hierarchy()->NumAxes() == 2
        && in_parallel_distribution.sbp_parallel(1) == out_parallel_distribution.sbp_parallel(1)
        && in_parallel_distribution.sbp_parallel(0) != out_parallel_distribution.sbp_parallel(0)
        && !ParallelDistributionAllSameSplitParallel(in_parallel_distribution)
        && !ParallelDistributionAllSameSplitParallel(out_parallel_distribution)) {
      const auto& hierarchy = in_parallel_desc.hierarchy();
      std::vector<SubTskGphBuilderStatus> status;
      const int64_t num_groups = hierarchy->At(0);
      const int64_t group_size = hierarchy->At(1);
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
          in_tasks.push_back(sorted_in_tasks.at(parallel_id));
          in_parallel_conf.add_device_name(
              std::to_string(JUST(in_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(in_parallel_desc.DeviceId4ParallelId(parallel_id))));
          out_parallel_conf.add_device_name(
              std::to_string(JUST(out_parallel_desc.MachineId4ParallelId(parallel_id))) + ":"
              + std::to_string(JUST(out_parallel_desc.DeviceId4ParallelId(parallel_id))));
        }
        DimVector dim_vec = logical_blob_desc.shape().dim_vec();
        if (in_parallel_distribution.sbp_parallel(1).has_split_parallel()) {
          const int64_t axis = in_parallel_distribution.sbp_parallel(1).split_parallel().axis();
          dim_vec.at(axis) /= hierarchy->At(1);
        }
        BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type());
        std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
            JUST(sub_tsk_gph_builder_->Build(
                ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
                ParallelDesc(out_parallel_conf), lbi, new_blob_desc,
                in_parallel_distribution.sbp_parallel(0), out_parallel_distribution.sbp_parallel(0),
                time_shape));
        status.push_back(*boxing_builder_status);
        CHECK_EQ_OR_RETURN(out_tasks.size(), num_groups);
        FOR_RANGE(int64_t, j, 0, num_groups) {
          const int64_t parallel_id = j * group_size + i;
          sorted_out_tasks->at(parallel_id) = out_tasks.at(j);
          if (!ctrl_tasks.empty()) {
            for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {
              sorted_ctrl_tasks->at(parallel_id).push_back(ctrl_node);
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

struct DispatchHierarchicalSubTskGphBuilder::Impl {
  Impl();
  std::unique_ptr<FlatSubTskGphBuilder> flat_sub_tsk_gph_builder_;
  std::unique_ptr<IntraGroupSubTskGphBuilder> intra_group_sub_tsk_gph_builder_;
  std::unique_ptr<InterGroupSubTskGphBuilder> inter_group_sub_tsk_gph_builder_;
};

DispatchHierarchicalSubTskGphBuilder::Impl::Impl() {
  flat_sub_tsk_gph_builder_.reset(new FlatSubTskGphBuilder());
  intra_group_sub_tsk_gph_builder_.reset(new IntraGroupSubTskGphBuilder());
  inter_group_sub_tsk_gph_builder_.reset(new InterGroupSubTskGphBuilder());
}

DispatchHierarchicalSubTskGphBuilder::DispatchHierarchicalSubTskGphBuilder() {
  impl_.reset(new Impl());
}

DispatchHierarchicalSubTskGphBuilder::~DispatchHierarchicalSubTskGphBuilder() = default;

Maybe<SubTskGphBuilderStatus> DispatchHierarchicalSubTskGphBuilder::BuildSame2DHierarchySubGraph(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) const {
  if ((in_parallel_desc.hierarchy()->NumAxes() == 2)
      && (*in_parallel_desc.hierarchy() == *out_parallel_desc.hierarchy())) {
    if (in_parallel_distribution.sbp_parallel(0) == out_parallel_distribution.sbp_parallel(0)) {
      return impl_->intra_group_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
          out_parallel_desc, lbi, logical_blob_desc, in_parallel_distribution,
          out_parallel_distribution, time_shape);
    } else if (in_parallel_distribution.sbp_parallel(1)
               == out_parallel_distribution.sbp_parallel(1)) {
      if (!(ParallelDistributionAllSameSplitParallel(in_parallel_distribution)
            || ParallelDistributionAllSameSplitParallel(out_parallel_distribution))) {
        return impl_->inter_group_sub_tsk_gph_builder_->Build(
            ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
            out_parallel_desc, lbi, logical_blob_desc, in_parallel_distribution,
            out_parallel_distribution, time_shape);
      } else {
        return Error::BoxingNotSupportedError();
      }
    } else {
      return Error::BoxingNotSupportedError();
    }
  } else {
    return Error::BoxingNotSupportedError();
  }
}

Maybe<SubTskGphBuilderStatus> DispatchHierarchicalSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const ParallelDistribution& in_parallel_distribution,
    const ParallelDistribution& out_parallel_distribution, const Shape& time_shape) const {
  ParallelDesc reduced_in_parallel_desc = in_parallel_desc;
  ParallelDesc reduced_out_parallel_desc = out_parallel_desc;
  ParallelDistribution reduced_in_parallel_distribution;
  ParallelDistribution reduced_out_parallel_distribution;
  InOutParallelDimReduce(in_parallel_desc, out_parallel_desc, in_parallel_distribution,
                         out_parallel_distribution, &reduced_in_parallel_desc,
                         &reduced_out_parallel_desc, &reduced_in_parallel_distribution,
                         &reduced_out_parallel_distribution);
  const auto& in_hierarchy = reduced_in_parallel_desc.hierarchy();
  const auto& out_hierarchy = reduced_out_parallel_desc.hierarchy();
  if (in_hierarchy->NumAxes() <= 2 && out_hierarchy->NumAxes() <= 2) {
    if (in_hierarchy->NumAxes() == 1 && out_hierarchy->NumAxes() == 1) {
      return impl_->flat_sub_tsk_gph_builder_->Build(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          reduced_out_parallel_desc, lbi, logical_blob_desc, reduced_in_parallel_distribution,
          reduced_out_parallel_distribution, time_shape);
    } else if ((in_hierarchy->NumAxes() == 2) && (*in_hierarchy == *out_hierarchy)) {
      return BuildSame2DHierarchySubGraph(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                                          reduced_in_parallel_desc, reduced_out_parallel_desc, lbi,
                                          logical_blob_desc, reduced_in_parallel_distribution,
                                          reduced_out_parallel_distribution, time_shape);
    } else if (in_hierarchy->NumAxes() == 2 && out_hierarchy->NumAxes() == 1
               && in_hierarchy->elem_cnt() == out_hierarchy->elem_cnt()) {
      ParallelConf intermediate_parallel_conf = reduced_out_parallel_desc.parallel_conf();
      reduced_in_parallel_desc.hierarchy()->ToProto(intermediate_parallel_conf.mutable_hierarchy());
      ParallelDistribution intermediate_parallel_distribution;
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_out_parallel_distribution.sbp_parallel(0);
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_out_parallel_distribution.sbp_parallel(0);
      return BuildSame2DHierarchySubGraph(
          ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, reduced_in_parallel_desc,
          ParallelDesc(intermediate_parallel_conf), lbi, logical_blob_desc,
          reduced_in_parallel_distribution, intermediate_parallel_distribution, time_shape);
    } else if (in_hierarchy->NumAxes() == 1 && out_hierarchy->NumAxes() == 2
               && in_hierarchy->elem_cnt() == out_hierarchy->elem_cnt()) {
      ParallelConf intermediate_parallel_conf = reduced_in_parallel_desc.parallel_conf();
      reduced_out_parallel_desc.hierarchy()->ToProto(
          intermediate_parallel_conf.mutable_hierarchy());
      ParallelDistribution intermediate_parallel_distribution;
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_in_parallel_distribution.sbp_parallel(0);
      *intermediate_parallel_distribution.add_sbp_parallel() =
          reduced_in_parallel_distribution.sbp_parallel(0);
      return BuildSame2DHierarchySubGraph(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                                          ParallelDesc(intermediate_parallel_conf),
                                          reduced_out_parallel_desc, lbi, logical_blob_desc,
                                          intermediate_parallel_distribution,
                                          reduced_out_parallel_distribution, time_shape);
    }
  }
  return Error::BoxingNotSupportedError();
}

}  // namespace oneflow
