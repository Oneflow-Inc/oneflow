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
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_util.h"
#include "oneflow/cambricon/graph/boxing/cncl_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/fallback_to_cpu_slice_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/b21_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/framework/sbp_infer_util.h"

namespace oneflow {

class MluHierarchicalSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MluHierarchicalSubTskGphBuilder);
  MluHierarchicalSubTskGphBuilder();
  ~MluHierarchicalSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(SubTskGphBuilderCtx* ctx,
                                      const std::vector<TaskNode*>& sorted_in_tasks,
                                      std::vector<TaskNode*>* sorted_out_tasks,
                                      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks,
                                      const ParallelDesc& in_parallel_desc,
                                      const ParallelDesc& out_parallel_desc,
                                      const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                                      const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
                                      const Shape& time_shape) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace {

std::shared_ptr<ChainSubTskGphBuilder> Make1DSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new OneToOneSubTskGphBuilder());
  builders.emplace_back(new B21SubTskGphBuilder());
  builders.emplace_back(new CnclSubTskGphBuilder());
  builders.emplace_back(new SliceBoxingSubTskGphBuilder());
  builders.emplace_back(new FallbackToCpuSliceBoxingSubTskGphBuilder());
  builders.emplace_back(new NaiveB2BSubTskGphBuilder());
  builders.emplace_back(new NaiveB2PSubTskGphBuilder());
  return std::make_shared<ChainSubTskGphBuilder>(builders);
}

class Dim0NdSbpMismatchedSubTskGphBuilder final : public HierarchicalSubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dim0NdSbpMismatchedSubTskGphBuilder);
  Dim0NdSbpMismatchedSubTskGphBuilder() {
    inter_group_sub_tsk_gph_builder_.reset(
        new InterGroupSubTskGphBuilder(Make1DSubTskGphBuilder()));
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
    intra_group_sub_tsk_gph_builder_.reset(
        new IntraGroupSubTskGphBuilder(Make1DSubTskGphBuilder()));
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

}  // namespace

struct MluHierarchicalSubTskGphBuilder::Impl {
  Impl();
  std::unique_ptr<FlatSubTskGphBuilder> flat_sub_tsk_gph_builder_;
  std::unique_ptr<Same2DHierarchySubTskGphBuilder> same_2d_hierarchy_sub_tsk_gph_builder_;
};

MluHierarchicalSubTskGphBuilder::Impl::Impl() {
  flat_sub_tsk_gph_builder_.reset(
      new FlatSubTskGphBuilder(std::make_shared<CnclSubTskGphBuilder>()));
  same_2d_hierarchy_sub_tsk_gph_builder_.reset(new Same2DHierarchySubTskGphBuilder());
}

MluHierarchicalSubTskGphBuilder::MluHierarchicalSubTskGphBuilder() { impl_.reset(new Impl()); }

Maybe<SubTskGphBuilderStatus> MluHierarchicalSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
    const Shape& time_shape) const {
  if (in_parallel_desc.device_type() != DeviceType::kMLU
      && in_parallel_desc.device_type() != DeviceType::kMLU) {
    return Error::BoxingNotSupportedError();
  }
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
    } else {
      return Error::BoxingNotSupportedError();
    }
  }
  return Error::BoxingNotSupportedError();
}

REGISTER_CREATE_SUB_TASK_GRAPH_BUILDER_FN([]() -> std::unique_ptr<HierarchicalSubTskGphBuilder> {
  return std::make_unique<MluHierarchicalSubTskGphBuilder>();
});

}  // namespace oneflow
