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
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> FlatSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
    const Shape& time_shape) const {
  if (in_parallel_desc.hierarchy()->NumAxes() == 1
      && out_parallel_desc.hierarchy()->NumAxes() == 1) {
    return sub_tsk_gph_builder_->Build(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                                       in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
                                       in_nd_sbp.sbp_parallel(0), out_nd_sbp.sbp_parallel(0),
                                       time_shape);
  } else {
    return Error::BoxingNotSupportedError();
  }
}

Maybe<SubTskGphBuilderStatus> IntraGroupSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
    const Shape& time_shape) const {
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
        in_tasks.emplace_back(sorted_in_tasks.at(parallel_id));  // NOLINT
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
      BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type(),
                             logical_blob_desc.memory_format());
      std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
          JUST(sub_tsk_gph_builder_->Build(
              ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
              ParallelDesc(out_parallel_conf), lbi, new_blob_desc, in_nd_sbp.sbp_parallel(1),
              out_nd_sbp.sbp_parallel(1), time_shape));
      status.emplace_back(*boxing_builder_status);
      CHECK_EQ_OR_RETURN(out_tasks.size(), group_size);  // NOLINT
      FOR_RANGE(int64_t, j, 0, group_size) {
        const int64_t parallel_id = i * group_size + j;
        sorted_out_tasks->at(parallel_id) = out_tasks.at(j);  // NOLINT
        if (!ctrl_tasks.empty()) {
          for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {                 // NOLINT
            sorted_ctrl_tasks->at(parallel_id).emplace_back(ctrl_node);  // NOLINT
          }
        }
      }
    }
    return MakeComposedSubTskGphBuilderStatus(status);
  } else {
    return Error::BoxingNotSupportedError();
  }
}

Maybe<SubTskGphBuilderStatus> InterGroupSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const NdSbp& in_nd_sbp, const NdSbp& out_nd_sbp,
    const Shape& time_shape) const {
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
        in_tasks.emplace_back(sorted_in_tasks.at(parallel_id));  // NOLINT
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
      BlobDesc new_blob_desc(Shape(dim_vec), logical_blob_desc.data_type(),
                             logical_blob_desc.memory_format());
      std::shared_ptr<SubTskGphBuilderStatus> boxing_builder_status =
          JUST(sub_tsk_gph_builder_->Build(
              ctx, in_tasks, &out_tasks, &ctrl_tasks, ParallelDesc(in_parallel_conf),
              ParallelDesc(out_parallel_conf), lbi, new_blob_desc, in_nd_sbp.sbp_parallel(0),
              out_nd_sbp.sbp_parallel(0), time_shape));
      status.emplace_back(*boxing_builder_status);
      CHECK_EQ_OR_RETURN(out_tasks.size(), num_groups);  // NOLINT
      FOR_RANGE(int64_t, j, 0, num_groups) {
        const int64_t parallel_id = j * group_size + i;
        sorted_out_tasks->at(parallel_id) = out_tasks.at(j);  // NOLINT
        if (!ctrl_tasks.empty()) {
          for (TaskNode* ctrl_node : ctrl_tasks.at(j)) {                 // NOLINT
            sorted_ctrl_tasks->at(parallel_id).emplace_back(ctrl_node);  // NOLINT
          }
        }
      }
    }
    return MakeComposedSubTskGphBuilderStatus(status);
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
