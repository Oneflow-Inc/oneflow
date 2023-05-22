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
#ifndef ONEFLOW_CORE_GRAPH_BOXING_CCL_SUB_TASK_GRAPH_BUILDER_H_
#define ONEFLOW_CORE_GRAPH_BOXING_CCL_SUB_TASK_GRAPH_BUILDER_H_

#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"

namespace oneflow {

bool IsSourceTimeShape(const Shape& shape);

class CclAllReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclAllReduceSubTskGphBuilder);
  CclAllReduceSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclAllReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclReduceScatterSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclReduceScatterSubTskGphBuilder);
  CclReduceScatterSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclReduceScatterSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclP2SNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclP2SNoncontinuousSubTskGphBuilder);
  CclP2SNoncontinuousSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclP2SNoncontinuousSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclAllGatherSubTskGphBuilder);
  CclAllGatherSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclS2BNoncontinuousSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclS2BNoncontinuousSubTskGphBuilder);
  CclS2BNoncontinuousSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclS2BNoncontinuousSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclReduceSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclReduceSubTskGphBuilder);
  CclReduceSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclReduceSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclScatterThenAllGatherSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclScatterThenAllGatherSubTskGphBuilder);
  CclScatterThenAllGatherSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclScatterThenAllGatherSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclBroadcastSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclBroadcastSubTskGphBuilder);
  CclBroadcastSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclBroadcastSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

class CclAll2AllSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CclAll2AllSubTskGphBuilder);
  CclAll2AllSubTskGphBuilder(DeviceType device_type) : device_type_(device_type) {}
  ~CclAll2AllSubTskGphBuilder() override = default;

  Maybe<SubTskGphBuilderStatus> Build(
      SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
      std::vector<TaskNode*>* sorted_out_tasks,
      std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
      const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
      const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
      const SbpParallel& out_sbp_parallel, const Shape& time_shape) const override;

 private:
  DeviceType device_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_CCL_SUB_TASK_GRAPH_BUILDER_H_
