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
#ifndef ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
#define ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_

#include "oneflow/core/graph/compute_task_node.h"

namespace oneflow {

class SubTskGphBuilderStatus;

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(
    const TaskNode* src_node, const TaskNode* dst_node, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const std::string& builder_name, const std::string& comment);

class SubTskGphBuilderStatus final {
 public:
  SubTskGphBuilderStatus(const ParallelDesc& src_parallel_desc,
                         const ParallelDesc& dst_parallel_desc,
                         const SbpParallel& src_sbp_parallel_, const SbpParallel& dst_sbp_parallel,
                         const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                         const std::string& builder_name, const std::string& comment)
      : src_parallel_desc_(src_parallel_desc),
        dst_parallel_desc_(dst_parallel_desc),
        src_sbp_parallel_(src_sbp_parallel_),
        dst_sbp_parallel_(dst_sbp_parallel),
        lbi_(lbi),
        logical_blob_desc_(logical_blob_desc),
        builder_name_(builder_name),
        comment_(comment){};
  ~SubTskGphBuilderStatus() = default;

  // Getters
  const ParallelDesc& src_parallel_desc() const { return src_parallel_desc_; }
  const ParallelDesc& dst_parallel_desc() const { return dst_parallel_desc_; }
  const SbpParallel& src_sbp_parallel() const { return src_sbp_parallel_; }
  const SbpParallel& dst_sbp_parallel() const { return dst_sbp_parallel_; }
  const LogicalBlobId& lbi() const { return lbi_; }
  const BlobDesc& logical_blob_desc() const { return logical_blob_desc_; }
  const std::string& builder_name() const { return builder_name_; }
  const std::string& comment() const { return comment_; }

 private:
  ParallelDesc src_parallel_desc_;
  ParallelDesc dst_parallel_desc_;
  SbpParallel src_sbp_parallel_;
  SbpParallel dst_sbp_parallel_;
  LogicalBlobId lbi_;
  BlobDesc logical_blob_desc_;
  std::string builder_name_;
  std::string comment_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
