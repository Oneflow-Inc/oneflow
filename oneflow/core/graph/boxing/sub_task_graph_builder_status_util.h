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

#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/graph/compute_task_node.h"
#include "oneflow/core/register/register_desc.pb.h"

namespace oneflow {

class SubTskGphBuilderStatus;

std::string SerializeSbpParallel(const SbpParallel& sbp_parallel);
std::string SerializeParallelDesc(const ParallelDesc& parallel_desc);
std::string SerializeLogicalBlobId(const LogicalBlobId& lbi);
std::string GetBlobInfo4LogicalBlobDesc(const BlobDesc& blob_desc);
Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(
    const CompTaskNode* src_node, const CompTaskNode* dst_node,
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
    const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc, const std::string& builder_name,
    const std::string& boxing_type);

class SubTskGphBuilderStatus final {
 public:
  SubTskGphBuilderStatus(const std::string& src_op_name, const std::string& dst_op_name,
                         const ParallelDesc& src_parallel_desc,
                         const ParallelDesc& dst_parallel_desc,
                         const SbpParallel& src_sbp_parallel_, const SbpParallel& dst_sbp_parallel,
                         const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                         const std::string& builder_name, const std::string& boxing_type)
      : src_op_name_(src_op_name),
        dst_op_name_(dst_op_name),
        src_parallel_desc_(src_parallel_desc),
        dst_parallel_desc_(dst_parallel_desc),
        src_sbp_parallel_(src_sbp_parallel_),
        dst_sbp_parallel_(dst_sbp_parallel),
        lbi_(lbi),
        logical_blob_desc_(logical_blob_desc),
        builder_name_(builder_name),
        boxing_type_(boxing_type){};
  ~SubTskGphBuilderStatus() = default;

  // Getters
  std::string src_op_name() { return src_op_name_; }
  std::string dst_op_name() { return dst_op_name_; }
  ParallelDesc src_parallel_desc() { return src_parallel_desc_; }
  ParallelDesc dst_parallel_desc() { return dst_parallel_desc_; }
  SbpParallel src_sbp_parallel() { return src_sbp_parallel_; }
  SbpParallel dst_sbp_parallel() { return dst_sbp_parallel_; }
  LogicalBlobId lbi() { return lbi_; }
  BlobDesc& logical_blob_desc() { return logical_blob_desc_; }
  std::string builder_name() { return builder_name_; }
  std::string boxing_type() { return boxing_type_; }

  std::string ToString() const;

 private:
  std::string src_op_name_;
  std::string dst_op_name_;
  ParallelDesc src_parallel_desc_;
  ParallelDesc dst_parallel_desc_;
  SbpParallel src_sbp_parallel_;
  SbpParallel dst_sbp_parallel_;
  LogicalBlobId lbi_;
  BlobDesc logical_blob_desc_;
  std::string builder_name_;
  std::string boxing_type_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_SUB_TASK_GRAPH_BUILDER_STATUS_UTIL_H_
