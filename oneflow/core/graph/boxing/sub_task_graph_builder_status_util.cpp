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
#include "oneflow/core/graph/boxing/sub_task_graph_builder_status_util.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(
    const CompTaskNode* src_node, const CompTaskNode* dst_node,
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
    const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc, const std::string& builder_name,
    const std::string& comment) {
  std::string src_op_name = src_node->logical_node()->op_vec().at(0)->op_name();
  std::string dst_op_name = dst_node->logical_node()->op_vec().at(0)->op_name();
  SubTskGphBuilderStatus status(src_op_name, dst_op_name, src_parallel_desc, dst_parallel_desc,
                                src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
                                builder_name, comment);

  return status;
}

}  // namespace oneflow
