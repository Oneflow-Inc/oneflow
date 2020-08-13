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

std::string SerializeSbpParallel(const SbpParallel& sbp_parallel) {
  std::string serialized_sbp_parallel = "";
  if (sbp_parallel.has_broadcast_parallel()) {
    serialized_sbp_parallel = "B";
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    serialized_sbp_parallel = "P";

  } else if (sbp_parallel.has_split_parallel()) {
    serialized_sbp_parallel = "S(" + std::to_string(sbp_parallel.split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED();
  }
  return serialized_sbp_parallel;
}

std::string SerializeParallelDesc(const ParallelDesc& parallel_desc) {
  std::string serialized_parallel_desc;
  serialized_parallel_desc = PbMessage2TxtString(parallel_desc.parallel_conf());
  StringReplace(&serialized_parallel_desc, '\n', ' ');
  serialized_parallel_desc.pop_back();
  return serialized_parallel_desc;
}

std::string SerializeLogicalBlobId(const LogicalBlobId& lbi) {
  std::string lbi_info = "";
  lbi_info += "op_name: " + lbi.op_name() + " ";
  lbi_info += "blob_name: " + GenLogicalBlobName(lbi);
  return lbi_info;
}

std::string GetBlobInfo4LogicalBlobDesc(const BlobDesc& logical_blob_desc) {
  std::string blob_desc_info = "dtype: ";
  blob_desc_info += DataType_Name(logical_blob_desc.data_type()) + " ";
  auto shape_info = logical_blob_desc.shape().ToString();
  StringReplace(&shape_info, ',', ' ');
  blob_desc_info += " shape: " + shape_info;
  return blob_desc_info;
}

Maybe<SubTskGphBuilderStatus> BuildSubTskGphBuilderStatus(
    const CompTaskNode* src_node, const CompTaskNode* dst_node,
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
    const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc, const std::string& builder_name,
    const std::string& boxing_type) {
  std::string src_op_name = src_node->logical_node()->op_vec().at(0)->op_name();
  std::string dst_op_name = dst_node->logical_node()->op_vec().at(0)->op_name();
  SubTskGphBuilderStatus status(src_op_name, dst_op_name, src_parallel_desc, dst_parallel_desc,
                                src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
                                builder_name, boxing_type);

  return status;
}

std::string SubTskGphBuilderStatus::ToString() const {
  std::string serialized_status("");
  serialized_status += src_op_name_ + ",";
  serialized_status += SerializeParallelDesc(src_parallel_desc_) + ",";
  serialized_status += SerializeSbpParallel(src_sbp_parallel_) + ",";
  serialized_status += SerializeLogicalBlobId(lbi_) + ",";
  serialized_status += GetBlobInfo4LogicalBlobDesc(logical_blob_desc_) + ",";
  if (boxing_type_ == std::string("")) {
    serialized_status += builder_name_ + ",";
  } else {
    serialized_status += builder_name_ + ":" + boxing_type_ + ",";
  }

  serialized_status += dst_op_name_ + ",";
  serialized_status += SerializeParallelDesc(dst_parallel_desc_) + ",";
  serialized_status += SerializeSbpParallel(dst_sbp_parallel_) + "\n";
  return serialized_status;
}

}  // namespace oneflow
