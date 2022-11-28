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

#include <memory>
#include "oneflow/core/auto_parallel/sbp_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/graph/boxing/hierarchical_sub_task_graph_builder_impl.h"

namespace oneflow {
namespace auto_parallel {

// Judge whether we need the same SBP for both producer and consumer
bool RequireSameSbp(const OpNode* consumer, const std::string& ibn) {
  // is mutable
  const auto& input_blob_modifier_ = consumer->op().InputBlobModifier4Ibn(ibn);
  if (input_blob_modifier_.has_is_mutable() && input_blob_modifier_.is_mutable()) { return true; }
  // kOFRecord or kTensorBuffer don't accept boxing
  const LogicalBlobId& lbi = consumer->op().BnInOp2Lbi(ibn);
  const OpNode& producer = consumer->ProducerOpNode4Lbi(lbi);
  const BlobDesc& logical_blob_desc = producer.LogicalBlobDesc4Lbi(lbi);
  return (logical_blob_desc.data_type() == DataType::kOFRecord
          || logical_blob_desc.data_type() == DataType::kTensorBuffer);
}

}  // namespace auto_parallel
}  // namespace oneflow
