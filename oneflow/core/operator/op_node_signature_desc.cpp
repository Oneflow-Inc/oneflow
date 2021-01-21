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
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"

namespace oneflow {

OpNodeSignatureDesc::OpNodeSignatureDesc(int64_t symbol_id,
                                         const OpNodeSignature& op_node_signature)
    : symbol_id_(symbol_id),
      op_node_signature_(op_node_signature),
      cfg_op_node_signature_(std::make_shared<cfg::OpNodeSignature>(op_node_signature)) {
  const auto& logical_blob_desc_sig = op_node_signature.logical_blob_desc_signature();
  for (const auto& pair : logical_blob_desc_sig.bn_in_op2blob_desc()) {
    auto blob_desc = std::make_unique<BlobDesc>(pair.second);
    CHECK(bn_in_op2blob_desc_.emplace(pair.first, std::move(blob_desc)).second);
  }
}

Maybe<const BlobDesc&> OpNodeSignatureDesc::LogicalBlobDesc4BnInOp(
    const std::string& bn_in_op) const {
  const auto& iter = bn_in_op2blob_desc_.find(bn_in_op);
  CHECK_OR_RETURN(iter != bn_in_op2blob_desc_.end());
  return *iter->second;
}

}  // namespace oneflow
