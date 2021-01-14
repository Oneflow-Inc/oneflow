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
#ifndef ONEFLOW_CORE_OPERATOR_SIG_DESC_H_
#define ONEFLOW_CORE_OPERATOR_SIG_DESC_H_

#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/parallel_signature.pb.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace cfg {

class OpNodeSignature;
}

class OpNodeSignatureDesc final {
 public:
  OpNodeSignatureDesc(const OpNodeSignatureDesc&) = delete;
  OpNodeSignatureDesc(OpNodeSignatureDesc&&) = delete;
  OpNodeSignatureDesc(int64_t symbol_id, const OpNodeSignature& op_node_signature);

  const Maybe<int64_t>& symbol_id() const { return symbol_id_; }
  const std::shared_ptr<cfg::OpNodeSignature>& cfg_op_node_signature() const {
    return cfg_op_node_signature_;
  }
  const SbpSignature& sbp_signature() const { return op_node_signature_.sbp_signature(); }
  const ParallelSignature& parallel_signature() const {
    return op_node_signature_.parallel_signature();
  }

  Maybe<const BlobDesc&> LogicalBlobDesc4BnInOp(const std::string& bn_in_op) const;

 private:
  Maybe<int64_t> symbol_id_;
  OpNodeSignature op_node_signature_;
  std::shared_ptr<cfg::OpNodeSignature> cfg_op_node_signature_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn_in_op2blob_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SIG_DESC_H_
