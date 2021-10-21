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
#include "oneflow/core/job_rewriter/autograd.h"

namespace oneflow {

namespace {

void GenerateBackwardOpConf4DynamicReshape(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_dynamic_reshape_conf());
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf reverse_reshape_op;
    reverse_reshape_op.set_name(op.op_name() + "_grad");
    DynamicReshapeLikeOpConf* reshape_like_op_conf =
        reverse_reshape_op.mutable_dynamic_reshape_like_conf();
    reshape_like_op_conf->set_x(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    reshape_like_op_conf->set_like(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    reshape_like_op_conf->set_y("y");
    op_confs->push_back(reverse_reshape_op);
    DiffLbi4BnInOp("in")->set_op_name(reverse_reshape_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("y");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDynamicReshapeConf, GenerateBackwardOpConf4DynamicReshape);

}  // namespace oneflow
