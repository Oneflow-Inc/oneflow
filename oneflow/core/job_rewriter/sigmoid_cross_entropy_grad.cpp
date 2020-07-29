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

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_sigmoid_cross_entropy_conf());
  if (DiffLbi4BnInOp("prediction") != nullptr) {
    OperatorConf sigmoid_cross_entropy_grad_op;
    sigmoid_cross_entropy_grad_op.set_name(op.op_name() + "_grad");

    SigmoidCrossEntropyGradOpConf* sigmoid_cross_entropy_grad_op_conf =
        sigmoid_cross_entropy_grad_op.mutable_sigmoid_cross_entropy_grad_conf();
    sigmoid_cross_entropy_grad_op_conf->set_label(GenLogicalBlobName(op.BnInOp2Lbi("label")));
    sigmoid_cross_entropy_grad_op_conf->set_prediction(
        GenLogicalBlobName(op.BnInOp2Lbi("prediction")));
    sigmoid_cross_entropy_grad_op_conf->set_loss_diff(GenLogicalBlobName(*DiffLbi4BnInOp("loss")));
    sigmoid_cross_entropy_grad_op_conf->set_prediction_diff("prediction_diff");
    sigmoid_cross_entropy_grad_op_conf->set_label_type(
        op.op_conf().sigmoid_cross_entropy_conf().label_type());

    op_confs->push_back(sigmoid_cross_entropy_grad_op);
    DiffLbi4BnInOp("prediction")->set_op_name(sigmoid_cross_entropy_grad_op.name());
    DiffLbi4BnInOp("prediction")->set_blob_name("prediction_diff");
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kSigmoidCrossEntropyConf, &GenerateBackwardOpConf);

}  // namespace oneflow
