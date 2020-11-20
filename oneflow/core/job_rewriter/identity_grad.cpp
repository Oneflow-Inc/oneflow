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

void GenerateIdentityBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    IdentityOpConf* identity_op_conf = grad_op.mutable_identity_conf();
    identity_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    identity_op_conf->set_out("out");
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(identity_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kIdentityConf, &GenerateIdentityBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kCopyConf, &GenerateIdentityBackwardOpConf);

}  // namespace

void GenerateBwSbpParallel(SbpParallel* bw_sbp_parallel, const SbpParallel& fw_sbp_parallel) {
  if (fw_sbp_parallel.has_split_parallel()) {
    *bw_sbp_parallel = fw_sbp_parallel;
  } else if (fw_sbp_parallel.has_broadcast_parallel()) {
    bw_sbp_parallel->mutable_partial_sum_parallel();
  } else if (fw_sbp_parallel.has_partial_sum_parallel()) {
    bw_sbp_parallel->mutable_broadcast_parallel();
  } else {
    UNIMPLEMENTED();
  }
}

namespace {

void GenerateCastToMirroredBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_cast_to_mirrored_conf());
  const auto& fw_op_conf = op.op_conf().cast_to_mirrored_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    grad_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
    CastFromMirroredOpConf* bw_op_conf = grad_op.mutable_cast_from_mirrored_conf();
    bw_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    bw_op_conf->set_out("out");
    GenerateBwSbpParallel(bw_op_conf->mutable_sbp_parallel(), fw_op_conf.sbp_parallel());
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(bw_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kCastToMirroredConf, &GenerateCastToMirroredBackwardOpConf);

}  // namespace

namespace {

void GenerateCastFromMirroredBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_cast_from_mirrored_conf());
  const auto& fw_op_conf = op.op_conf().cast_from_mirrored_conf();
  if (DiffLbi4BnInOp("in") != nullptr) {
    OperatorConf grad_op{};
    grad_op.set_name("System-AutoGrad-" + op.op_name());
    grad_op.set_scope_symbol_id(op.op_conf().scope_symbol_id());
    CastToMirroredOpConf* bw_op_conf = grad_op.mutable_cast_to_mirrored_conf();
    bw_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    bw_op_conf->set_out("out");
    GenerateBwSbpParallel(bw_op_conf->mutable_sbp_parallel(), fw_op_conf.sbp_parallel());
    op_confs->push_back(grad_op);
    DiffLbi4BnInOp("in")->set_op_name(grad_op.name());
    DiffLbi4BnInOp("in")->set_blob_name(bw_op_conf->out());
  }
}

REGISTER_OP_GRAD(OperatorConf::kCastFromMirroredConf, &GenerateCastFromMirroredBackwardOpConf);

}  // namespace

}  // namespace oneflow
