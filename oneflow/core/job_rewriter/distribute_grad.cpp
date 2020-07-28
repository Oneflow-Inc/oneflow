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

Maybe<void> GenerateBackwardOpConf4Concat(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_distribute_concat_conf());
  const DistributeConcatOpConf& distribute_concat_conf = op.op_conf().distribute_concat_conf();
  OperatorConf split_op;
  split_op.set_name(op.op_conf().name() + "_grad");
  DistributeSplitOpConf* split_op_conf = split_op.mutable_distribute_split_conf();
  split_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  split_op_conf->set_axis(distribute_concat_conf.axis());
  FOR_RANGE(int32_t, i, 0, distribute_concat_conf.in_size()) {
    const std::string& ibn_of_distribute_concat_op = op.input_bns().Get(i);
    const std::string& obn = "out_" + std::to_string(i);
    split_op_conf->add_out(obn);
    if (DiffLbi4BnInOp(ibn_of_distribute_concat_op) != nullptr) {
      DiffLbi4BnInOp(ibn_of_distribute_concat_op)->set_op_name(split_op.name());
      DiffLbi4BnInOp(ibn_of_distribute_concat_op)->set_blob_name(obn);
    }
  }
  op_confs->push_back(split_op);
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4Split(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_distribute_split_conf());
  const DistributeSplitOpConf& distribute_split_conf = op.op_conf().distribute_split_conf();
  OperatorConf concat_op;
  concat_op.set_name(op.op_conf().name() + "_grad");
  DistributeConcatOpConf* concat_op_conf = concat_op.mutable_distribute_concat_conf();
  concat_op_conf->set_axis(distribute_split_conf.axis());
  const bool has_diff_0 = DiffLbi4BnInOp(op.output_bns().Get(0)) != nullptr;
  FOR_RANGE(int32_t, i, 0, distribute_split_conf.out_size()) {
    const std::string& obn_of_distribute_split_op = op.output_bns().Get(i);
    const bool has_diff_i = DiffLbi4BnInOp(obn_of_distribute_split_op) != nullptr;
    CHECK_EQ(has_diff_i, has_diff_0);
    if (has_diff_i) {
      concat_op_conf->add_in(GenLogicalBlobName(*DiffLbi4BnInOp(obn_of_distribute_split_op)));
    }
  }
  concat_op_conf->set_out("out");
  if (DiffLbi4BnInOp("in") != nullptr) {
    CHECK_EQ(concat_op_conf->in_size(), distribute_split_conf.out_size());
    DiffLbi4BnInOp("in")->set_op_name(concat_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
    op_confs->push_back(concat_op);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4Clone(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_distribute_clone_conf());
  const DistributeCloneOpConf& conf = op.op_conf().distribute_clone_conf();
  OperatorConf partial_op;
  partial_op.set_name(op.op_conf().name() + "_grad");
  DistributeAddOpConf* partial_op_conf = partial_op.mutable_distribute_add_conf();
  const bool has_diff_0 = DiffLbi4BnInOp(op.output_bns().Get(0)) != nullptr;
  FOR_RANGE(int32_t, i, 0, conf.out_size()) {
    const std::string& obn_of_distribute_clone_op = op.output_bns().Get(i);
    const bool has_diff_i = DiffLbi4BnInOp(obn_of_distribute_clone_op) != nullptr;
    CHECK_EQ(has_diff_i, has_diff_0);
    if (has_diff_i) {
      partial_op_conf->add_in(GenLogicalBlobName(*DiffLbi4BnInOp(obn_of_distribute_clone_op)));
    }
  }
  partial_op_conf->set_out("out");
  if (DiffLbi4BnInOp("in") != nullptr) {
    CHECK_EQ(partial_op_conf->in_size(), conf.out_size());
    DiffLbi4BnInOp("in")->set_op_name(partial_op.name());
    DiffLbi4BnInOp("in")->set_blob_name("out");
    op_confs->push_back(partial_op);
  }
  return Maybe<void>::Ok();
}

Maybe<void> GenerateBackwardOpConf4Add(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp,
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4BnInOp) {
  CHECK(op.op_conf().has_distribute_add_conf());
  const auto& distribute_add_conf = op.op_conf().distribute_add_conf();
  OperatorConf broadcast_op;
  broadcast_op.set_name(op.op_conf().name() + "_grad");
  DistributeCloneOpConf* broadcast_op_conf = broadcast_op.mutable_distribute_clone_conf();
  broadcast_op_conf->set_in(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
  FOR_RANGE(int32_t, i, 0, distribute_add_conf.in_size()) {
    const std::string& ibn_of_distribute_add_op = op.input_bns().Get(i);
    const std::string& obn = "out_" + std::to_string(i);
    broadcast_op_conf->add_out(obn);
    if (DiffLbi4BnInOp(ibn_of_distribute_add_op) != nullptr) {
      DiffLbi4BnInOp(ibn_of_distribute_add_op)->set_op_name(broadcast_op.name());
      DiffLbi4BnInOp(ibn_of_distribute_add_op)->set_blob_name(obn);
    }
  }
  op_confs->push_back(broadcast_op);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kDistributeConcatConf, &GenerateBackwardOpConf4Concat);
REGISTER_OP_GRAD(OperatorConf::kDistributeSplitConf, &GenerateBackwardOpConf4Split);
REGISTER_OP_GRAD(OperatorConf::kDistributeCloneConf, &GenerateBackwardOpConf4Clone);
REGISTER_OP_GRAD(OperatorConf::kDistributeAddConf, &GenerateBackwardOpConf4Add);

}  // namespace oneflow
