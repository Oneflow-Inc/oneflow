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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/launch_op.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace oneflow {

void XrtLaunchOp::InitFromOpConf() {
  CHECK(op_conf().has_xrt_launch_conf());
  const auto &launch_conf = op_conf().xrt_launch_conf();
  int inputs_num = launch_conf.in().size();
  int outputs_num = launch_conf.out().size();

  const auto &mutability_table = launch_conf.input_mutability();
  for (int i = 0; i < inputs_num; ++i) {
    // const std::string &input = launch_conf.in().at(i);
    const std::string &input = launch_conf.in()[i];
    bool mutability = mutability_table.count(input) > 0;
    EnrollInputBn(absl::StrCat("in_", i))->set_is_mutable(mutability);
  }
  if (outputs_num > 0) { EnrollRepeatedOutputBn("out"); }
}

Maybe<void> XrtLaunchOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const auto &launch_conf = op_conf().xrt_launch_conf();
  const auto &io_mapping = launch_conf.input_output_mapping();
  const auto &lbn2logical_blob_desc = launch_conf.lbn2logical_blob_desc();
  // check input blob descs
  for (const std::string &bn : this->input_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    const std::string &mapping_input = io_mapping.at(blob_name);
    auto it = lbn2logical_blob_desc.find(mapping_input);
    CHECK_OR_RETURN(it != lbn2logical_blob_desc.end());
    CHECK_OR_RETURN(*BlobDesc4BnInOp(bn) == BlobDesc(it->second));
  }
  for (const std::string &bn : this->output_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    const std::string &mapping_output = io_mapping.at(blob_name);
    auto it = lbn2logical_blob_desc.find(mapping_output);
    CHECK_OR_RETURN(it != lbn2logical_blob_desc.end());
    *BlobDesc4BnInOp(bn) = BlobDesc(it->second);
  }
  return Maybe<void>::Ok();
}


Maybe<void> XrtLaunchOp::InferOutBlobDescs(
    std::function<BlobDesc *(const std::string &)> GetBlobDesc4BnInOp,
    const ParallelContext *parallel_ctx, const SbpSignature* sbp_signature) const {
  const auto &launch_conf = op_conf().xrt_launch_conf();
  const auto &io_mapping = launch_conf.input_output_mapping();
  const auto &lbn2logical_blob_desc = launch_conf.lbn2logical_blob_desc();

  // Prepare outer input blob descs
  std::unordered_map<std::string, BlobDesc> blob_descs;
  for (const std::string &bn : this->input_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);
    BlobDesc blob_desc(this->job_desc().DefaultDataType());
    blob_desc.CopyFrom(*GetBlobDesc4BnInOp(bn));

    const std::string &mapping_input = io_mapping.at(blob_name);
    blob_descs.emplace(mapping_input, blob_desc);
  }
  // Build graph from launch conf, and inference output shape.
  {
    // Run InferShape pass
    const auto &sbp_signatures = launch_conf.sbp_signatures();
    auto options = xrt::CreateDefaultXrtPassOptions();
    DeviceType device_type = JUST(DeviceType4DeviceTag(op_conf().device_tag()));
    auto graph =
        xrt::BuildXrtGraph(launch_conf.function(), device_type, this->job_desc());
    const ParallelDesc& op_parallel_desc = *JUST(GetOpParallelDesc());
    xrt::RunXrtPass("InferShape", graph.get(), options, &this->job_desc(), parallel_ctx, &op_parallel_desc,
                    &sbp_signatures, &lbn2logical_blob_desc, &blob_descs);
  }

  // Fetch output blob descs
  for (const std::string &bn : this->output_bns()) {
    const LogicalBlobId &lbi = this->BnInOp2Lbi(bn);
    std::string blob_name = xrt::BlobIdToName(lbi);

    const std::string &mapping_output = io_mapping.at(blob_name);
    CHECK_GT(blob_descs.count(mapping_output), 0);
    *GetBlobDesc4BnInOp(bn) = blob_descs.at(mapping_output);
  }
  return Maybe<void>::Ok();
}

Maybe<void> XrtLaunchOp::InferSbpSignature(
    SbpSignature *sbp_signature, const SbpSignature &sbp_sig_conf,
    const std::function<int32_t(const SbpSignature &)> &CalcOrderValue4SbpSig,
    XrtLaunchOp::SbpInferHint4IbnFunc SbpInferHint4Ibn, const ParallelDesc &parallel_desc) const {
  *sbp_signature = sbp_sig_conf;
  // Check existence of inputs and outputs sbp parallel.
  const auto &bn2sbp_parallel = sbp_sig_conf.bn_in_op2sbp_parallel();
  for (const std::string &bn : this->input_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0)
        << "Input sbp parallel is not found for operator " << this->op_conf().name();
  }
  for (const std::string &bn : this->output_bns()) {
    CHECK_GT(bn2sbp_parallel.count(bn), 0)
        << "Output sbp parallel is not found for operator " << this->op_conf().name();
  }
  return Maybe<void>::Ok();
}

void XrtLaunchOp::VirtualGenKernelConf(
    std::function<const BlobDesc *(const std::string &)> GetBlobDesc4BnInOp,
    const ParallelContext *parallel_ctx, KernelConf *kernel_conf) const {
  *(kernel_conf->mutable_xrt_launch_conf()->mutable_parallel_ctx()) = *parallel_ctx;
}

REGISTER_OP(OperatorConf::kXrtLaunchConf, XrtLaunchOp);

}  // namespace oneflow
