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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  const SyncDynamicResizeOpConf& conf = op_conf.sync_dynamic_resize_conf();
  CHECK_EQ_OR_RETURN(conf.axis(), 0);
  const BlobDesc* in = BlobDesc4BnInOp("in");
  const BlobDesc* size = BlobDesc4BnInOp("size");
  CHECK_EQ_OR_RETURN(size->shape().elem_cnt(), 1);
  CHECK_OR_RETURN(IsIntegralDataType(size->data_type()));
  BlobDesc* out = BlobDesc4BnInOp("out");
  *out = *in;
  out->set_is_dynamic(true);
  return Maybe<void>::Ok();
}

}  // namespace

class SyncDynamicResizeOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeOp);
  SyncDynamicResizeOp() = default;
  ~SyncDynamicResizeOp() override = default;

  void InitFromOpConf() override {
    EnrollInputBn("in");
    EnrollInputBn("size", false);
    EnrollOutputBn("out")->set_header_infered_before_compute(false);
  }

  Maybe<void> InferLogicalOutBlobDescs(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature) const override {
    return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    kernel_conf->mutable_sync_dynamic_resize_conf()->set_size_data_type(
        GetBlobDesc4BnInOp("size")->data_type());
  }
};

REGISTER_OP(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeOp);

}  // namespace oneflow
