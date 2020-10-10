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

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const SyncDynamicResizeOpConf& conf = op_conf().sync_dynamic_resize_conf();
    CHECK_EQ_OR_RETURN(conf.axis(), 0);
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* size = GetBlobDesc4BnInOp("size");
    CHECK_EQ_OR_RETURN(size->shape().elem_cnt(), 1);
    CHECK_OR_RETURN(IsIntegralDataType(size->data_type()));
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->set_is_dynamic(true);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
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
