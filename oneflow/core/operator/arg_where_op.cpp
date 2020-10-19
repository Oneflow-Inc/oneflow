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
#include "oneflow/core/operator/arg_where_op_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class ArgWhereOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgWhereOp);
  ArgWhereOp() = default;
  ~ArgWhereOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_arg_where_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollOutputBn("out_size", false);
    EnrollTmpBn("tmp");
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx,
                                const SbpSignature* sbp_signature,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    const int64_t elem_cnt = in_desc->shape().elem_cnt();
    const DataType out_data_type = op_conf().arg_where_conf().data_type();
    CHECK_LE_OR_RETURN(in_desc->shape().NumAxes(), 8);
    CHECK_OR_RETURN(IsIntegralDataType(out_data_type));
    BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    out_desc->mut_shape() = Shape({elem_cnt, in_desc->shape().NumAxes()});
    out_desc->set_data_type(out_data_type);
    BlobDesc* out_size_desc = GetBlobDesc4BnInOp("out_size");
    out_size_desc->mut_shape() = Shape({1});
    out_size_desc->set_data_type(out_data_type);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    InferOutBlobDescs(GetBlobDesc4BnInOp, parallel_ctx, sbp_signature, EnrollOpCtx);
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    const BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    BlobDesc* tmp_desc = GetBlobDesc4BnInOp("tmp");
    CHECK_NOTNULL_OR_RETURN(tmp_desc);
    int64_t tmp_bytes = 0;
    InferArgWhereWorkspaceSizeInBytes(device_type(), in_desc->data_type(), out_desc->data_type(),
                                      in_desc->shape().NumAxes(), in_desc->shape().elem_cnt(),
                                      &tmp_bytes);
    tmp_desc->mut_shape() = Shape({tmp_bytes});
    tmp_desc->set_data_type(DataType::kChar);
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    const BlobDesc* in_desc = GetBlobDesc4BnInOp("in");
    const BlobDesc* out_desc = GetBlobDesc4BnInOp("out");
    kernel_conf->set_data_type(out_desc->data_type());
    kernel_conf->mutable_arg_where_conf()->set_in_data_type(in_desc->data_type());
    kernel_conf->mutable_arg_where_conf()->set_num_axes(in_desc->shape().NumAxes());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    if (BatchAxis4BnInOp("in")->has_value()) {
      BatchAxis4BnInOp("out")->set_value(0);
    } else {
      BatchAxis4BnInOp("out")->clear_value();
    }
    BatchAxis4BnInOp("out_size")->clear_value();
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kArgWhereConf, ArgWhereOp);

}  // namespace oneflow
