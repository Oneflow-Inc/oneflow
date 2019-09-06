#include "oneflow/core/operator/prelu_alpha_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluAlphaGradOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_alpha_grad_conf());
  EnrollInputBn("dy", false);
  EnrollInputBn("x", false);
  EnrollOutputBn("alpha_grad", false);
  if (device_type() == DeviceType::kGPU) {
    EnrollTmpBn("bw_buf");
    EnrollTmpBn("alpha_grad_buf");
  }
}

const PbMessage& PReluAlphaGradOp::GetCustomizedConf() const {
  return op_conf().prelu_alpha_grad_conf();
}

Maybe<void> PReluAlphaGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const PReluAlphaGradOpConf& conf = op_conf().prelu_alpha_grad_conf();
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  BlobDesc* alpha_grad_blob_desc = GetBlobDesc4BnInOp("alpha_grad");
  if (conf.channel_shared()) {
    alpha_grad_blob_desc->mut_shape() = Shape({1});
  } else {
    if (conf.data_format() == "channels_first") {
      alpha_grad_blob_desc->mut_shape() = Shape({x_blob_desc->shape().At(1)});
    } else if (conf.data_format() == "channels_last") {
      alpha_grad_blob_desc->mut_shape() =
          Shape({x_blob_desc->shape().At(x_blob_desc->shape().NumAxes() - 1)});
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  alpha_grad_blob_desc->set_data_type(x_blob_desc->data_type());
  if (device_type() == DeviceType::kGPU) {
    BlobDesc* bw_buf_desc = GetBlobDesc4BnInOp("bw_buf");
    BlobDesc* alpha_grad_buf_desc = GetBlobDesc4BnInOp("alpha_grad_buf");
    *alpha_grad_buf_desc = *x_blob_desc;
    if (conf.channel_shared()) {
      *bw_buf_desc = *x_blob_desc;
    } else {
      bw_buf_desc->set_data_type(x_blob_desc->data_type());
      std::vector<int64_t> bw_buf_shape_vec = x_blob_desc->shape().dim_vec();
      if (conf.data_format() == "channels_first") {
        bw_buf_shape_vec[0] = x_blob_desc->shape().At(1);
        bw_buf_shape_vec[1] = x_blob_desc->shape().At(0);
        bw_buf_desc->mut_shape() = Shape(bw_buf_shape_vec);
      } else if (conf.data_format() == "channels_last") {
        bw_buf_shape_vec[0] = x_blob_desc->shape().At(x_blob_desc->shape().NumAxes() - 1);
        bw_buf_shape_vec[x_blob_desc->shape().NumAxes() - 1] = x_blob_desc->shape().At(0);
        bw_buf_desc->mut_shape() = Shape(bw_buf_shape_vec);
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
  }
  return Maybe<void>::Ok();
}

void PReluAlphaGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PReluAlphaGradOpConf& conf = op_conf().prelu_alpha_grad_conf();
  PbRf<int32_t>* perm = kernel_conf->mutable_prelu_alpha_grad_conf()->mutable_perm();
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  int64_t num_axes = x_blob_desc->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, num_axes) { perm->Add(i); }
  if (!conf.channel_shared()) {
    if (conf.data_format() == "channels_first") {
      (*perm)[0] = 1;
      (*perm)[1] = 0;
    } else if (conf.data_format() == "channels_last") {
      (*perm)[num_axes - 1] = 0;
      (*perm)[0] = num_axes - 1;
    } else {
      UNIMPLEMENTED();
    }
  }
}

Maybe<void> PReluAlphaGradOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(*BatchAxis4BnInOp("dy") == *BatchAxis4BnInOp("x"));
  BatchAxis4BnInOp("alpha_grad")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> PReluAlphaGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("dy", 0)
      .Split("x", 0)
      .PartialSum("alpha_grad")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kPreluAlphaGradConf, PReluAlphaGradOp);

}  // namespace oneflow
