#include "oneflow/core/operator/prelu_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {

void PReluOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_conf());
  StrFieldTolower("data_format");
  EnrollInputBn("in");
  EnrollModelBn("alpha");
  EnrollOutputBn("out");
  if (device_type() == DeviceType::kGPU) { EnrollBwBufBn("bw_buf"); }
}

const PbMessage& PReluOp::GetCustomizedConf() const { return op_conf().prelu_conf(); }

void PReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const {
  const PReluOpConf& conf = op_conf().prelu_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc* alpha_blob_desc = GetBlobDesc4BnInOp("alpha");
  if (conf.channel_shared()) {
    alpha_blob_desc->mut_shape() = Shape({1});
  } else {
    if (conf.data_format() == "channels_first") {
      alpha_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(1)});
    } else if (conf.data_format() == "channels_last") {
      alpha_blob_desc->mut_shape() =
          Shape({in_blob_desc->shape().At(in_blob_desc->shape().NumAxes() - 1)});
    } else {
      UNIMPLEMENTED();
    }
  }
  alpha_blob_desc->set_data_type(in_blob_desc->data_type());
}

void PReluOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext*) const {
  if (device_type() == DeviceType::kGPU) {
    BlobDesc* bw_buf_desc = GetBlobDesc4BnInOp("bw_buf");
    if (op_conf().prelu_conf().channel_shared()) {
      *bw_buf_desc = *GetBlobDesc4BnInOp("in");
    } else {
      const PReluOpConf& conf = op_conf().prelu_conf();
      const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
      bw_buf_desc->set_data_type(in_blob_desc->data_type());
      std::vector<int64_t> bw_buf_shape_vec = in_blob_desc->shape().dim_vec();
      if (conf.data_format() == "channels_first") {
        bw_buf_shape_vec[0] = in_blob_desc->shape().At(1);
        bw_buf_shape_vec[1] = in_blob_desc->shape().At(0);
        bw_buf_desc->mut_shape() = Shape(bw_buf_shape_vec);
      } else if (conf.data_format() == "channels_last") {
        bw_buf_shape_vec[0] = in_blob_desc->shape().At(in_blob_desc->shape().NumAxes() - 1);
        bw_buf_shape_vec[in_blob_desc->shape().NumAxes() - 1] = in_blob_desc->shape().At(0);
        bw_buf_desc->mut_shape() = Shape(bw_buf_shape_vec);
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

void PReluOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

void PReluOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const PReluOpConf& conf = op_conf().prelu_conf();
  PbRf<int32_t>* perm = kernel_conf->mutable_prelu_conf()->mutable_perm();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int64_t num_axes = in_blob_desc->shape().NumAxes();
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

REGISTER_OP(OperatorConf::kPreluConf, PReluOp);

}  // namespace oneflow
