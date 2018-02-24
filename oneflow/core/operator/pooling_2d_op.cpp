#include "oneflow/core/operator/pooling_2d_op.h"

namespace oneflow {

void Pooling2DOp::InitFromOpConf() {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  std::transform(padding_mthd.begin(), padding_mthd.end(), padding_mthd.begin(),
                 ::tolower);
  if (padding_mthd != "same" && padding_mthd != "valid") {
    LOG(FATAL) << "Invalid padding method in " << op_name();
  }
  SetStringInSpecialConf("padding", padding_mthd);

  EnrollInputBn("in");
  EnrollOutputBn("out");
  VirtualEnrollDataTmpBn();
}

void Pooling2DOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->shape().NumAxes(), 4);
  CHECK_EQ(in_blob_desc->data_type(), JobDesc::Singleton()->DefaultDataType());
  // out
  std::tuple<int32_t, int32_t> out_size =
      CalcOutSize(in_blob_desc->shape().At(2), in_blob_desc->shape().At(3));
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() =
      Shape({in_blob_desc->shape().At(0), in_blob_desc->shape().At(1),
             std::get<0>(out_size), std::get<1>(out_size)});

  VirtualInferDataTmpBlobDesc(GetBlobDesc4BnInOp);
}

void Pooling2DOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  if (padding_mthd == "same") {
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    std::tuple<int32_t, int32_t> out_size =
        CalcOutSize(in_blob_desc->shape().At(2), in_blob_desc->shape().At(3));
    const int32_t padding_needed_h =
        (std::get<0>(out_size) - 1) * GetInt32FromSpecialConf("strides_h")
        + GetInt32FromSpecialConf("pool_size_h") - in_blob_desc->shape().At(2);
    const int32_t padding_needed_w =
        (std::get<1>(out_size) - 1) * GetInt32FromSpecialConf("strides_w")
        + GetInt32FromSpecialConf("pool_size_w") - in_blob_desc->shape().At(3);
    Pooling2DKernelConf* pooling_conf = GetMutPooling2DKernelConf(kernel_conf);
    pooling_conf->set_padding_top(padding_needed_h / 2);
    pooling_conf->set_padding_bottom(padding_needed_h - padding_needed_h / 2);
    pooling_conf->set_padding_left(padding_needed_w / 2);
    pooling_conf->set_padding_right(padding_needed_w - padding_needed_w / 2);
  }
}

std::tuple<int32_t, int32_t> Pooling2DOp::CalcOutSize(int32_t in_h,
                                                      int32_t in_w) const {
  int32_t pool_size_h = GetInt32FromSpecialConf("pool_size_h");
  int32_t pool_size_w = GetInt32FromSpecialConf("pool_size_w");
  int32_t strides_h = GetInt32FromSpecialConf("strides_h");
  int32_t strides_w = GetInt32FromSpecialConf("strides_w");
  std::string padding_mthd = GetStringFromSpecialConf("padding");
  int32_t out_h = 0;
  int32_t out_w = 0;
  if (padding_mthd == "valid") {
    out_h = ceil((in_h - pool_size_h + 1.f) / static_cast<float>(strides_h));
    out_w = ceil((in_w - pool_size_w + 1.f) / static_cast<float>(strides_w));
  } else if (padding_mthd == "same") {
    out_h = ceil(in_h / static_cast<float>(strides_h));
    out_w = ceil(in_w / static_cast<float>(strides_w));
  } else {
    UNEXPECTED_RUN();
  }
  return std::make_tuple(out_h, out_w);
}

}  // namespace oneflow
